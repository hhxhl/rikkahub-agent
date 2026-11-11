package me.rerere.locallm.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-decision-function tests for [LiteRtProvider]'s image-forwarding gate.
 *
 * [decideImageForwarding] guards against a hard native crash. When a vision-capable model is
 * loaded but the device's GPU vision executor fails to initialise, [LiteRtRuntime.ensureLoaded]
 * falls back to a TEXT-ONLY engine (see [LiteRtRuntime.LoadOutcome.visionEnabled] /
 * [LiteRtRuntime.LoadOutcome.visionFellBackToTextOnly]). Forwarding image bytes to an engine
 * that has no vision executor null-derefs inside `liblitertlm_jni.so` -> SIGSEGV. The gate
 * MUST therefore key off the ACTUAL post-load vision state, never the pre-load estimate (which
 * is still `true` at the moment of the doomed first load).
 */
class LiteRtProviderTest {

    @Test
    fun `forwards images when vision is live post-load`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = true,
            userSentImages = true,
        )
        assertTrue("vision live -> forward", d.forwardImages)
        assertFalse("nothing dropped when we forward", d.noteImagesDropped)
    }

    @Test
    fun `does NOT forward images when vision fell back to text-only (the crash case)`() {
        // Vision-capable model, but the GPU vision executor failed and the engine loaded
        // text-only. Forwarding here is the SIGSEGV. Drop the images, and flag a user note.
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = false,
            userSentImages = true,
        )
        assertFalse("vision not live -> never forward", d.forwardImages)
        assertTrue("user attached images that were dropped -> note them", d.noteImagesDropped)
    }

    @Test
    fun `no note when vision fell back but the user attached no images`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = false,
            userSentImages = false,
        )
        assertFalse(d.forwardImages)
        assertFalse("nothing to note when no images were attached", d.noteImagesDropped)
    }

    @Test
    fun `text-only model with a stray image attachment drops silently (no note)`() {
        // The user picked a model that never supported vision. Preserve the prior silent-drop
        // behaviour: no "vision unavailable on this device" note, because the device's vision
        // capability was never the issue here.
        val d = decideImageForwarding(
            modelImageCapable = false,
            visionEnabledPostLoad = false,
            userSentImages = true,
        )
        assertFalse(d.forwardImages)
        assertFalse("text-only model is not a device-vision failure", d.noteImagesDropped)
    }

    @Test
    fun `no images with vision live is a no-op`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = true,
            userSentImages = false,
        )
        assertTrue(d.forwardImages)
        assertFalse(d.noteImagesDropped)
    }
}

/** Prompt budgets share one engine-sized input window rather than fixed maxima. */
class LiteRtToolDeclarationBudgetTest {

    private fun budget(contextTokens: Int, system: Int, last: Int) =
        LiteRtProvider.toolDeclarationCharBudget(contextTokens, system, last)

    @Test
    fun `1280 context no longer has negative tool budget for a new chat`() {
        assertEquals(2304, LiteRtProvider.promptInputCharBudget(1280))
        assertEquals(1152, budget(1280, 537, 2))
    }

    @Test
    fun `tool budget responds to actual system and latest turn`() {
        assertTrue(budget(1280, 537, 2) > budget(1280, 537, 1000))
        assertEquals(0, budget(1280, 1500, 1000))
    }

    @Test
    fun `tool budget is never negative and never takes over half the input`() {
        assertEquals(0, budget(1280, 99999, 99999))
        assertEquals(0, budget(0, 0, 0))
        assertTrue(budget(32768, 0, 0) <= LiteRtProvider.promptInputCharBudget(32768) / 2)
    }

    @Test
    fun `history limit subtracts actual tool declaration cost`() {
        assertEquals(2304 - 537 - 300,
            LiteRtProvider.historyCharBudget(1280, 537, 300))
        assertEquals(0, LiteRtProvider.historyCharBudget(1280, 2304, 300))
    }

    @Test
    fun `all prompt components fit the estimated input allowance`() {
        val engine = LiteRtProvider.engineContextTokens(1280, 1280)
        val system = 537
        val newest = 2
        val toolCost = 300
        assertTrue(toolCost <= budget(engine, system, newest))
        val history = LiteRtProvider.historyCharBudget(engine, system, toolCost)
        assertTrue(system + toolCost + history <= LiteRtProvider.promptInputCharBudget(engine))
        assertTrue(history >= newest)
    }
}

/**
 * The prompt has to be budgeted against the context the ENGINE was configured with, never
 * against the ceiling the model file could theoretically hold. Those are different numbers
 * whenever the catalog allocates less than the file supports, and budgeting against the
 * ceiling let a 32768-token budget build a prompt for an engine holding 4096: on
 * gemma-4-E2B the whole enabled tool set was declared, 31067 chars of it, and the engine
 * rejected the prefill at 7868 tokens.
 */
class LiteRtEngineContextTokensTest {

    @Test
    fun `the allocation wins when the file ceiling is higher`() {
        // gemma-4-E2B: catalog allocates 4096, the file could hold 32768.
        assertEquals(4096, LiteRtProvider.engineContextTokens(4096, 32768))
    }

    @Test
    fun `the ceiling wins when the request is higher`() {
        // qwen3_0_6b_mixed_int4 declares 2048; asking for more faults the native executor.
        assertEquals(2048, LiteRtProvider.engineContextTokens(32000, 2048))
    }

    @Test
    fun `an unknown ceiling leaves the request untouched`() {
        assertEquals(4096, LiteRtProvider.engineContextTokens(4096, null))
    }

    @Test
    fun `the tool budget derived from the engine size never exceeds the engine`() {
        // The regression this guards: budget(ceiling) was handed to an engine sized at the
        // allocation, so the committed prefill could run past what the engine can hold.
        val engine = LiteRtProvider.engineContextTokens(4096, 32768)
        val systemChars = 500
        val latestTurnChars = 300
        val toolBudget = LiteRtProvider.toolDeclarationCharBudget(
            engine, systemChars, latestTurnChars)
        val historyBudget = LiteRtProvider.historyCharBudget(engine, systemChars, toolBudget)
        val committedChars = systemChars + toolBudget + historyBudget
        assertTrue(
            "prefill budget ${committedChars}c must fit the engine's input allowance",
            committedChars <= LiteRtProvider.promptInputCharBudget(engine),
        )
    }
}
