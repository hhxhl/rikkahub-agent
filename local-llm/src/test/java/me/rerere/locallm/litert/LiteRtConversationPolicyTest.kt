package me.rerere.locallm.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtConversationPolicyTest {
    @Test
    fun `1280 window leaves output and template margin`() {
        assertEquals(544, LiteRtConversationPolicy.inputTokenBudget(1280))
        assertEquals(0, LiteRtConversationPolicy.inputTokenBudget(0))
    }

    @Test
    fun `CJK characters are not treated as four characters per token`() {
        assertEquals(4, LiteRtConversationPolicy.estimateTextTokens("你好世界"))
        assertEquals(4, LiteRtConversationPolicy.estimateTextTokens("abcdefghijkl"))
        assertEquals(2, LiteRtConversationPolicy.estimateTextTokens("😀"))
    }

    @Test
    fun `short prompt fits and large Chinese message is rejected`() {
        assertTrue(LiteRtConversationPolicy.fits(1280, "You are helpful.", emptyList(), "你好", 1))
        assertFalse(LiteRtConversationPolicy.fits(
            1280, "S".repeat(537), emptyList(), "你好".repeat(300), 1,
        ))
    }

    @Test
    fun `tool declarations must share tokens with history`() {
        val system = "S".repeat(500)
        val history = "你好".repeat(130)
        assertTrue(LiteRtConversationPolicy.fits(1280, system, emptyList(), history, 2))
        assertFalse(LiteRtConversationPolicy.fits(
            1280, system, listOf("a".repeat(900)), history, 2,
        ))
    }

    @Test
    fun `trimming drops oldest complete exchange and keeps latest`() {
        val turns = listOf(
            Turn(ROLE_USER, "old-user"),
            Turn(ROLE_ASSISTANT, "old-answer"),
            Turn(ROLE_USER, "new-user"),
            Turn(ROLE_ASSISTANT, "new-answer"),
            Turn(ROLE_USER, "current-question"),
        )
        val trimmed = LiteRtConversationPolicy.trimOldestTurns(
            turns, fits = { it.size <= 3 },
        )
        assertEquals(turns.takeLast(3), trimmed)
        assertEquals("current-question", trimmed.last().rawText)
    }

    @Test
    fun `short prior assistant answer remains when older user question is too large`() {
        val turns = listOf(
            Turn(ROLE_USER, "a".repeat(2000)),
            Turn(ROLE_ASSISTANT, "It was blue"),
            Turn(ROLE_USER, "Which color did you say?"),
        )
        val kept = LiteRtConversationPolicy.trimOldestTurns(
            turns, fits = { renderLiteRtColdHistory(it).length < 200 },
        )
        assertEquals(turns.takeLast(2), kept)
        assertTrue(renderLiteRtColdHistory(kept).contains("It was blue"))
    }

    @Test
    fun `oversized single newest request is left for explicit rejection`() {
        val turns = listOf(Turn(ROLE_USER, "old"), Turn(ROLE_USER, "x".repeat(2000)))
        val trimmed = LiteRtConversationPolicy.trimOldestTurns(
            turns, fits = { false },
        )
        assertEquals(listOf(turns.last()), trimmed)
        assertFalse(LiteRtConversationPolicy.fits(1280, "S".repeat(500),
            emptyList(), trimmed.last().rawText, 1))
    }

    @Test
    fun `cold rebuild has explicit roles but no duplicate final assistant cue`() {
        val turns = listOf(
            Turn(ROLE_USER, "Remember the color blue"),
            Turn(ROLE_ASSISTANT, "I will"),
            Turn(ROLE_USER, "What is the color?"),
        )
        val rendered = renderLiteRtColdHistory(turns)
        assertEquals("User: Remember the color blue\nAssistant: I will\nUser: What is the color?", rendered)
        assertFalse(rendered.endsWith("Assistant: "))
        assertEquals("hello", renderLiteRtColdHistory(listOf(Turn(ROLE_USER, "hello"))))
    }
}
