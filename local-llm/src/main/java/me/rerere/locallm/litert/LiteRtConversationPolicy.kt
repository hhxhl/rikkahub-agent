package me.rerere.locallm.litert

/**
 * Conservative preflight for LiteRT's small, fixed KV windows. This is NOT a tokenizer:
 * ASCII/JSON generally uses more tokens than length / 4, while CJK often needs one or
 * more tokens per character. Always leave generation and chat-template headroom.
 *
 * The provider calls this on the full cold render, even if the next call may be warm,
 * so trimming a conversation can never rely on a smaller single-turn input estimate.
 */
internal object LiteRtConversationPolicy {
    /** Half the engine window is kept for generation; another 96 tokens (or less for
     * tiny contexts) cover SDK templates, role delimiters and tokenizer estimation error. */
    fun inputTokenBudget(contextTokens: Int): Int {
        val tokens = contextTokens.coerceAtLeast(0)
        return (tokens / 2 - minOf(96, tokens / 12)).coerceAtLeast(0)
    }

    /** ASCII: ~3 chars/token; non-ASCII UTF-16 units: ~1 token/unit. Intentional
     * overestimates for most English and CJK prompts, but does not guarantee exact count. */
    fun estimateTextTokens(text: String): Int {
        var ascii = 0L
        var nonAscii = 0L
        for (c in text) {
            if (c.code <= 0x7f) ascii++ else nonAscii++
        }
        return ((ascii + 2) / 3 + nonAscii)
            .coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    }

    fun fits(
        contextTokens: Int,
        systemText: String,
        toolJsons: List<String>,
        historyText: String,
        historyTurns: Int,
    ): Boolean {
        // The cold path puts the history in one native user message, but the warm path
        // holds native role-wrapped turns. Budget for the more expensive warm structure.
        val estimated = 16L + historyTurns.toLong() * 8 + toolJsons.size.toLong() * 8 +
            estimateTextTokens(systemText) + estimateTextTokens(historyText) +
            toolJsons.sumOf { estimateTextTokens(it).toLong() }
        return estimated <= inputTokenBudget(contextTokens).toLong()
    }

    /** Keep the longest recent suffix that fits, never discarding the latest turn.
     * An assistant answer may be retained even when its older user question does not fit:
     * discarding both would lose useful immediate context in a 1280-token window.
     */
    fun <T> trimOldestTurns(
        turns: List<T>,
        fits: (List<T>) -> Boolean,
    ): List<T> {
        var kept = turns
        while (kept.size > 1 && !fits(kept)) {
            kept = kept.drop(1)
        }
        return kept
    }
}

/**
 * The SDK receives the entire cold history as ONE user message. Label the turns as
 * quoted context, but do NOT add a trailing "Assistant:" prompt: the SDK's own chat
 * template already introduces the assistant role, and two cues can confuse continuations.
 */
internal fun renderLiteRtColdHistory(turns: List<Turn>): String = when (turns.size) {
    0 -> ""
    1 -> turns[0].rawText
    else -> buildString {
        for ((index, turn) in turns.withIndex()) {
            if (index != 0) append('\n')
            append(if (turn.role == ROLE_USER) "User: " else "Assistant: ")
            append(turn.rawText)
        }
    }
}
