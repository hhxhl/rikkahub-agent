package me.rerere.asr.providers

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import me.rerere.asr.ASRController
import me.rerere.asr.ASRProviderSetting
import me.rerere.asr.ASRState
import me.rerere.asr.ASRStatus
import java.util.UUID

/**
 * Android SpeechRecognizer adapter for ordinary dictation and Voice Mode.
 * The selected RecognitionService (for example, Marmalade) owns microphone capture and VAD.
 *
 * One recognizer and one listener are reused for this controller's lifetime. In particular,
 * a successful result does not cause cancel(), destroy(), or immediate re-creation.
 */
class SystemASRController(
    context: Context,
    private val provider: ASRProviderSetting.AndroidSystem,
) : ASRController {
    private val context = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val _state = MutableStateFlow(
        ASRState(isAvailable = SpeechRecognizer.isRecognitionAvailable(this.context))
    )
    override val state: StateFlow<ASRState> = _state.asStateFlow()

    private var recognizer: SpeechRecognizer? = null
    private var onTranscriptChange: ((String) -> Unit)? = null
    private var disposed = false
    private var listening = false
    private var finished = true
    private var sessionToken = 0
    private var turnId = ""
    private var busyRetries = 0
    private var pendingRetry: Runnable? = null

    // SpeechRecognizer callbacks are delivered on the main thread. Keep a single listener
    // instead of replacing the listener whenever a new utterance starts.
    private val listener = object : RecognitionListener {
        private fun isCurrent() = !disposed && listening && !finished

        override fun onReadyForSpeech(params: Bundle?) {
            if (!isCurrent()) return
            _state.update { it.copy(status = ASRStatus.Listening) }
        }

        override fun onBeginningOfSpeech() {
            if (!isCurrent()) return
            _state.update {
                it.copy(
                    status = ASRStatus.Listening,
                    voiceTurn = it.voiceTurn.started(turnId),
                )
            }
        }

        override fun onRmsChanged(rmsdB: Float) {
            if (!isCurrent()) return
            val amplitude = ((rmsdB + 2f) / 12f).coerceIn(0f, 1f)
            _state.update { it.copy(amplitudes = (it.amplitudes + amplitude).takeLast(40)) }
        }

        override fun onBufferReceived(buffer: ByteArray?) = Unit

        override fun onPartialResults(partialResults: Bundle?) {
            if (!isCurrent()) return
            val text = extractText(partialResults)
            // Partial results are a live preview, NOT successful completion.
            onTranscriptChange?.invoke(text)
            _state.update {
                it.copy(
                    status = ASRStatus.Listening,
                    transcript = text,
                    voiceTurn = it.voiceTurn.started(turnId),
                )
            }
            Log.d(TAG, "partial result session=$sessionToken")
        }

        override fun onEndOfSpeech() {
            if (!isCurrent()) return
            // Marmalade may still be doing inference: wait for onResults/onError.
            _state.update {
                it.copy(status = ASRStatus.Listening, voiceTurn = it.voiceTurn.stopped(turnId))
            }
            Log.d(TAG, "end of speech session=$sessionToken")
        }

        override fun onResults(results: Bundle?) {
            if (!isCurrent()) return
            Log.d(TAG, "final result session=$sessionToken")
            complete(extractText(results))
        }

        override fun onError(error: Int) {
            if (!isCurrent()) return
            Log.w(TAG, "recognition error=$error session=$sessionToken")
            when (error) {
                SpeechRecognizer.ERROR_NO_MATCH,
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> complete("")
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> retryBusy()
                else -> {
                    // Preserve real errors, including ERROR_SERVER_DISCONNECTED (11).
                    fail("Android speech recognition failed (code $error)")
                    if (error == SpeechRecognizer.ERROR_SERVER_DISCONNECTED) {
                        // The existing binder is no longer usable. Release it now,
                        // but do NOT automatically destroy/create in a tight loop.
                        releaseRecognizer(cancelFirst = false)
                    }
                }
            }
        }

        override fun onEvent(eventType: Int, params: Bundle?) = Unit
    }

    private fun onMain(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else main.post(block)
    }

    override fun start(onTranscriptChange: (String) -> Unit) {
        onMain {
            if (disposed || listening || pendingRetry != null) return@onMain
            this.onTranscriptChange = onTranscriptChange
            busyRetries = 0
            startSession()
        }
    }

    private fun getRecognizer(): SpeechRecognizer {
        recognizer?.let { return it }
        val created = SpeechRecognizer.createSpeechRecognizer(context)
        try {
            created.setRecognitionListener(listener)
        } catch (e: Exception) {
            created.destroy()
            throw e
        }
        recognizer = created
        return created
    }

    private fun startSession() {
        if (disposed) return
        listening = true
        finished = false
        turnId = UUID.randomUUID().toString()
        ++sessionToken
        _state.value = ASRState(status = ASRStatus.Connecting, isAvailable = true)
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            fail("Android speech recognition service unavailable")
            return
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            if (provider.language.isNotBlank()) {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, provider.language)
            }
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        }

        try {
            getRecognizer().startListening(intent)
            Log.d(TAG, "startListening session=$sessionToken")
        } catch (e: Exception) {
            fail("Unable to start Android speech recognition: ${e.message ?: e.javaClass.simpleName}")
            // This is an exceptional startup failure, not normal utterance cleanup.
            releaseRecognizer(cancelFirst = false)
        }
    }

    private fun extractText(bundle: Bundle?): String =
        bundle?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()

    private fun complete(text: String) {
        if (disposed || finished) return
        finished = true
        listening = false
        onTranscriptChange?.invoke(text)
        _state.update {
            it.copy(
                status = ASRStatus.Idle,
                transcript = text,
                errorMessage = null,
                voiceTurn = it.voiceTurn.stopped(turnId).completed(turnId, text),
            )
        }
        // Do not cancel or destroy a recognizer after a normal final result.
    }

    private fun fail(message: String) {
        if (disposed || finished) return
        finished = true
        listening = false
        _state.update { it.copy(status = ASRStatus.Error, errorMessage = message) }
    }

    private fun retryBusy() {
        if (busyRetries >= MAX_BUSY_RETRIES) {
            fail("Android speech recognition service is busy")
            return
        }
        busyRetries++
        finished = true // Ignore later callbacks from the failed attempt.
        listening = false
        _state.value = ASRState(status = ASRStatus.Connecting, isAvailable = true)
        val expectedToken = ++sessionToken
        val retry = object : Runnable {
            override fun run() {
                if (pendingRetry !== this) return
                pendingRetry = null
                if (!disposed && expectedToken == sessionToken) startSession()
            }
        }
        pendingRetry = retry
        main.postDelayed(retry, BUSY_RETRY_DELAY_MS * busyRetries)
    }

    override fun stop() {
        onMain {
            if (disposed) return@onMain
            if (pendingRetry != null) {
                cancelPendingRetry()
                finished = true
                listening = false
                _state.update { it.copy(status = ASRStatus.Idle) }
                return@onMain
            }
            if (!listening || finished) return@onMain
            _state.update { it.copy(status = ASRStatus.Stopping) }
            try {
                recognizer?.stopListening() // Await onResults or onError.
            } catch (e: Exception) {
                fail("Unable to stop Android speech recognition: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    override fun pauseCapture() {
        // The external service owns recording; cancel() here would lose the final result.
    }

    private fun cancelPendingRetry() {
        pendingRetry?.let { main.removeCallbacks(it) }
        pendingRetry = null
        ++sessionToken
    }

    private fun releaseRecognizer(cancelFirst: Boolean) {
        val old = recognizer ?: return
        recognizer = null
        if (cancelFirst) runCatching { old.cancel() }
        runCatching { old.destroy() }
    }

    override fun dispose() {
        onMain {
            if (disposed) return@onMain
            // A Voice Mode controller is currently disposed after each utterance.
            // Do not cancel a session that has already completed successfully.
            val needsCancel = listening && !finished
            disposed = true
            finished = true
            listening = false
            cancelPendingRetry()
            onTranscriptChange = null
            releaseRecognizer(cancelFirst = needsCancel)
        }
    }

    private companion object {
        const val TAG = "SystemASR"
        const val MAX_BUSY_RETRIES = 2
        const val BUSY_RETRY_DELAY_MS = 700L
    }
}
