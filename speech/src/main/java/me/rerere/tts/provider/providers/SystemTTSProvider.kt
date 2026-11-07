package me.rerere.tts.provider.providers

import android.content.Context
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import me.rerere.common.android.appTempFolder
import me.rerere.tts.model.AudioChunk
import me.rerere.tts.model.AudioFormat
import me.rerere.tts.model.TTSRequest
import me.rerere.tts.provider.TTSProvider
import me.rerere.tts.provider.TTSProviderSetting
import java.io.File
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

private const val TAG = "SystemTTSProvider"
private const val INIT_TIMEOUT_MS = 15_000L
private const val SYNTH_TIMEOUT_MS = 120_000L
private const val IDLE_SHUTDOWN_MS = 10 * 60 * 1_000L

/**
 * System-only TTS implementation. Cloud providers and the shared audio player are unchanged.
 *
 * One TextToSpeech instance is kept warm, one progress listener is registered for that
 * instance, and a mutex serializes synthesizeToFile() calls. The instance is released
 * after a period of inactivity so a changed system TTS engine can be picked up later.
 */
class SystemTTSProvider : TTSProvider<TTSProviderSetting.SystemTTS> {
    private data class PendingSynthesis(
        val id: String,
        val completed: CompletableDeferred<Unit> = CompletableDeferred()
    )

    private val synthesisMutex = Mutex()
    private val pending = AtomicReference<PendingSynthesis?>(null)
    private val idleScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // Access engine and idleShutdownJob only while holding synthesisMutex.
    private var engine: TextToSpeech? = null
    private var idleShutdownJob: Job? = null

    // A single listener per engine: callbacks may arrive on Binder/background threads.
    private val progressListener = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {
            if (utteranceId == pending.get()?.id) {
                Log.d(TAG, "TTS synthesis started: $utteranceId")
            }
        }

        override fun onDone(utteranceId: String?) {
            finish(utteranceId, null)
        }

        @Deprecated("Legacy Android TTS callback")
        override fun onError(utteranceId: String?) {
            finish(utteranceId, IllegalStateException("System TTS synthesis failed"))
        }

        override fun onError(utteranceId: String?, errorCode: Int) {
            finish(utteranceId, IllegalStateException("System TTS error: $errorCode"))
        }

        override fun onStop(utteranceId: String?, interrupted: Boolean) {
            finish(utteranceId, IllegalStateException("System TTS stopped"))
        }
    }

    private fun finish(utteranceId: String?, error: Throwable?) {
        val task = pending.get() ?: return
        if (utteranceId != task.id) return // Ignore callbacks from a cancelled/older request.
        if (error == null) {
            task.completed.complete(Unit)
        } else {
            task.completed.completeExceptionally(error)
        }
    }

    /** Must be called under synthesisMutex. */
    private suspend fun getOrCreateEngine(context: Context): TextToSpeech {
        engine?.let { return it }

        val started = SystemClock.elapsedRealtime()
        val initResult = CompletableDeferred<Int>()

        // Construct on the main thread; the init callback can arrive on another thread.
        val instance = withContext(Dispatchers.Main.immediate) {
            TextToSpeech(context.applicationContext) { status ->
                initResult.complete(status)
            }
        }

        try {
            val status = withTimeout(INIT_TIMEOUT_MS) { initResult.await() }
            check(status == TextToSpeech.SUCCESS) {
                "System TTS initialization failed: $status"
            }
            instance.setOnUtteranceProgressListener(progressListener)
            engine = instance
            Log.i(TAG, "TTS initialized in ${SystemClock.elapsedRealtime() - started}ms")
            return instance
        } catch (t: Throwable) {
            instance.shutdown()
            throw t
        }
    }

    /** Must be called under synthesisMutex. */
    private fun scheduleIdleShutdown() {
        idleShutdownJob?.cancel()
        idleShutdownJob = idleScope.launch {
            delay(IDLE_SHUTDOWN_MS)
            synthesisMutex.withLock {
                // A new synthesis cancels this job before using the engine.
                engine?.shutdown()
                engine = null
                Log.d(TAG, "Idle system TTS engine released")
            }
        }
    }

    override fun generateSpeech(
        context: Context,
        providerSetting: TTSProviderSetting.SystemTTS,
        request: TTSRequest
    ): Flow<AudioChunk> = flow {
        val audioData = synthesisMutex.withLock {
            idleShutdownJob?.cancel()
            idleShutdownJob = null

            try {
                val tts = getOrCreateEngine(context)
                val locale = Locale.getDefault()
                val languageResult = tts.setLanguage(locale)
                if (languageResult == TextToSpeech.LANG_MISSING_DATA ||
                    languageResult == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.w(TAG, "Language $locale not supported by system TTS engine")
                }
                tts.setSpeechRate(providerSetting.speechRate)
                tts.setPitch(providerSetting.pitch)

                val utteranceId = UUID.randomUUID().toString()
                val audioFile = File(context.appTempFolder, "tts_$utteranceId.wav")
                val task = PendingSynthesis(utteranceId)
                val started = SystemClock.elapsedRealtime()
                pending.set(task)

                try {
                    val result = tts.synthesizeToFile(
                        request.text,
                        null,
                        audioFile,
                        utteranceId
                    )
                    check(result == TextToSpeech.SUCCESS) {
                        "Failed to queue system TTS synthesis"
                    }

                    // No audio is played here: the existing ExoPlayer receives the WAV bytes.
                    withTimeout(SYNTH_TIMEOUT_MS) { task.completed.await() }
                    check(audioFile.isFile && audioFile.length() > 0L) {
                        "System TTS output WAV is missing or empty"
                    }
                    val bytes = audioFile.readBytes()
                    Log.i(
                        TAG,
                        "WAV synthesis=${SystemClock.elapsedRealtime() - started}ms, " +
                            "bytes=${bytes.size}, textLength=${request.text.length}"
                    )
                    bytes
                } catch (t: Throwable) {
                    // Stop only the failing/cancelled request; keep the initialized engine
                    // for the next healthy request. The mutex prevents stop/new work races.
                    runCatching { tts.stop() }
                    throw t
                } finally {
                    pending.compareAndSet(task, null)
                    audioFile.delete()
                }
            } finally {
                scheduleIdleShutdown()
            }
        }

        emit(
            AudioChunk(
                data = audioData,
                format = AudioFormat.WAV,
                isLast = true,
                metadata = mapOf(
                    "provider" to "system",
                    "speechRate" to providerSetting.speechRate.toString(),
                    "pitch" to providerSetting.pitch.toString()
                )
            )
        )
    }
}
