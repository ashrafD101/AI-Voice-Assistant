#!/usr/bin/env python3
"""
Voice-enabled AI assistant powered by ElevenLabs.

Two modes:

  1. agent   -> ElevenLabs Agents Platform (Conversational AI).
                ElevenLabs handles speech recognition, LLM response generation,
                turn-taking/interruptions and voice synthesis in one realtime
                session. Least code, lowest latency.

  2. custom  -> A pipeline you control, step by step:
                Microphone -> ElevenLabs Scribe (speech-to-text)
                           -> LLM (Claude) for the response
                           -> ElevenLabs Text-to-Speech -> Speakers

Setup
-----
    pip install "elevenlabs[pyaudio]" anthropic sounddevice numpy python-dotenv
    # PyAudio needs PortAudio:  macOS: brew install portaudio
    #                           Debian/Ubuntu: sudo apt install portaudio19-dev

    Create a .env file next to this script:

        ELEVENLABS_API_KEY=your_elevenlabs_key
        ELEVENLABS_AGENT_ID=your_agent_id        # agent mode only
        ANTHROPIC_API_KEY=your_anthropic_key     # custom mode only

Run
---
    python voice_assistant.py agent
    python voice_assistant.py custom
    python voice_assistant.py custom --voice-id <id> --threshold 600
"""

import argparse
import io
import os
import queue
import signal
import sys
import wave
from collections import deque

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # fine if variables are already in the environment


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")

# Custom pipeline settings (all overridable via environment variables)
STT_MODEL = os.getenv("STT_MODEL", "scribe_v1")
TTS_MODEL = os.getenv("TTS_MODEL", "eleven_flash_v2_5")  # low latency
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # "George"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. Your replies are spoken aloud, "
    "so answer in one to three short sentences of plain conversational speech. "
    "Never use markdown, bullet points, emojis, or code blocks."
)

EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "stop", "good bye"}

# Audio settings
MIC_RATE = 16000          # Hz, good for speech recognition
CHUNK_MS = 30
CHUNK_SAMPLES = MIC_RATE * CHUNK_MS // 1000
TTS_RATE = 22050          # matches output_format "pcm_22050"
MAX_HISTORY_TURNS = 10


# --------------------------------------------------------------------------- #
# MODE 1: ElevenLabs Agents Platform (all-in-one)
# --------------------------------------------------------------------------- #
def run_agent_mode() -> None:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.conversation import Conversation
    from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

    if not ELEVENLABS_AGENT_ID:
        sys.exit(
            "Missing ELEVENLABS_AGENT_ID. Create an agent in the ElevenLabs "
            "dashboard (Agents) and put its ID in your .env file."
        )

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    conversation = Conversation(
        client,
        ELEVENLABS_AGENT_ID,
        # Only required for private agents; harmless for public ones.
        requires_auth=bool(ELEVENLABS_API_KEY),
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=lambda text: print(f"\nAssistant: {text}"),
        callback_agent_response_correction=lambda original, corrected: print(
            f"Assistant (interrupted): {corrected}"
        ),
        callback_user_transcript=lambda text: print(f"You: {text}"),
    )

    # Ctrl+C ends the session cleanly
    signal.signal(signal.SIGINT, lambda *_: conversation.end_session())

    print("Listening... speak to the assistant. Press Ctrl+C to end.\n")
    conversation.start_session()
    conversation_id = conversation.wait_for_session_end()
    print(f"\nSession ended. Conversation ID: {conversation_id}")


# --------------------------------------------------------------------------- #
# MODE 2: Custom pipeline (Scribe STT -> Claude -> ElevenLabs TTS)
# --------------------------------------------------------------------------- #
class VoiceAssistant:
    def __init__(self, voice_id: str, threshold: float, silence_secs: float):
        import anthropic
        from elevenlabs.client import ElevenLabs

        if not ELEVENLABS_API_KEY:
            sys.exit("Missing ELEVENLABS_API_KEY.")
        if not os.getenv("ANTHROPIC_API_KEY"):
            sys.exit("Missing ANTHROPIC_API_KEY (used to generate responses).")

        self.eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.llm = anthropic.Anthropic()
        self.voice_id = voice_id
        self.threshold = threshold
        self.silence_secs = silence_secs
        self.history: list[dict] = []

    # ---- 1. Listen ------------------------------------------------------- #
    def record_utterance(self, start_timeout: float = 15.0, max_secs: float = 30.0):
        """Record from the mic until the speaker pauses. Returns WAV bytes or None."""
        import numpy as np
        import sounddevice as sd

        q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(indata.copy())

        pre_roll = deque(maxlen=10)  # ~300 ms kept so first syllable isn't clipped
        frames: list = []
        speaking = False
        silent_chunks = 0
        waited_chunks = 0
        silence_limit = int(self.silence_secs * 1000 / CHUNK_MS)
        start_limit = int(start_timeout * 1000 / CHUNK_MS)
        max_chunks = int(max_secs * 1000 / CHUNK_MS)

        with sd.InputStream(
                samplerate=MIC_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
                callback=callback,
        ):
            while True:
                chunk = q.get()
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

                if not speaking:
                    pre_roll.append(chunk)
                    waited_chunks += 1
                    if rms > self.threshold:
                        speaking = True
                        frames.extend(pre_roll)
                    elif waited_chunks > start_limit:
                        return None  # nobody spoke
                else:
                    frames.append(chunk)
                    silent_chunks = silent_chunks + 1 if rms < self.threshold else 0
                    if silent_chunks > silence_limit or len(frames) > max_chunks:
                        break

        audio = np.concatenate(frames, axis=0)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(MIC_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    # ---- 2. Speech recognition (ElevenLabs Scribe) ----------------------- #
    def transcribe(self, wav_bytes: bytes) -> str:
        result = self.eleven.speech_to_text.convert(
            file=io.BytesIO(wav_bytes),
            model_id=STT_MODEL,
            tag_audio_events=False,  # skip "(laughter)" style tags
        )
        return (result.text or "").strip()

    # ---- 3. Response generation (LLM) ------------------------------------ #
    def think(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-MAX_HISTORY_TURNS * 2:]

        response = self.llm.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )
        reply = "".join(b.text for b in response.content if b.type == "text").strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # ---- 4. Audio creation (ElevenLabs TTS) + playback -------------------- #
    def speak(self, text: str) -> None:
        import numpy as np
        import sounddevice as sd

        audio_stream = self.eleven.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=TTS_MODEL,
            output_format=f"pcm_{TTS_RATE}",  # raw 16-bit PCM, no codec needed
        )
        pcm = b"".join(audio_stream)
        samples = np.frombuffer(pcm, dtype=np.int16)
        sd.play(samples, samplerate=TTS_RATE)
        sd.wait()

    # ---- Main loop --------------------------------------------------------- #
    def run(self) -> None:
        print("Voice assistant ready. Say 'goodbye' or press Ctrl+C to quit.\n")
        self.speak("Hi! How can I help you today?")

        while True:
            print("Listening...")
            wav = self.record_utterance()
            if wav is None:
                continue  # silence, keep listening

            try:
                user_text = self.transcribe(wav)
            except Exception as e:  # network/API hiccup shouldn't kill the session
                print(f"[STT error] {e}")
                continue

            if not user_text:
                continue
            print(f"You: {user_text}")

            if user_text.lower().strip(" .!?,") in EXIT_PHRASES:
                self.speak("Goodbye!")
                break

            try:
                reply = self.think(user_text)
                print(f"Assistant: {reply}\n")
                self.speak(reply)
            except Exception as e:
                print(f"[Response error] {e}")


def run_custom_mode(args) -> None:
    assistant = VoiceAssistant(
        voice_id=args.voice_id,
        threshold=args.threshold,
        silence_secs=args.silence,
    )
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nBye!")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="ElevenLabs voice assistant")
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("agent", help="Use an ElevenLabs Agent (all-in-one, realtime)")

    custom = sub.add_parser("custom", help="Scribe STT -> Claude -> ElevenLabs TTS")
    custom.add_argument("--voice-id", default=TTS_VOICE_ID, help="ElevenLabs voice ID")
    custom.add_argument(
        "--threshold",
        type=float,
        default=500.0,
        help="Mic loudness (RMS) that counts as speech. Raise in noisy rooms.",
    )
    custom.add_argument(
        "--silence",
        type=float,
        default=1.2,
        help="Seconds of silence that end your turn.",
    )

    args = parser.parse_args()
    if args.mode == "agent":
        run_agent_mode()
    else:
        run_custom_mode(args)


if __name__ == "__main__":
    main()