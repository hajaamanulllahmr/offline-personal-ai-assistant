import os
import sys
import time
import queue
import threading
import numpy as np
import pyaudio
import ollama
from faster_whisper import WhisperModel
import pyttsx3

# Configuration
MODEL_NAME = "gemma3:4b"
WHISPER_MODEL_SIZE = "large-v3" 
WAKE_WORD = "assistant"

class OfflineAssistant:
    def __init__(self):
        print("Initializing Offline AI Assistant...")
        
        # Initialize TTS Engine
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        # Safety check for voice selection
        if len(self.voices) > 1:
            self.engine.setProperty('voice', self.voices[1].id) 
        
        self.engine.setProperty('rate', 180)
        
        # Initialize STT (Faster-Whisper)
        print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
        # Run on CPU with int8 quantization for speed
        self.stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        
        # Audio constants
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.p = pyaudio.PyAudio()
        
        # State Management
        self.is_speaking = False
        self.stop_event = threading.Event()

    def speak(self, text):
        """Convert text to speech without locking the main thread permanently."""
        if not text: return
        print(f"Assistant: {text}")
        
        self.is_speaking = True
        try:
            # Note: pyttsx3 can be finicky in threads. 
            # If it hangs, consider running this in a separate process.
            self.engine.say(text)
            self.engine.runAndWait()
        finally:
            self.is_speaking = False

    def get_ollama_response(self, prompt):
        """Get response from Ollama."""
        try:
            response = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': 'You are a helpful offline personal AI assistant. Be concise.'},
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        except Exception as e:
            return f"I'm having trouble reaching the brain: {str(e)}"

    def listen_and_transcribe(self):
        """Continuously capture audio and transcribe."""
        stream = self.p.open(format=self.format,
                             channels=self.channels,
                             rate=self.rate,
                             input=True,
                             frames_per_buffer=self.chunk_size)
        
        print("\n--- Assistant is Listening ---")
        
        frames = []
        silent_chunks = 0
        speech_started = False
        
        try:
            while not self.stop_event.is_set():
                # Don't listen to yourself speaking
                if self.is_speaking:
                    time.sleep(0.5)
                    continue
                
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    # Convert to float32 for Whisper
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    rms = np.sqrt(np.mean(audio_data**2))
                    
                    if rms > 0.012: # Slightly higher threshold to avoid background hum
                        if not speech_started:
                            print("Listening...")
                            speech_started = True
                        frames.append(audio_data)
                        silent_chunks = 0
                    elif speech_started:
                        frames.append(audio_data)
                        silent_chunks += 1
                        
                    # If silence for ~1.2 seconds (roughly 18 chunks at 16k rate)
                    if speech_started and silent_chunks > 18:
                        print("Processing voice...")
                        full_audio = np.concatenate(frames)
                        
                        # Transcribe using Faster-Whisper
                        segments, _ = self.stt_model.transcribe(full_audio, beam_size=5)
                        text = " ".join([segment.text for segment in segments]).strip()
                        
                        if text:
                            print(f"You: {text}")
                            self.handle_input(text)
                        
                        # Reset for next phrase
                        frames = []
                        speech_started = False
                        silent_chunks = 0
                except IOError:
                    # Handle buffer overflow
                    pass
        finally:
            stream.stop_stream()
            stream.close()

    def handle_input(self, text):
        """Logic for acting on commands."""
        text_lower = text.lower()
        
        if any(cmd in text_lower for cmd in ["exit", "stop", "quit", "goodbye"]):
            self.speak("Goodbye!")
            self.stop_event.set()
            return

        response = self.get_ollama_response(text)
        self.speak(response)

    def run(self):
        """Start the assistant."""
        self.speak("System online. How can I help?")
        try:
            self.listen_and_transcribe()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_event.set()
            self.p.terminate()

if __name__ == "__main__":
    assistant = OfflineAssistant()
    assistant.run()