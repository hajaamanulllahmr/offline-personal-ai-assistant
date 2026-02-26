# Offline Personal AI Assistant

This is a high-performance, 100% offline personal AI assistant built with Python.

## Features
- **Brain:** Ollama Gemma 3:4b (State-of-the-art local LLM).
- **Speech-to-Text (STT):** Faster-Whisper (Extremely fast and accurate offline transcription).
- **Text-to-Speech (TTS):** pyttsx3 (Uses your computer's built-in voices).
- **Voice Command:** Continuous listening with voice activity detection.

## Prerequisites
1. **Ollama:** Ensure Ollama is installed and running.
   - Download from [ollama.com](https://ollama.com)
   - Pull the model: `ollama pull gemma3:4b`
2. **Python 3.9+** (Tested on Python 3.9)
3. **Microphone:** A working microphone for voice commands.

## Setup
1. Open a terminal in this directory.
2. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Assistant
Simply run the following command:
```bash
python assistant.py
```

## How to Use
- Once started, the assistant will greet you.
- Just speak! The assistant uses Voice Activity Detection to know when you are talking.
- When you stop talking for about 1.5 seconds, it will process your request using Gemma 3.
- Say "Exit", "Stop", or "Goodbye" to turn it off.

## Troubleshooting
- **No Sound:** Ensure your default microphone is set correctly in Windows.
- **Slow Response:** 
  - Change `WHISPER_MODEL_SIZE` in `assistant.py` to `"tiny.en"` for faster STT.
  - Ensure you have enough RAM for Gemma 3:4b (approx 4-8GB free).
