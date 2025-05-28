from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import tempfile
import os
import asyncio
import sounddevice as sd
import numpy as np
import wave
from langdetect import detect, DetectorFactory
from pydub import AudioSegment
import soundfile as sf
import noisereduce as nr
from scipy.signal import lfilter
import aiohttp
import re
from datetime import datetime
import uuid
import uvicorn
from groq import Groq
import logging
import subprocess
import librosa
from TTS.api import TTS
import torch
import warnings
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.tts.models.xtts import XttsArgs

app = FastAPI()

# Directory Configuration
CLONES_DIR = "playground/clones"
REFERENCE_DIR = "playground/reference"

# Create directories if they don't exist
os.makedirs(CLONES_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5501", "http://127.0.0.1:5501", "*"],  # Added specific origins for Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"
WHISPER_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_API_KEY = ""

# Suppress the GenerationMixin warning
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")

# Fix PyTorch 2.6 weights_only issue
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Import required classes for safe globals
try:
    torch.serialization.add_safe_globals([
        BaseDatasetConfig,
        XttsConfig, 
        XttsAudioConfig,
        XttsArgs
    ])
except ImportError as e:
    print(f"Warning: Could not import some TTS classes for safe globals: {e}")

# Fix GenerationMixin inheritance issue
try:
    from transformers import GenerationMixin
    from TTS.tts.layers.xtts.gpt import GPT2InferenceModel
    import TTS.tts.layers.xtts.gpt as gpt_module
    
    class PatchedGPT2InferenceModel(GPT2InferenceModel, GenerationMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def generate(self, *args, **kwargs):
            outputs = super().generate(*args, **kwargs)
            if hasattr(outputs, 'sequences'):
                return outputs.sequences
            elif hasattr(outputs, 'logits'):
                return outputs.logits.argmax(-1)
            else:
                return outputs
    
    gpt_module.GPT2InferenceModel = PatchedGPT2InferenceModel
except ImportError as e:
    print(f"Warning: Could not patch GenerationMixin: {e}")

# Initialize TTS with error handling
def initialize_tts(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False):
    TTS_class = None
    try:
        from TTS import TTS as TTS_class
        print("TTS imported directly from TTS package")
    except ImportError:
        pass
    
    if TTS_class is None:
        try:
            from TTS.api import TTS as TTS_class
            print("TTS imported from TTS.api")
        except ImportError:
            pass
    
    if TTS_class is None:
        try:
            from TTS.utils.synthesizer import Synthesizer
            print("Warning: Using Synthesizer instead of TTS class")
        except ImportError:
            pass
    
    if TTS_class is None:
        raise ImportError("Could not import TTS class from any location")
    
    try:
        return TTS_class(model_name, gpu=gpu)
    except Exception as e:
        print(f"Error initializing TTS with {model_name}: {e}")
        try:
            return TTS_class("tts_models/en/ljspeech/tacotron2-DDC", gpu=gpu)
        except Exception as e2:
            print(f"Fallback TTS also failed: {e2}")
            raise e

try:
    tts = initialize_tts()
    print("TTS initialized successfully!")
except Exception as e:
    print(f"Failed to initialize TTS: {e}")
    print("Please reinstall TTS library: pip uninstall TTS && pip install TTS")
    tts = None

# Prompt and other configurations
PROMPT = """
You are an AI assistant. Your responses must follow strict ethical guidelines. The following categories should not be violated:
- **Sexual Content**: Do not generate or promote any sexual content, except for educational or wellness purposes.
- **Hate and Harassment**: Avoid hate speech, bullying, and harassment of any kind.
- **Self-harm or Suicide**: Do not encourage or promote any form of self-harm, suicide, or dangerous behavior.
- **Violence and Harm**: Avoid promoting violence or graphic harm to individuals or groups.
- **Sexual Reference to Minors**: Avoid any content with sexual references to minors.

If the user's query violates any of these guidelines, respond with:
'Violation of the [category]: [Brief explanation of why the response is violating the guideline].'

IMPORTANT: You are an AI assistant that MUST provide responses in 50 words or less. NO EXCEPTIONS.
CRITICAL RULES:
1. NEVER exceed 100 words in your response unless it is strictly required in exceptional cases.
2. Always give a complete sentence with full context.
3. Answer directly and precisely what is asked.
4. Use simple, clear language appropriate for voice.
5. Maintain polite, professional tone.
6. NEVER provide lists, bullet points, or numbered items.
7. NEVER write multiple paragraphs.
8. NEVER apologize for brevity — embrace it.

ADDITIONAL IMPORTANT RULE:
Before generating a response, check if the language of the query is one of the following supported languages: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, or Hindi.
If the query's language is not supported, respond in English. If the query's language is supported, respond in that language.
REMEMBER: Your responses will be converted to speech. Exactly ONE brief paragraph. Maximum 50 words providing full contextual understanding.
"""
messages = []
SystemChatBot = [{"role": "system", "content": PROMPT}]

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Utilities
async def cleanup_directory(directory):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                print(f"Removing old file: {file_path}")
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up directory {directory}: {e}")

async def save_file(file: UploadFile, prefix="file", directory=REFERENCE_DIR):
    # Clean up old files before saving new ones
    await cleanup_directory(REFERENCE_DIR)
    await cleanup_directory(CLONES_DIR)
    
    ext = os.path.splitext(file.filename)[-1]
    path = os.path.join(directory, f"{prefix}_{uuid.uuid4().hex}{ext}")
    try:
        print(f"Attempting to save file: {file.filename} to {path}")
        async with aiofiles.open(path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        print(f"File saved successfully at: {path}, size: {os.path.getsize(path)} bytes")
        return path
    except Exception as e:
        print(f"Error saving file to {path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

async def convert_audio_to_wav(path):
    try:
        print(f"Converting audio to WAV: {path}")
        audio = AudioSegment.from_file(path)
        wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=REFERENCE_DIR)
        audio.export(wav_temp.name, format="wav")
        print(f"Converted audio saved at: {wav_temp.name}, size: {os.path.getsize(wav_temp.name)} bytes")
        return wav_temp.name
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        raise HTTPException(status_code=500, detail="Failed to convert audio")

async def detect_language_async(text):
    print(f"Detecting language for text: {text[:50]}...")
    try:
        language = await asyncio.to_thread(lambda: detect(text))
        print(f"Detected language: {language}")
        return language
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "en"  # Fallback to English

async def AnswerModifier(Answer):
    print(f"Modifying answer: {Answer[:50]}...")
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer

async def query_llm(user_input):
    messages.append({"role": "user", "content": f"{user_input}"})
    try:
        print(f"Querying LLM with input: {user_input[:50]}...")
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )
        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content
        Answer = Answer.replace("</s>", "")
        modified_answer = await AnswerModifier(Answer=Answer)
        print(f"LLM response: {modified_answer[:50]}...")
        return modified_answer
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return user_input

async def transcribe_llm_audio(audio_path: str):
    try:
        print(f"Transcribing audio from: {audio_path}")
        async with aiohttp.ClientSession() as session:
            with open(audio_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field("file", f, filename="audio.wav")
                form.add_field("model", WHISPER_MODEL)
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                async with session.post(WHISPER_ENDPOINT, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        print(f"Transcription failed with status {resp.status}")
                        raise HTTPException(status_code=500, detail="Transcription failed")
                    result = await resp.json()
                    transcription = result.get("text", "")
                    print(f"Transcription result: {transcription[:50]}...")
                    return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

async def generate_voice(text, ref_path, lang):
    output = os.path.join(CLONES_DIR, f"cloned_{uuid.uuid4().hex}.wav")
    try:
        print(f"Generating voice for text: {text[:50]}... to {output}")
        await asyncio.to_thread(tts.tts_to_file, text=text, file_path=output, speaker_wav=ref_path, language=lang)
        print(f"Voice generated at: {output}, file exists: {os.path.exists(output)}, size: {os.path.getsize(output)} bytes")
        return output
    except Exception as e:
        print(f"Error generating voice: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate voice")

# Endpoint to log frontend messages
@app.post("/log-frontend-message")
async def log_frontend_message(message: dict):
    print(f"Frontend Log: {message.get('message', 'No message provided')}")
    return {"status": "logged"}

# Endpoints
@app.post("/clone-text/")
async def clone_text(refAudio: UploadFile = File(...), userText: str = Form(...), language: str = Form(...)):
    try:
        print(f"Received request for /clone-text/ with userText: {userText[:50]}..., language: {language}")
        ref_path = await save_file(refAudio, "ref", REFERENCE_DIR)
        if not ref_path.endswith(".wav"):
            ref_path = await convert_audio_to_wav(ref_path)
        detected_language = await detect_language_async(userText)
        if detected_language != language:
            language = "en"
            print(f"Language mismatch, defaulting to: {language}")
        out_path = await generate_voice(userText, ref_path, language)
        print(f"Sending FileResponse for: {out_path}")
        return FileResponse(out_path, media_type="audio/wav")
    except Exception as e:
        print(f"Error in /clone-text/: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/clone-llm-text/")
async def clone_llm_text(refAudio: UploadFile = File(...), userText: str = Form(...), language: str = Form(...)):
    try:
        print(f"Received request for /clone-llm-text/ with userText: {userText[:50]}..., language: {language}")
        ref_path = await save_file(refAudio, "ref", REFERENCE_DIR)
        if not ref_path.endswith(".wav"):
            ref_path = await convert_audio_to_wav(ref_path)
        detected_language = await detect_language_async(userText)
        if detected_language != language:
            language = "en"
            print(f"Language mismatch, defaulting to: {language}")
        response = await query_llm(userText)
        out_path = await generate_voice(response, ref_path, language)
        print(f"Sending FileResponse for: {out_path}")
        return FileResponse(out_path, media_type="audio/wav")
    except Exception as e:
        print(f"Error in /clone-llm-text/: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/clone-llm-audio/")
async def clone_llm_audio(refAudio: UploadFile = File(...), audioFileInput: UploadFile = File(...), language: str = Form(...)):
    try:
        print(f"Received request for /clone-llm-audio/ with language: {language}")
        ref_path = await save_file(refAudio, "ref", REFERENCE_DIR)
        input_path = await save_file(audioFileInput, "input", REFERENCE_DIR)
        if not ref_path.endswith(".wav"):
            ref_path = await convert_audio_to_wav(ref_path)
        if not input_path.endswith(".wav"):
            input_path = await convert_audio_to_wav(input_path)
        transcription = await transcribe_llm_audio(input_path)
        detected_language = await detect_language_async(transcription)
        if detected_language != language:
            language = "en"
            print(f"Language mismatch, defaulting to: {language}")
        response = await query_llm(transcription)
        out_path = await generate_voice(response, ref_path, language)
        print(f"Sending FileResponse for: {out_path}")
        return FileResponse(out_path, media_type="audio/wav")
    except Exception as e:
        print(f"Error in /clone-llm-audio/: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/clone-my-audio/")
async def clone_my_audio(refAudio: UploadFile = File(...), audioFileInput: UploadFile = File(...), language: str = Form(...)):
    try:
        print(f"Received request for /clone-my-audio/ with language: {language}")
        ref_path = await save_file(refAudio, "ref", REFERENCE_DIR)
        input_path = await save_file(audioFileInput, "input", REFERENCE_DIR)
        if not ref_path.endswith(".wav"):
            ref_path = await convert_audio_to_wav(ref_path)
        if not input_path.endswith(".wav"):
            input_path = await convert_audio_to_wav(input_path)
        transcription = await transcribe_llm_audio(input_path)
        detected_language = await detect_language_async(transcription)
        if detected_language != language:
            language = "en"
            print(f"Language mismatch, defaulting to: {language}")
        out_path = await generate_voice(transcription, ref_path, language)
        print(f"Sending FileResponse for: {out_path}")
        return FileResponse(out_path, media_type="audio/wav")
    except Exception as e:
        print(f"Error in /clone-my-audio/: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
