from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import tempfile
import os
import asyncio
import requests
import sounddevice as sd
import numpy as np
import wave
from langdetect import detect
import edge_tts
from pydub import AudioSegment
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import lfilter
import aiohttp
import re
from datetime import datetime
from TTS.api import TTS
import uuid
import uvicorn
from groq import Groq


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # TTS Setup
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Mount static directory for audio downloads
app.mount("/static", StaticFiles(directory="static"), name="static")


# Constants
MODEL_NAME = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"

# Configuration
GROQ_API_KEY = ""
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
WHISPER_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
FREESOUND_API_KEY = ""
FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/text/"

client = Groq(api_key=GROQ_API_KEY)



## ========================================================================================================

# Prompt
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
# Define a system message that provides context to the AI chatbot.
SystemChatBot = [
    {"role": "system", "content": PROMPT}
]

# Utilities
async def save_file(file: UploadFile, prefix="file"):
    ext = os.path.splitext(file.filename)[-1]
    path = f"{prefix}_{uuid.uuid4().hex}{ext}"
    async with aiofiles.open(path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    return path

async def convert_audio_to_wav(path):
    audio = AudioSegment.from_file(path)
    wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(wav_temp.name, format="wav")
    return wav_temp.name

async def detect_language_async(text):
    return await asyncio.to_thread(lambda: detect(text))

async def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer

async def query_llm(user_input):
    messages.append({"role": "user", "content": f"{user_input}"})
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot  + messages,
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

        return await AnswerModifier(Answer=Answer)

    except Exception as e:
        print(f"Error: {e}")
        return user_input
    
async def transcribe_llm_audio(audio_path: str):
    # Open the file from the path provided and send it for transcription
    async with aiohttp.ClientSession() as session:
        with open(audio_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename="audio.wav")  # Use file path directly
            form.add_field("model", WHISPER_MODEL)
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            async with session.post(WHISPER_ENDPOINT, headers=headers, data=form) as resp:
                result = await resp.json()
                return result.get("text", "")


async def generate_voice(text, ref_path, lang):
    output = f"cloned_{uuid.uuid4().hex}.wav"
    await asyncio.to_thread(tts.tts_to_file, text=text, file_path=output, speaker_wav=ref_path, language=lang)
    return output

# Endpoints
@app.post("/clone-text/")
async def clone_text(refAudio: UploadFile = File(...), userText: str = Form(...), language: str = Form(...)):
    ref_path = await save_file(refAudio, "ref")
    if not ref_path.endswith(".wav"):
        ref_path = await convert_audio_to_wav(ref_path)
    if await detect_language_async(userText) != language:
        language = "en"
    out_path = await generate_voice(userText, ref_path, language)
    return FileResponse(out_path, media_type="audio/wav")

@app.post("/clone-llm-text/")
async def clone_llm_text(refAudio: UploadFile = File(...), userText: str = Form(...), language: str = Form(...)):
    ref_path = await save_file(refAudio, "ref")
    if not ref_path.endswith(".wav"):
        ref_path = await convert_audio_to_wav(ref_path)
    if await detect_language_async(userText) != language:
        language = "en"
    response = await query_llm(userText)
    out_path = await generate_voice(response, ref_path, language)
    return FileResponse(out_path, media_type="audio/wav")

@app.post("/clone-llm-audio/")
async def clone_llm_audio(refAudio: UploadFile = File(...), audioFileInput: UploadFile = File(...), language: str = Form(...)):
    # Await save_file call as it's an async function
    ref_path = await save_file(refAudio, "ref")
    input_path = await save_file(audioFileInput, "input")

    # Convert files to WAV if not already in WAV format
    if not ref_path.endswith(".wav"):
        ref_path = await convert_audio_to_wav(ref_path)
    if not input_path.endswith(".wav"):
        input_path = await convert_audio_to_wav(input_path)

    # Get transcription of the input audio
    transcription = await transcribe_llm_audio(input_path)

    # Detect language and use default if necessary
    if await detect_language_async(transcription) != language:
        language = "en"

    # Query LLaMA for the response based on the transcription
    response = await query_llm(transcription)

    # Generate the output voice with the generated LLaMA response
    out_path = await generate_voice(response, ref_path, language)

    # Return the generated voice file
    return FileResponse(out_path, media_type="audio/wav")

@app.post("/clone-my-audio/")
async def clone_my_audio(refAudio: UploadFile = File(...), audioFileInput: UploadFile = File(...), language: str = Form(...)):
    # Await save_file call as it's an async function
    ref_path = await save_file(refAudio, "ref")
    input_path = await save_file(audioFileInput, "input")

    # Convert files to WAV if not already in WAV format
    if not ref_path.endswith(".wav"):
        ref_path = await convert_audio_to_wav(ref_path)
    if not input_path.endswith(".wav"):
        input_path = await convert_audio_to_wav(input_path)

    # Get transcription of the input audio
    transcription = await transcribe_llm_audio(input_path)  # Pass the file path here

    # Detect language and use default if necessary
    if await detect_language_async(transcription) != language:
        language = "en"

    # Generate the output voice with the generated transcription
    out_path = await generate_voice(transcription, ref_path, language)

    # Return the generated voice file
    return FileResponse(out_path, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
