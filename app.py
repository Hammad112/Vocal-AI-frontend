from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
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
import librosa  # Added missing import
from TTS.api import TTS
import torch
from TTS.tts.models.xtts import XttsArgs
import warnings


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Constants and Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"
WHISPER_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/text/"
VOICES = {
    "en": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
    "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
}
SAVE_DIR = "processed_audio"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("static/sounds", exist_ok=True)

# In-memory task store for SFX (use Redis/DB in production)
sfx_tasks = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

# Transcription with Groq API
async def transcribe_audio_groq(audio_path: str, detect_language: bool = True) -> dict:
    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                WHISPER_ENDPOINT,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": audio_file},
                data={"model": WHISPER_MODEL},
            )
            response.raise_for_status()
            transcription = response.json().get("text", "")
            language = detect(transcription) if detect_language and transcription else None
            return {"transcription": transcription, "language": language}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error transcribing audio: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_audio_groq: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


from pydub.utils import mediainfo
from pydub import AudioSegment
import traceback
import shutil


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Allow .wav, .mp3, and .webm extensions
    if not file.filename.lower().endswith((".wav", ".mp3", ".webm")):
        raise HTTPException(status_code=400, detail="Invalid file format. Use WAV, MP3, or WebM.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
    temp_file_path = temp_file.name
    temp_file.close()
    wav_file_path = temp_file_path + ".wav"

    # Save a copy for debugging
    debug_file_path = os.path.join("debug_uploads", file.filename)
    os.makedirs("debug_uploads", exist_ok=True)

    try:
        # Write uploaded file
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            await out_file.flush()
            if os.path.getsize(temp_file_path) == 0:
                logger.error("Temporary file is empty")
                raise HTTPException(status_code=500, detail="Failed to write audio file")

        # Save a copy for debugging
        shutil.copy(temp_file_path, debug_file_path)
        logger.info(f"Saved uploaded file to {debug_file_path}")

        # Convert to WAV
        try:
            audio = AudioSegment.from_file(temp_file_path)
            audio.export(wav_file_path, format="wav")
            logger.info(f"Converted uploaded file to {wav_file_path}")
        except Exception as e:
            logger.error(f"Conversion to WAV failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail="Failed to convert audio to WAV")

        # Validate audio file
        try:
            audio_info = mediainfo(wav_file_path)
            logger.info(f"Mediainfo output: {audio_info}")
            if not audio_info.get('format_name', '').lower() == 'wav':
                raise ValueError("Invalid audio file content after conversion")
        except Exception as e:
            logger.warning(f"Mediainfo failed: {e}, trying librosa")
            try:
                librosa.load(wav_file_path, sr=None)
            except Exception as le:
                logger.error(f"Librosa validation failed: {le}\n{traceback.format_exc()}")
                raise HTTPException(status_code=400, detail="Invalid or corrupted audio file")

        result = await transcribe_audio_groq(wav_file_path)
        return result
    finally:
        for path in [temp_file_path, wav_file_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except PermissionError as e:
                    logger.warning(f"Could not delete file {path}: {e}")

# Text-to-Speech
async def text_to_speech(text: str, output_file: str, gender: str = "male", speed: float = 1.25):
    try:
        detected_lang = detect(text)
        voice = VOICES.get(detected_lang, VOICES["en"]).get(gender.lower(), VOICES["en"]["male"])
        temp_output = "temp_tts.mp3"

        tts = edge_tts.Communicate(text, voice)
        await tts.save(temp_output)

        audio = AudioSegment.from_file(temp_output, format="mp3")
        faster_audio = audio.speedup(playback_speed=speed)
        faster_audio.export(output_file, format="mp3")

        return output_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_speech(text: str = Form(...), gender: str = Form("male")):
    output_file = "output_speech.mp3"
    await text_to_speech(text, output_file, gender=gender)
    return {"message": "Speech generated", "file": output_file}

@app.get("/download")
async def download_speech():
    return FileResponse("output_speech.mp3", media_type="audio/mpeg", filename="speech.mp3")

# Voice Effect Processing
async def change_voice(input_file: str, mode: str):
    try:
        # Load audio with soundfile
        y, sr = sf.read(input_file)
        logger.info(f"Loaded audio file {input_file} with sample rate {sr}")

        # Noise reduction
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        logger.info("Applied noise reduction")

        # Apply effects
        if mode == "robot":
            y_shifted = librosa.effects.pitch_shift(y_denoised, sr=sr, n_steps=3)
            mod_freq = 5
            t = np.linspace(0, len(y_shifted) / sr, num=len(y_shifted))
            tremolo = 0.8 + 0.2 * np.sin(2 * np.pi * mod_freq * t)
            y_final = y_shifted * tremolo
        elif mode == "alien":
            y_shifted = librosa.effects.pitch_shift(y_denoised, sr=sr, n_steps=8)
            y_final = np.sin(2 * np.pi * 0.1 * np.arange(len(y_shifted))) * y_shifted
        else:
            raise HTTPException(status_code=400, detail=f"Invalid effect: {mode}. Use 'robot' or 'alien'.")

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(SAVE_DIR, f"processed_{mode}_{timestamp}.wav")
        sf.write(output_filename, y_final, sr)
        logger.info(f"Successfully wrote processed audio to {output_filename}")

        # Verify the file exists
        if not os.path.exists(output_filename):
            raise RuntimeError(f"Failed to create processed file at {output_filename}")

        return output_filename
    except Exception as e:
        logger.error(f"Error processing audio in change_voice: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

async def cleanup_file(file_path: str, delay: int):
    try:
        await asyncio.sleep(delay)
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up file {file_path}: {e}")

@app.post("/process")
async def process_audio(background_tasks: BackgroundTasks, effect: str = Form(...), audio: UploadFile = File(...)):
    if not audio.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid file format. Use WAV or MP3.")

    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input_path = temp_input.name
    temp_input.close()

    processed_file = None
    try:
        # Read the uploaded file content
        content = await audio.read()
        logger.info(f"Received audio file: {audio.filename}, size: {len(content)} bytes")

        # Write content to temporary file
        async with aiofiles.open(temp_input_path, 'wb') as out_file:
            await out_file.write(content)
            await out_file.flush()  # Ensure file is written to disk

        # Convert to WAV using pydub with explicit format detection
        try:
            file_format = audio.filename.split('.')[-1].lower()
            if file_format not in ['wav', 'mp3']:
                raise ValueError(f"Unsupported format: {file_format}")
            audio_segment = AudioSegment.from_file(temp_input_path, format=file_format)
            audio_segment.export(temp_input_path, format="wav")
            logger.info(f"Successfully converted audio to WAV at {temp_input_path}")
        except Exception as e:
            logger.error(f"Failed to convert audio to WAV with pydub: {e}")
            # Fallback to ffmpeg for conversion
            try:
                converted_path = temp_input_path + ".converted.wav"
                subprocess.run(["ffmpeg", "-i", temp_input_path, "-f", "wav", converted_path],
                              capture_output=True, text=True, check=False)
                if os.path.exists(converted_path):
                    os.replace(converted_path, temp_input_path)
                    logger.info("Recovered audio conversion with ffmpeg")
                else:
                    raise Exception("FFmpeg conversion failed")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error: {e.stderr}")
                raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e.stderr}")

        processed_file = await change_voice(temp_input_path, effect)
        if not processed_file or not os.path.exists(processed_file):
            raise HTTPException(status_code=500, detail="Processed file not created")

        # Schedule cleanup after 5 minutes (300 seconds)
        background_tasks.add_task(cleanup_file, processed_file, delay=300)
        background_tasks.add_task(cleanup_file, temp_input_path, delay=300)

        return FileResponse(
            processed_file,
            media_type="audio/wav",
            filename=os.path.basename(processed_file)
        )
    except Exception as e:
        logger.error(f"Error in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")
    finally:
        # Do not delete files here; handled by background task
        pass

# Sound Effect Generation
async def sanitize_filename(text):
    words = re.findall(r'\b\w+\b', text)
    selected = "_".join(words[:2])
    return selected[:15]

async def fetch_sound_url(query):
    params = {
        "query": query,
        "token": FREESOUND_API_KEY,
        "fields": "id,previews"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(FREESOUND_SEARCH_URL, params=params) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch sound URL for query '{query}': HTTP {response.status}")
                return None
            data = await response.json()
            if data["results"]:
                sound_id = data["results"][0]["id"]
                preview_url = data["results"][0]["previews"]["preview-hq-mp3"]
                return sound_id, preview_url
    return None

async def download_sound(sound_id, url, filename):
    headers = {"Authorization": f"Token {FREESOUND_API_KEY}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                logger.error(f"Failed to download sound {sound_id}: HTTP {response.status}")
                return False
            async with aiofiles.open(filename, "wb") as f:
                await f.write(await response.read())
    return True

async def generate_sfx_background(task_id: str, query: str, save_path: str):
    try:
        result = await fetch_sound_url(query)
        if not result:
            sfx_tasks[task_id] = {"status": "failed", "error": "Sound not found"}
            logger.error(f"Task {task_id}: Sound not found for query '{query}'")
            return

        sound_id, url = result
        if not os.path.exists(save_path):
            success = await download_sound(sound_id, url, save_path)
            if not success:
                sfx_tasks[task_id] = {"status": "failed", "error": "Failed to download sound"}
                logger.error(f"Task {task_id}: Failed to download sound")
                return

        if not os.path.exists(save_path):
            sfx_tasks[task_id] = {"status": "failed", "error": "Sound file not created"}
            logger.error(f"Task {task_id}: Sound file not created at {save_path}")
            return

        sfx_tasks[task_id] = {
            "status": "completed",
            "url": f"/static/sounds/{os.path.basename(save_path)}"
        }
        logger.info(f"Task {task_id}: SFX generated at {save_path}")
    except Exception as e:
        sfx_tasks[task_id] = {"status": "failed", "error": str(e)}
        logger.error(f"Task {task_id}: Error generating SFX: {e}")

@app.post("/generate_sfx")
async def generate_sfx(data: dict, background_tasks: BackgroundTasks):
    query = data.get("text", "")
    if not query:
        raise HTTPException(status_code=400, detail="Text is required")

    task_id = str(uuid.uuid4())
    filename = await sanitize_filename(query) or task_id
    save_path = f"static/sounds/{filename}.mp3"

    sfx_tasks[task_id] = {"status": "processing"}
    logger.info(f"Task {task_id}: Started SFX generation for query '{query}'")

    background_tasks.add_task(generate_sfx_background, task_id, query, save_path)
    return {"task_id": task_id}

@app.get("/check_sfx_status/{task_id}")
async def check_sfx_status(task_id: str):
    task = sfx_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/static/sounds/{filename:path}")
async def get_sound(filename: str):
    file_path = os.path.join("static/sounds", filename)
    if not os.path.exists(file_path):
        logger.error(f"Sound file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )



## ========================================================================================================


# Suppress the GenerationMixin warning
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")

# 1. Fix PyTorch 2.6 weights_only issue
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# 2. Import required classes for safe globals
try:
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
    from TTS.tts.models.xtts import XttsArgs
    
    # Add safe globals
    torch.serialization.add_safe_globals([
        BaseDatasetConfig,
        XttsConfig, 
        XttsAudioConfig,
        XttsArgs
    ])
except ImportError as e:
    print(f"Warning: Could not import some TTS classes for safe globals: {e}")

# 3. Fix GenerationMixin inheritance issue
try:
    from transformers import GenerationMixin
    from TTS.tts.layers.xtts.gpt import GPT2InferenceModel
    import TTS.tts.layers.xtts.gpt as gpt_module
    
    # Create patched model class
    class PatchedGPT2InferenceModel(GPT2InferenceModel, GenerationMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def generate(self, *args, **kwargs):
            # Call the parent generate method
            outputs = super().generate(*args, **kwargs)
            
            # Handle different output types
            if hasattr(outputs, 'sequences'):
                return outputs.sequences
            elif hasattr(outputs, 'logits'):
                # For CausalLMOutputWithCrossAttentions, extract sequences properly
                return outputs.logits.argmax(-1)
            else:
                return outputs
    
    # Replace the original class
    gpt_module.GPT2InferenceModel = PatchedGPT2InferenceModel
    
except ImportError as e:
    print(f"Warning: Could not patch GenerationMixin: {e}")

# 4. Initialize TTS with error handling
def initialize_tts(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False):
    # Try different import methods
    TTS_class = None
    
    # Method 1: Direct import
    try:
        from TTS import TTS as TTS_class
        print("TTS imported directly from TTS package")
    except ImportError:
        pass
    
    # Method 2: Import from api
    if TTS_class is None:
        try:
            from TTS.api import TTS as TTS_class
            print("TTS imported from TTS.api")
        except ImportError:
            pass
    
    # Method 3: Import from utils
    if TTS_class is None:
        try:
            from TTS.utils.synthesizer import Synthesizer
            print("Warning: Using Synthesizer instead of TTS class")
            # This is a more complex fallback - you'd need to handle this differently
        except ImportError:
            pass
    
    if TTS_class is None:
        raise ImportError("Could not import TTS class from any location")
    
    try:
        return TTS_class(model_name, gpu=gpu)
    except Exception as e:
        print(f"Error initializing TTS with {model_name}: {e}")
        # Fallback to a simpler model
        try:
            return TTS_class("tts_models/en/ljspeech/tacotron2-DDC", gpu=gpu)
        except Exception as e2:
            print(f"Fallback TTS also failed: {e2}")
            raise e

# Initialize TTS
try:
    tts = initialize_tts()
    print("TTS initialized successfully!")
except Exception as e:
    print(f"Failed to initialize TTS: {e}")
    print("Please reinstall TTS library: pip uninstall TTS && pip install TTS")
    tts = None


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
