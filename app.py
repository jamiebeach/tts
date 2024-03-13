from fastapi import FastAPI, WebSocket, responses
import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import asyncio
from fastapi.staticfiles import StaticFiles
import io
from pydub import AudioSegment
import subprocess
import threading
import json
import tempfile

app = FastAPI()
config = XttsConfig()
config.load_json("C:\\Users\\jamie\\AppData\\Local\\tts\\tts_models--multilingual--multi-dataset--xtts_v2\\config.json")
model = Xtts.init_from_config(config)
gpt_cond_latent = None
speaker_embedding = None

app.mount("/static", StaticFiles(directory="static"), name="static")

# Path to the subdirectory for temporary files (relative to this script)
temp_files_dir = os.path.join(os.path.dirname(__file__), 'tempfiles')

# Ensure the subdirectory exists
os.makedirs(temp_files_dir, exist_ok=True)


@app.get("/")
async def main():
    return responses.RedirectResponse(url='/static/index.html')

@app.get("/index2")
async def index2():
    return responses.RedirectResponse(url='/static/index2.html')

@app.get("/load")
async def load():
    global model, gpt_cond_latent, speaker_embedding
    print("Loading model...")
    model.load_checkpoint(config, checkpoint_dir="C:\\Users\\jamie\\AppData\\Local\\tts\\tts_models--multilingual--multi-dataset--xtts_v2\\", use_deepspeed=False)
    model.cuda()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[".\\samples\\example_reference.mp3"])

@app.websocket("/audio_stream")
async def audio_stream(websocket: WebSocket):
    global speaker_embedding, gpt_cond_latent, model

    await websocket.accept()

    chunks = model.inference_stream(
        "Hello there. I'm Bob. I'm a lumberjack and I like cheese. Be careful out there. The rabbits are crazy!",
        "en",
        gpt_cond_latent,
        speaker_embedding
    )

    # Assuming 'chunks' is the list of audio chunks you generated
    wav_chuncks = []
    for chunk in chunks:
        buffer = io.BytesIO()
        wav_chuncks.append(chunk)

        # Save the chunk to the buffer
        torchaudio.save(buffer, chunk.squeeze().unsqueeze(0).cpu(), 24000, format='wav')

        # Get the buffer data as bytes
        buffer_bytes = buffer.getvalue()

        # Send the buffer bytes over the WebSocket
        await websocket.send_bytes(buffer_bytes)

        await asyncio.sleep(0.05)  # Adjust based on your chunk length and network conditions
    
    await asyncio.sleep(1.5)
    await websocket.close()
    wav = torch.cat(wav_chuncks, dim=0)
    torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)

@app.websocket("/audio_stream2")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()

    # Open the generated_audio.wav file
    with open("xtts_streaming.wav", "rb") as audio_file:
        # Stream the file data over the WebSocket
        while True:
            data = audio_file.read(24000)
            if not data:
                break
            await websocket.send_bytes(data)
            await asyncio.sleep(0.01)  # Adjust the delay as needed

    await websocket.close()

async def process_audio_chunk(websocket, chunk, wav_chunks):
    buffer = io.BytesIO()
    #wav_chunks.append(chunk)
    torchaudio.save(buffer, chunk.squeeze().unsqueeze(0).cpu(), 24000, format='wav')
    buffer_bytes = buffer.getvalue()
    await websocket.send_bytes(buffer_bytes)
    await asyncio.sleep(0.05)

    # Run rhubarb on the audio chunk
    viseme_data = await asyncio.to_thread(run_rhubarb, chunk)
    viseme_json = json.dumps(viseme_data)
    await websocket.send_text(viseme_json)

def run_rhubarb(chunk):
    global temp_files_dir

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        torchaudio.save(temp_wav.name, chunk.squeeze().unsqueeze(0).cpu(), 24000)
        rhubarb_output = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", dir=temp_files_dir)
        try:
            result = subprocess.run(["rhubarb.exe", "-o", rhubarb_output.name, temp_wav.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())  # Now this should work as stdout is captured
            print(result.stderr.decode())  # Same for stderr
        except subprocess.CalledProcessError as e:
            print(f"Rhubarb execution failed: {e}")
            print(e.stdout.decode() if e.stdout else "No stdout")
            print(e.stderr.decode() if e.stderr else "No stderr")
            return {}
        with open(rhubarb_output.name, "r") as f:
            viseme_data = parse_viseme_data(f.readlines())
        #os.remove(temp_wav.name)
        #os.remove(rhubarb_output.name)
        #safe_remove(temp_wav.name)
        #safe_remove(rhubarb_output.name)
    return viseme_data


# Updated audio_stream function to use the refactored process_audio_chunk
@app.websocket("/audio_stream3")
async def audio_stream(websocket: WebSocket):
    global speaker_embedding, gpt_cond_latent, model
    await websocket.accept()
    while True:
        text_data = await websocket.receive_text()
        if text_data == "END":
            break

        chunks = model.inference_stream(
            text_data,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            50
        )
        wav_chunks = []

        tasks = [asyncio.create_task(process_audio_chunk(websocket, chunk, wav_chunks)) for chunk in chunks]

        await asyncio.gather(*tasks)

        # Indicate end of processing for this text input
        await websocket.send_text("END")

    await asyncio.sleep(1.5)
    await websocket.close()

# Updated audio_stream function to use the refactored process_audio_chunk
@app.websocket("/audio_stream4")
async def audio_stream(websocket: WebSocket):
    global speaker_embedding, gpt_cond_latent, model
    await websocket.accept()
    while True:
        text_data = await websocket.receive_text()
        if text_data == "END":
            break

        chunks = model.inference_stream(
            text_data,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            50
        )
        wav_chunks = []

        # Assuming 'chunks' is the list of audio chunks you generated
        wav_chuncks = []
        for chunk in chunks:
            wav_chuncks.append(chunk)

        wav = torch.cat(wav_chuncks, dim=0)
        await process_audio_chunk(websocket, wav, wav_chunks)

        # Indicate end of processing for this text input
        await websocket.send_text("END")

    await asyncio.sleep(1.5)
    await websocket.close()

def parse_viseme_data(viseme_lines):
    viseme_data = []
    for line in viseme_lines:
        time, viseme = line.strip().split()
        viseme_data.append({"time": float(time), "viseme": viseme})
    return viseme_data

import time

def safe_remove(file_path, max_attempts=5, wait_seconds=1):
    """Attempt to remove a file with retries on PermissionError."""
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            print(f"Successfully deleted {file_path}")
            break  # Exit the loop if file deletion was successful
        except PermissionError as e:
            print(f"Attempt {attempt + 1}: Could not delete {file_path} - {e}")
            time.sleep(wait_seconds)  # Wait for a bit before retrying
    else:
        print(f"Failed to delete {file_path} after {max_attempts} attempts.")

# Replace direct os.remove calls with safe_remove in your run_rhubarb function
# Example:
# safe_remove(temp_wav.name)
# safe_remove(rhubarb_output.name)
