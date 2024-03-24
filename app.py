from fastapi import FastAPI, WebSocket, responses, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os
import time
import torch
import torchaudio
import asyncio
import io
import subprocess
import threading
import json
import tempfile
import time

app = FastAPI()
config = XttsConfig()
config.load_json("./config.json")
model = Xtts.init_from_config(config)
gpt_cond_latent = None
speaker_embedding = None

app.mount("/static", StaticFiles(directory="static"), name="static")

# Path to the subdirectory for temporary files (relative to this script)
temp_files_dir = os.path.join(os.path.dirname(__file__), 'tempfiles')

templates = Jinja2Templates(directory="templates")

# Ensure the subdirectory exists
os.makedirs(temp_files_dir, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    domain = request.url.hostname
    path = request.url.path
    query_params = request.query_params
    protocol = request.url.scheme
    port = request.url.port

    return templates.TemplateResponse(
        request=request, name="index.html", context={"domain":domain, "protocol":protocol, "port":port}
    )


@app.get("/load")
async def load():
    global model, gpt_cond_latent, speaker_embedding
    print("Loading model...")
    model.load_checkpoint(config, checkpoint_dir="./XTTS-v2", use_deepspeed=True)
    model.cuda()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["./samples/sample.mp3"])

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
            result = subprocess.run(["./rhubarb/rhubarb", "-r", "phonetic", "-o", rhubarb_output.name, temp_wav.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())  # Now this should work as stdout is captured
            print(result.stderr.decode())  # Same for stderr
        except subprocess.CalledProcessError as e:
            print(f"Rhubarb execution failed: {e}")
            print(e.stdout.decode() if e.stdout else "No stdout")
            print(e.stderr.decode() if e.stderr else "No stderr")
            return {}
        with open(rhubarb_output.name, "r") as f:
            viseme_data = parse_viseme_data(f.readlines())
    return viseme_data


# Updated audio_stream function to use the refactored process_audio_chunk
@app.websocket("/audio_stream")
async def audio_stream(websocket: WebSocket):
    global speaker_embedding, gpt_cond_latent, model
    await websocket.accept()
    wav_chuncks = []
    while True:
        print('awaiting receiving text')
        text_data = await websocket.receive_text()
        print('received : ' + text_data)

        if text_data == "END":
            print('text_data == end')
            break

        chunks = model.inference_stream(
            text_data,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            50
        )
        print('processed chunk')

        wav_chunks = []

        # Assuming 'chunks' is the list of audio chunks you generated
        for chunk in chunks:
            wav_chunks.append(chunk)

        print('converting chunks to wav')
        wav = torch.cat(wav_chunks, dim=0)

        print('processing audio chunk')
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
