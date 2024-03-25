# Character Lip-Sync with Xttsv2 and Rhubarb

NOTE: This code is really just me experimenting with stuff. I really haven't cleaned it up at all...

This is my experimentation with xttsv2 audio generation and extracting visemes\phonemes out of the audio and trying to lip-sync the audio with the mouth movements.

Repo includes :
- python notebook experimenting with coqui
- python web app and html\javascript for real-time voice synth and lip-syncing with xttsv2

(I've used the mouth shapes from the oculus viseme git repo.)

In order to use this, you will need to ensure you have a working python venv with the following :
- FastAPI
- coqui tts (xttsv2)
- uvicorn 

You will also need to:
- download rhubarb and place in the same directory as the app.py
- create the tempfiles directory (I used a local tempfiles directory because I had trouble deleting home path based tempfiles)

This isn't production ready code at all. Just experiments and commiting to git really to just be able to use elsewhere.

I wasn't able to get DeepSpeed working with the xttsv2 on Windows but when I have done this previously for other projects, deepspeed does seem to help speed up xttsv2 inference significantly.

This will run on either windows or linux. For Windows, would have to modify the code to correctly point to the Rhubarb.exe path. Also note that deepspeed is super annoying to get working on windows. You are probably just better off running this in WSL if running windows. Deepspeed provides a significant speed boost for xttsv2 so it's worthwhile.

I've gotten near real-time performance when running on an A6000 on Lambda Labs cloud gpu. The A10's seem to be more available and are pretty good too (also a few cents/hour less expensive). If running on cloud gpu, recommend setting up a tunnel with ngrok. Works very well.

The intention of this app is to provide a backend to VRMChatbot (also available in my git repos). This provides a websocket based integration with xttsv2 and also provides vieseme data for lip-sync with the VRM models in that application.

### Instructions for running on Ubuntu

1. Setup python venv
```bash
sudo apt install python3.10-venv
python3 -m venv ./venv
source ./venv/bin/activate
```

2. Install dependencies
```bash
pip3 install torch torchvision torchaudio TTS deepspeed FASTAPI uvicorn websockets
```

3. Download Rhubarb
```bash
wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip
unzip Rhubarb-Lip-Sync*.zip
mv Rhubarb-Lip-Sync-1.13.0-Linux rhubarb
rm *.zip
```

4. Download xtts model
```bash
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/coqui/XTTS-v2
```

5. You should now be able to run the server
```bash
uvicorn app:app
```
