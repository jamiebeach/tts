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
- pydub
- uvicorn 

(note that pydub isn't reall used in the end but I haven't removed the imports)

You will also need to:
- download rhubarb and place in the same directory as the app.py
- create the tempfiles directory (I used a local tempfiles directory because I had trouble deleting home path based tempfiles)

This isn't production ready code at all. Just experiments and commiting to git really to just be able to use elsewhere.

I wasn't able to get DeepSpeed working with the xttsv2 on Windows but when I have done this previously for other projects, deepspeed does seem to help speed up xttsv2 inference significantly.

I've only tried this particular code on Windows. I don't know how well rhubarb works yet on Linux, but intention would be to ultimately use this as a basis for an xtts\lip-sync server for other projects.