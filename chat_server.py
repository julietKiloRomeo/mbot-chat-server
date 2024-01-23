import os
import pyaudio
import wave
from pydub import AudioSegment
# Set your OpenAI API key
from openai import OpenAI

from pydub.playback import play

from flask import Flask, request, send_file
import os
import urllib


app = Flask(__name__)


def record_audio(filename, duration=5):
    """ find the sandberg device with:
        devinfo = pyaudio.PyAudio().get_device_info_by_index(9)
        devinfo
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    device_index = 9

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=device_index,
    )

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


class Conversation:
    SYS = """Du er en venlig assistent for Alvilde. Du hedder Britney. Du svarer altid på dansk.

    Alvilde er 9 år og går på Birkerød privatskole i 3.Ø.

    Hendes mor hedder Camilla og er fra Norge. Camilla kan lide at strikke og træne og er klog
    og smuk.

    Alvildes far hedder Jimmy. Han kan lide kaffe og rødvin og er super pinlig.

    Alvilde kan godt lide vittigheder, dinosaurer, håndbold og judo.

    Selvom du er en tekstbaseret model, taler Alvilde med dig og du svarer hende med tale. Som tekst-model
    er du en del af et større system, der inkluderer tale-til-tekst og tekst-til tale moduler.
    """
    voice = "shimmer"
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.messages=[]
        self.add_message(self.SYS, role="system")

    def add_message(self, message, role="user"):
        if isinstance(message, str):
            self.messages.append(
                {
                    "role":role,
                    "content":message,
                })
        else:
            self.messages.append(message)

    def _clean_message(self, message):
        return {k:v for k,v in message.items() if k in ["role", "content"]}
        
    def reply(self):
        completion = self.client.chat.completions.create(
          model="gpt-4",
          messages=[self._clean_message(msg) for msg in self.messages],
        )
        latest_msg = completion.choices[0].message.model_dump()
        self.add_message(latest_msg, role="assistant")
        return latest_msg["content"]

    def tell_from_text(self, message):
        self.add_message(message)
        return self.reply()


    def tell_from_wav(self, wavfile = "test.wav"):
        transcript = self.transcribe(wavfile)
        return self.tell_from_text(transcript)

    def tell_from_mic(self, duration=10):
        record_audio("test.wav", duration=duration)
        return self.tell_from_wav()

    def say_latest(self, silent=False):
        response = self.client.audio.speech.create(
          model="tts-1",
          voice=self.voice,
          input=self.messages[-1]["content"],
        )
        
        response.stream_to_file("reply.mp3")
        
        if not silent:
            reply = AudioSegment.from_mp3("reply.mp3")
            play(reply)
    
    def transcribe(self, wavname):
        """convert wav file to text
        """
        name, _ = wavname.split(".")
        mp3name = f"{name}.mp3"
        AudioSegment.from_wav(wavname).export(mp3name, format="mp3")
        audio_file= open(mp3name, "rb")
        return self.client.audio.transcriptions.create(
          model="whisper-1", 
          language="da",
          prompt="Det er en dansk tekst, der nok mest er spørgsmål om Birkerød eller dinosaurer.",
          file=audio_file,
        ).model_dump()["text"]



    def make_image(self, description):
        response = client.images.generate(
          model="dall-e-3",
          prompt=description,
          size="1024x1024",
          quality="hd",
          style="vivid",
          n=1,
        )
        image_url = response.data[0].url

        urllib.request.urlretrieve(image_url, "image.png") 


conv = Conversation()

@app.route('/echo', methods=['POST'])
def echo():
    # Echoes back the received data
    data = request.get_data()
    return data

@app.route('/tell', methods=['POST'])
def tell():
    # Receives a file and responds with another file
    file = request.files['file']
    
    # Process the received file here if needed
    # For simplicity, this example just saves and sends back the same file
    filename = 'received_file.wav'
    file.save(filename)

    conv.tell_from_wav(wavfile = filename)
    conv.say_latest(silent=True)
    
    # Sending back a file
    return send_file("reply.mp3", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)




