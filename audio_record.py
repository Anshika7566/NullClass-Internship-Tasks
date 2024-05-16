import sounddevice
from scipy.io.wavfile import write

def savefile(sec):
    print("start")
    rece=sounddevice.rec((sec*44100),samplerate=44100,channels=2)
    sounddevice.wait()
    write("demo.wav",44100,rece)
    print("end")

savefile(10)