import os
import speech_recognition as sr 
r=sr.Recognizer()

myaudio=sr.AudioFile("/media/karthik/EC742F3F742F0C40/Friends_tech/gesture-recognition/speechprocessing/audio_files/audf.wav")

with myaudio as source:
    audio=r.record(source,duration=8)
print(type(audio))
print(r.recognize_google(audio))