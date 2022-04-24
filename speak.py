import sys
import pyttsx3

engine = pyttsx3.init()
engine.say(str(sys.argv[1]))
engine.runAndWait()