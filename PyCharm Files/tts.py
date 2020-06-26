
import pyttsx3

engine = pyttsx3.init('sapi5')
# testing
engine.say("My first code on text-to-speech")
engine.say("Thank you, Geeksforgeeks")
engine.runAndWait()

"""
def onStart():
    print('starting')


def onWord(name, location, length):
    print('word', name, location, length)


def onEnd(name, completed):
    print('finishing', name, completed)


engine = pyttsx3.init()

engine.connect('started-utterance', onStart)
engine.connect('started-word', onWord)
engine.connect('finished-utterance', onEnd)

sen = 'Geeks for geeks is a computer portal for Geeks'

engine.say(sen)
engine.runAndWait()
"""