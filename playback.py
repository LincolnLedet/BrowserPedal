import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
import keyboard  # Install with `pip install keyboard`
import math


samplerate = 22050
blocksize = 512
channels = 1


current_effect = "raw"  # Default effect

fig, ax = plt.subplots()
x = np.linspace(0, blocksize, blocksize)
y = np.zeros(blocksize)

line, = ax.plot(x, y)
ax.set_ylim(-1, 1)
ax.set_xlim(0, blocksize)
ax.set_title("Live Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

@jit(nopython=True)
def raw_data_output(audio_block):
    return audio_block  # No processing, just raw data

@jit(nopython=True)
def basicDistortion(audio_block):
    audio_block[audio_block > .15] = .15
    audio_block[audio_block < -.15] = -.15
    return audio_block 

@jit(nopython=True)
def audioBoost(audio_block):
    return audio_block * 1.5



def tremolo(audio_block, samplerate, depth=.4, freq=15.0):
    t = np.arange(len(audio_block)) / samplerate  # Time vector
    modulation = (depth * np.sin(2 * np.pi * freq * t))  # Sine LFO
    return audio_block * modulation 


def process_audio(indata, outdata, frames, time, status):
    global current_effect  

    if status:
        print(status)  

    
    if current_effect == "raw":
        processed = raw_data_output(indata[:, 0])
    elif current_effect == "distortion":
        processed = basicDistortion(indata[:, 0])
    elif current_effect == "boost":
        processed = audioBoost(indata[:, 0])
    elif current_effect == "tremolo":
        processed = tremolo(indata[:, 0], samplerate) 
    else:
        processed = indata[:, 0] 

    outdata[:, 0] = processed  

 
    global y
    y[:] = processed

def update_plot(frame):
    line.set_ydata(y)
    return line,

# Function to switch effects
def switch_effect(effect_name):
    global current_effect
    if effect_name in ["raw", "distortion", "boost","tremolo"]:
        current_effect = effect_name
        print(f"🔄 Switched to effect: {effect_name}")
    else:
        print("❌ Invalid effect name! Choose from 'raw', 'distortion', 'boost', 'tremolo'.")

# ✅ Use `keyboard.add_hotkey()` instead of a separate thread
keyboard.add_hotkey("1", lambda: switch_effect("raw"))
keyboard.add_hotkey("2", lambda: switch_effect("distortion"))
keyboard.add_hotkey("3", lambda: switch_effect("boost"))
keyboard.add_hotkey("4", lambda: switch_effect("tremolo"))


ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True)

with sd.Stream(callback=process_audio, samplerate=samplerate, blocksize=blocksize, channels=channels):
    print("🎵 Processing live audio... Press Ctrl+C to stop.")
    print("🎛️ Use keys: 1 = Raw, 2 = Distortion, 3 = Boost, 4 = tremolo")
    plt.show()  # Start the visualization
