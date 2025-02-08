import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
import threading
import keyboard  # Install with `pip install keyboard`

# Audio Stream Parameters
samplerate = 22050
blocksize = 512
channels = 1

# Global variable for effect selection
current_effect = "raw"  # Default effect

# Set up the figure and axis
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
    return audio_block  # Apply basic distortion

@jit(nopython=True)
def audioBoost(audio_block):
    return audio_block * 1.5  # Boost volume by 1.5x

def process_audio(indata, outdata, frames, time, status):
    global current_effect  # Access the global variable

    if status:
        print(status)  # Handle errors

    # Apply the selected effect
    if current_effect == "raw":
        processed = raw_data_output(indata[:, 0])
    elif current_effect == "distortion":
        processed = basicDistortion(indata[:, 0])
    elif current_effect == "boost":
        processed = audioBoost(indata[:, 0])
    else:
        processed = indata[:, 0]  # Default: raw audio

    outdata[:, 0] = processed  # Send processed audio to output

    # Update the waveform visualization
    global y
    y[:] = processed

def update_plot(frame):
    line.set_ydata(y)
    return line,

# Function to switch effects
def switch_effect(effect_name):
    global current_effect
    if effect_name in ["raw", "distortion", "boost"]:
        current_effect = effect_name
        print(f"🔄 Switched to effect: {effect_name}")
    else:
        print("❌ Invalid effect name! Choose from 'raw', 'distortion', 'boost'.")

# Function to listen for key presses
def listen_for_keys():
    while True:
        if keyboard.is_pressed("1"):
            switch_effect("raw")
        elif keyboard.is_pressed("2"):
            switch_effect("distortion")
        elif keyboard.is_pressed("3"):
            switch_effect("boost")

# Start listening for key presses in a separate thread
threading.Thread(target=listen_for_keys, daemon=True).start()

# Start the animation
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True)

with sd.Stream(callback=process_audio, samplerate=samplerate, blocksize=blocksize, channels=channels):
    print("🎵 Processing live audio... Press Ctrl+C to stop.")
    print("🎛️ Use keys: 1 = Raw, 2 = Distortion, 3 = Boost")
    plt.show()
