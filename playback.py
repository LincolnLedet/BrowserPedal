import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
import keyboard  # pip install keyboard

# -----------------------
#       DEVICE SETUP
# -----------------------
# According to your device list:
#  - Input (AudioBox USB 96, WASAPI) = 33
#  - Output (Headphones Realtek, WASAPI) = 27
INPUT_DEVICE = 33
OUTPUT_DEVICE = 27

# We'll run WASAPI in exclusive mode for lower latency
try:
    wasapi_exclusive = sd.WasapiSettings(exclusive=True)
    print("✅ Using WASAPI exclusive mode.\n")
except AttributeError:
    wasapi_exclusive = None
    print("❌ WASAPI exclusive mode not available; falling back to default.\n")

# -----------------------
#     AUDIO PARAMETERS
# -----------------------
# Matching Windows device settings is crucial:
samplerate = 48000
blocksize = 256  # a moderate buffer size; increase if you hear crackling
channels = 2     # We'll capture from the left channel, then duplicate for stereo output

# We'll use float32 so the effects can safely manipulate the signal
dtype = 'float32'

# -----------------------
#  EFFECT & PLOTTING SETUP
# -----------------------
current_effect = "raw"  # default effect

# We'll plot 'blocksize' samples in real time
fig, ax = plt.subplots()
x = np.arange(blocksize)
y = np.zeros(blocksize, dtype=np.float32)  # global array to store current block for plotting

line, = ax.plot(x, y)
ax.set_ylim(-1, 1)
ax.set_xlim(0, blocksize)
ax.set_title("Live Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# Precompute a tremolo LFO for 'blocksize' frames
# Speed = 15 Hz, depth = 0.4
t = np.arange(blocksize) / samplerate
lfo = 1.0 + 0.4 * np.sin(2.0 * np.pi * 15.0 * t)

# -----------------------
#      EFFECT FUNCTIONS
# -----------------------
@jit(nopython=True)
def raw_data_output(audio_block):
    """No processing, just pass raw data."""
    return audio_block

@jit(nopython=True)
def basicDistortion(audio_block):
    """
    Extremely simple 'clamp' distortion:
    Anything above +0.02 is pinned at +0.02,
    anything below -0.02 is pinned at -0.02
    """
    audio_block[audio_block > 0.02] = 0.02
    audio_block[audio_block < -0.02] = -0.02
    return audio_block

@jit(nopython=True)
def audioBoost(audio_block):
    """Increase amplitude by 1.5x."""
    return audio_block * 1.5

@jit(nopython=True)
def tremolo(audio_block, lfo_block):
    """
    Multiply by a sinusoidal LFO (1 + 0.4*sin...).
    This LFO array is blocksize in length.
    """
    return audio_block * lfo_block

# -----------------------
#   AUDIO CALLBACK
# -----------------------
def process_audio(indata, outdata, frames, time, status):
    """
    Called automatically by sounddevice for each audio block.
    indata.shape = (blocksize, channels)
    outdata.shape = (blocksize, channels)
    """
    global current_effect, y

    if status:
        print("Status:", status)

    # indata[:, 0] => left channel of input (assuming your signal is here)
    mono_in = indata[:, 0]

    # Apply the chosen effect
    if current_effect == "raw":
        processed = raw_data_output(mono_in)
    elif current_effect == "distortion":
        processed = basicDistortion(mono_in)
    elif current_effect == "boost":
        processed = audioBoost(mono_in)
    elif current_effect == "tremolo":
        processed = tremolo(mono_in, lfo)  # use our precomputed LFO
    else:
        processed = mono_in  # default fallback

    # Duplicate processed mono => stereo (both L & R channels)
    outdata[:, 0] = processed
    outdata[:, 1] = processed

    # Update the global 'y' array for plotting
    y[:] = processed[:blocksize]  # store the data for visualization

# -----------------------
#  MATPLOTLIB ANIMATION
# -----------------------
def update_plot(frame):
    """Updates the waveform line with the latest block data."""
    line.set_ydata(y)
    return line,

ani = animation.FuncAnimation(
    fig, update_plot, interval=20, blit=True
)

# -----------------------
#  KEYBOARD HOTKEYS
# -----------------------
def switch_effect(effect_name):
    global current_effect
    if effect_name in ["raw", "distortion", "boost", "tremolo"]:
        current_effect = effect_name
        print(f"🔄 Switched to effect: {effect_name}")
    else:
        print("❌ Invalid effect name! Choose from 'raw', 'distortion', 'boost', 'tremolo'.")

keyboard.add_hotkey("1", lambda: switch_effect("raw"))
keyboard.add_hotkey("2", lambda: switch_effect("distortion"))
keyboard.add_hotkey("3", lambda: switch_effect("boost"))
keyboard.add_hotkey("4", lambda: switch_effect("tremolo"))

# -----------------------
#       MAIN
# -----------------------
def main():
    # Print device info
    print("\n=== Available Devices ===")
    all_devices = sd.query_devices()
    for i, d in enumerate(all_devices):
        print(i, d["name"], d["hostapi"], d["max_input_channels"], d["max_output_channels"])

    print("\n=== Host APIs ===")
    host_apis = sd.query_hostapis()
    for i, api in enumerate(host_apis):
        print(i, api["name"])
    print()

    print("🎛️ Use keys: 1 = Raw, 2 = Distortion, 3 = Boost, 4 = Tremolo")
    print("🎵 Press Ctrl+C in the console to stop (or close the plot).")

    # Open a full-duplex stream using WASAPI exclusive mode
    with sd.Stream(
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        samplerate=samplerate,
        blocksize=blocksize,
        channels=channels,
        dtype=dtype,
        latency="low",
        callback=process_audio,
        extra_settings=wasapi_exclusive
    ):
        # Start the matplotlib animation (blocking call)
        plt.show()

if __name__ == "__main__":
    main()
