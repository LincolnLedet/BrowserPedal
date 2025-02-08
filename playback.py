import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard

# -----------------------
#       DEVICE SETUP
# -----------------------
INPUT_DEVICE = 33
OUTPUT_DEVICE = 27

try:
    wasapi_exclusive = sd.WasapiSettings(exclusive=True)
    print("Using WASAPI exclusive mode.\n")
except AttributeError:
    wasapi_exclusive = None
    print("WASAPI exclusive mode not available; using default.\n")

# -----------------------
#     AUDIO PARAMETERS
# -----------------------
samplerate = 48000
blocksize = 512
channels = 2
dtype = 'float32'

# -----------------------
#  EFFECT & PLOTTING
# -----------------------
current_effect = "raw"

fig, ax = plt.subplots()
x = np.arange(blocksize)
y = np.zeros(blocksize, dtype=np.float32)
line, = ax.plot(x, y)
ax.set_ylim(-1, 1)
ax.set_xlim(0, blocksize)
ax.set_title("Live Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# -----------------------
#  PARAMETERS FOR DELAY
# -----------------------
# Total ring buffer size (2 seconds @ 48kHz)
delay_buffer_size = samplerate * 2
delay_buffer = np.zeros(delay_buffer_size, dtype=np.float32)

# How many samples to delay the audio
delay_time_sec = 0.3  # 300 ms
delay_samples = int(delay_time_sec * samplerate)

# How much of the delayed signal to feed back into future repeats
feedback = 0.5  # 0.0 = single echo, near 1.0 = many repeats

# Wet mix = how much delayed signal vs. dry signal
wet_mix = 0.5

# This points to where we write new samples in the ring buffer
write_index = 0


# -----------------------
#      EFFECT FUNCTIONS
# -----------------------
def raw_data_output(audio_block):
    return audio_block

def basicDistortion(audio_block):
    audio_block[audio_block > 0.02] = 0.02
    audio_block[audio_block < -0.02] = -0.02
    return audio_block

def audioBoost(audio_block):
    return audio_block * 1.5

def tremolo(audio_block, phase):
    # Simple LFO-based tremolo
    return audio_block * (1.0 + 0.4 * np.sin(phase))

def delayEffect(audio_block):
    """
    Implements a ring-buffer-based delay with feedback and wet/dry mix.
    """
    global write_index, delay_buffer

    n = len(audio_block)
    out_block = np.zeros_like(audio_block)

    for i in range(n):
        # Where we read from in the delay buffer
        read_index = (write_index - delay_samples) % delay_buffer_size

        # Grab the delayed sample
        delayed_sample = delay_buffer[read_index]

        # Wet/dry mix:
        # out = dry*(1-wet) + delayed*(wet)
        out_block[i] = (audio_block[i] * (1.0 - wet_mix)) + (delayed_sample * wet_mix)

        # Write new sample + feedback into the delay buffer
        delay_buffer[write_index] = audio_block[i] + (delayed_sample * feedback)

        # Move write pointer forward
        write_index = (write_index + 1) % delay_buffer_size

    return out_block

# -----------------------
#   AUDIO CALLBACK
# -----------------------
trem_phase = 0.0
def process_audio(indata, outdata, frames, time_info, status):
    global current_effect, y, trem_phase

    if status:
        print("Status:", status)

    mono_in = indata[:, 0]

    if current_effect == "raw":
        processed = raw_data_output(mono_in)
    elif current_effect == "distortion":
        processed = basicDistortion(mono_in)
    elif current_effect == "boost":
        processed = audioBoost(mono_in)
    elif current_effect == "tremolo":
        processed = tremolo(mono_in, trem_phase)
        trem_phase += 0.4  # increment for next block's LFO
    elif current_effect == "delay":
        processed = delayEffect(mono_in)
    else:
        processed = mono_in

    outdata[:, 0] = processed
    outdata[:, 1] = processed

    y[:] = processed[:blocksize]

# -----------------------
#  MATPLOTLIB ANIMATION
# -----------------------
def update_plot(frame):
    line.set_ydata(y)
    return line,

ani = animation.FuncAnimation(fig, update_plot, interval=20, blit=True)

# -----------------------
#  KEYBOARD HOTKEYS
# -----------------------
def switch_effect(effect_name):
    global current_effect
    if effect_name in ["raw", "distortion", "boost", "tremolo", "delay"]:
        current_effect = effect_name
        print(f"Switched to effect: {effect_name}")
    else:
        print("Invalid effect! Choose: raw, distortion, boost, tremolo, or delay.")

keyboard.add_hotkey("1", lambda: switch_effect("raw"))
keyboard.add_hotkey("2", lambda: switch_effect("distortion"))
keyboard.add_hotkey("3", lambda: switch_effect("boost"))
keyboard.add_hotkey("4", lambda: switch_effect("tremolo"))
keyboard.add_hotkey("5", lambda: switch_effect("delay"))

# -----------------------
#       MAIN
# -----------------------
def main():
    print("Keys: 1=Raw, 2=Distortion, 3=Boost, 4=Tremolo, 5=Delay")
    print("Press Ctrl+C or close the plot to stop.")

    with sd.Stream(
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        samplerate=samplerate,
        blocksize=blocksize,
        channels=channels,
        dtype=dtype,
        latency="low",
        callback=process_audio,
    ):
        plt.show()

if __name__ == "__main__":
    main()
