import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# -----------------------
#       DEVICE SETUP
# -----------------------
INPUT_DEVICE = 33
OUTPUT_DEVICE = 27

# -----------------------
#     AUDIO PARAMETERS
# -----------------------
samplerate = 48000
blocksize = 1024
channels = 2
dtype = 'float32'

# -----------------------
#  GLOBAL AUDIO VARIABLES
# -----------------------
distortion_drive = 0.1
volume = 1.0
tremolo_volume = 0
tremolo_freq = 0

# Delay parameters
delay_time_sec = 0  # Delay time in seconds
feedback = 0       # Feedback amount
wet_mix = 0       # Mix of dry and delayed signal

# Delay buffer
delay_buffer_size = samplerate * 2
delay_buffer = np.zeros(delay_buffer_size, dtype=np.float32)
write_index = 0
delay_samples = 0

# bit deapth
bit_depth = 30

#  MATPLOTLIB SETUP

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.6)

x = np.arange(blocksize)
y = np.zeros(blocksize, dtype=np.float32)
line, = ax.plot(x, y)

ax.set_ylim(-1, 1)
ax.set_xlim(0, blocksize)
ax.set_title("Live Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# -----------------------
#  SLIDER WIDGETS [left, bottom, width, height]
# -----------------------
dist_ax = plt.axes([0.15, 0.05, 0.70, 0.02])
vol_ax = plt.axes([0.15, 0.1, 0.70, 0.02])
trem_vol_ax = plt.axes([0.15, 0.15, 0.70, 0.02])
trem_freq_ax = plt.axes([0.15, 0.2, 0.70, 0.02])
delay_time_ax = plt.axes([0.15, 0.25, 0.70, 0.02])
feedback_ax = plt.axes([0.15, 0.3, 0.70, 0.02])
wet_mix_ax = plt.axes([0.15, 0.35, 0.70, 0.02])
bit_depth_ax = plt.axes([0.15, 0.4, 0.70, 0.02])

# Create sliders
dist_slider = Slider(dist_ax, "Drive", 0.0, 1.0, valinit=distortion_drive)
vol_slider = Slider(vol_ax, "Volume", 0.5, 2.0, valinit=volume)
tremolo_vol_slider = Slider(trem_vol_ax, "Tremolo Volume", 0.0, 1.0, valinit=tremolo_volume)
tremolo_freq_slider = Slider(trem_freq_ax, "Tremolo Frequency", 0.1, 10.0, valinit=tremolo_freq)
delay_time_slider = Slider(delay_time_ax, "Delay Time (s)", 0.05, 1.0, valinit=delay_time_sec)
feedback_slider = Slider(feedback_ax, "Feedback", 0.0, 1.0, valinit=feedback)
wet_mix_slider = Slider(wet_mix_ax, "Wet Mix", 0.0, 1.0, valinit=wet_mix)
bit_depth_slider = Slider(bit_depth_ax, "Bit Crunch", 1, 55, valinit=bit_depth)


def update_distortion(val):
    global distortion_drive
    distortion_drive = val
dist_slider.on_changed(update_distortion)

def update_volume(val):
    global volume
    volume = val
vol_slider.on_changed(update_volume)

def update_tremolo_volume(val):
    global tremolo_volume
    tremolo_volume = val
tremolo_vol_slider.on_changed(update_tremolo_volume)

def update_tremolo_freq(val):
    global tremolo_freq
    tremolo_freq = val
tremolo_freq_slider.on_changed(update_tremolo_freq)

def update_delay_time(val):
    global delay_time_sec, delay_samples
    delay_time_sec = val
    delay_samples = int(delay_time_sec * samplerate)
delay_time_slider.on_changed(update_delay_time)

def update_feedback(val):
    global feedback
    feedback = val
feedback_slider.on_changed(update_feedback)

def update_wet_mix(val):
    global wet_mix
    wet_mix = val
wet_mix_slider.on_changed(update_wet_mix)

def update_bit_depth(val):
    global bit_depth
    bit_depth = int(val)  # Ensure it remains an integer
bit_depth_slider.on_changed(update_bit_depth)




# EFFECT 

def bitCrunch(audio_block):
    global bit_depth
    for i in range(0 , blocksize, bit_depth):
        for j in range(i, min(i + bit_depth, blocksize)):
            audio_block[j] = audio_block[i]
        
    return audio_block

def basicDistortion(audio_block):
    global distortion_drive
    audio_block[audio_block > 1 - distortion_drive] = 1 - distortion_drive
    audio_block[audio_block < -(1 - distortion_drive)] = -(1 - distortion_drive)
    return audio_block

def audioBoost(audio_block):
    global volume
    return audio_block * volume

def tremolo(audio_block, phase):
    """ Apply tremolo using a per-block LFO """
    global tremolo_volume, tremolo_freq, samplerate
    samples = np.arange(len(audio_block))
    lfo = np.sin(2 * np.pi * tremolo_freq * samples / samplerate + phase)
    return audio_block * (1.0 + tremolo_volume * lfo)

def delayEffect(audio_block):

    global write_index, delay_buffer, delay_samples, feedback, wet_mix

    n = len(audio_block)
    out_block = np.zeros_like(audio_block)

    for i in range(n):
        read_index = (write_index - delay_samples) % delay_buffer_size
        delayed_sample = delay_buffer[read_index]

        out_block[i] = (audio_block[i] * (1.0 - wet_mix)) + (delayed_sample * wet_mix)
        delay_buffer[write_index] = audio_block[i] + (delayed_sample * feedback)

        write_index = (write_index + 1) % delay_buffer_size

    return out_block

# -----------------------
#   AUDIO CALLBACK
# -----------------------
trem_time = 0.0

def process_audio(indata, outdata, frames, time_info, status):
    global y, trem_time, tremolo_freq

    if status:
        print("Status:", status)

    mono_in = indata[:, 0]

    processed = audioBoost(mono_in)
    processed = basicDistortion(processed)
    processed = tremolo(processed, trem_time)
    processed = delayEffect(processed)
    processed = bitCrunch(processed)


    trem_time += (2 * np.pi * tremolo_freq * frames) / samplerate
    trem_time %= (2 * np.pi)  # Keep phase within 0 - 2π

    outdata[:, 0] = processed
    outdata[:, 1] = processed
    y[:] = processed[:blocksize]

# -----------------------
#  MATPLOTLIB ANIMATION
# -----------------------
def update_plot(frame):
    line.set_ydata(y)
    return (line,)

ani = animation.FuncAnimation(fig, update_plot, interval=20, blit=True)

# -----------------------
#       MAIN
# -----------------------
def main():
    print("press Ctrl+C or close the plot to stop.\n")

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
