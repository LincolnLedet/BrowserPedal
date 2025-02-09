import sounddevice as sd
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

distortion_drive = 0.1
samplerate = 48000
blocksize = 512
channels = 2
dtype = 'float32'

latest_block = np.zeros(blocksize, dtype=np.float32)

def basicDistortion(audio_block):
    global distortion_drive
    audio_block[audio_block > distortion_drive] = distortion_drive
    audio_block[audio_block < -distortion_drive] = -distortion_drive
    return audio_block

def process_audio(indata, outdata, frames, time_info, status):
    global latest_block
    mono_in = indata[:, 0]
    processed = basicDistortion(mono_in.copy())
    outdata[:, 0] = processed
    outdata[:, 1] = processed
    latest_block[:] = processed

def audio_thread():
    with sd.Stream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=channels,
        dtype=dtype,
        latency="low",
        callback=process_audio
    ):
        # Keep the stream running
        while True:
            sd.sleep(100)

def start_audio():
    t = threading.Thread(target=audio_thread, daemon=True)
    t.start()

def start_plot():
    fig, ax = plt.subplots()
    x = np.arange(blocksize)
    line, = ax.plot(x, np.zeros(blocksize))

    ax.set_ylim(-1, 1)
    ax.set_xlim(0, blocksize)
    ax.set_title("Live Audio Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")

    def update_plot(frame):
        line.set_ydata(latest_block)
        return line,

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        interval=50,
        blit=True,
        cache_frame_data=False  # Disables frame caching
    )
    plt.show()

# Only run audio and plotting if this script is run directly
if __name__ == "__main__":
    start_audio()
    start_plot()
