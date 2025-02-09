# app.py
import streamlit as st
import audio_engine  # the module above


def main():
    st.title("Signal Processor!")
    # Start the audio stream once
    if st.button("apply audio effect"):
        audio_engine.start_audio()
        st.write("Audio stream started.")
    audio_engine.start_plot()


    # Slider for distortion amount
    # This will update the global 'distortion_drive' in audio_engine
    audio_engine.distortion_drive = st.slider(
        "Distortion Drive",
        .01,   # min
        .9,   # max
        audio_engine.distortion_drive,  # current
        0.01,  # step
    )
    st.write("Current Distortion Drive:", audio_engine.distortion_drive)


if __name__ == "__main__":
    main()
    audio_engine.start_plot()
