import os
# Install FFmpeg for pydub audio processing
os.system("apt-get update && apt-get install -y ffmpeg")

import torch
import numpy as np
import streamlit as st
from pydub import AudioSegment
from demucs.pretrained import get_model
from demucs.apply import apply_model

# 🎨 Streamlit Page Config
st.set_page_config(page_title="SunoIndia - Vocal & Instrumental Splitter", page_icon="🎵", layout="centered")

# -----------------------
# Authentication System
# -----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin"}  # default user

def login_page():
    st.title("🎶 SunoIndia")
    st.subheader("Music Separation App")

    st.markdown("#### 🌈 Welcome! Please Login, Signup or Continue without Login")

    tab1, tab2, tab3 = st.tabs(["🔑 Login", "📝 Signup", "🚀 Continue without Login"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("✅ Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")

    with tab2:
        new_user = st.text_input("Choose Username", key="signup_user")
        new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        if st.button("Signup"):
            if new_user in st.session_state.users:
                st.error("⚠️ Username already exists")
            elif new_user and new_pass:
                st.session_state.users[new_user] = new_pass
                st.success("✅ Account created! Please login now.")
            else:
                st.warning("Please fill all fields")

    with tab3:
        if st.button("Continue without Login"):
            st.session_state.logged_in = True
            st.experimental_rerun()

def app_page():
    st.title("🎶 SunoIndia")
    st.subheader("Separate Vocals & Instrumentals from any song instantly!")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    uploaded_file = st.file_uploader("Upload your MP3 file", type=["mp3"])

    if uploaded_file is not None:
        input_song = "input_song.mp3"
        with open(input_song, "wb") as f:
            f.write(uploaded_file.read())

        st.info("🎵 Converting MP3 to WAV...")
        temp_wav = "temp_song.wav"
        audio = AudioSegment.from_file(input_song, format="mp3")
        audio.export(temp_wav, format="wav")

        st.info("🎵 Loading Demucs model...")
        model = get_model("htdemucs")

        st.info("🎵 Loading WAV file...")
        song = AudioSegment.from_wav(temp_wav)

        samples = np.array(song.get_array_of_samples())
        wav = torch.tensor(samples, dtype=torch.float32).view(-1, song.channels).t() / 32768.0
        sr = song.frame_rate

        st.info("🎵 Separating vocals and instrumental...")
        progress = st.progress(0, text="Processing... Please wait")

        out = apply_model(model, wav.unsqueeze(0), device="cpu", split=True)
        sources = model.sources

        vocals = None
        instrumental = None

        steps = len(sources)
        for i, (source, audio_tensor) in enumerate(zip(sources, out[0])):
            audio_np = audio_tensor.cpu().numpy()
            audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)

            if source == "vocals":
                vocals = audio_int16
            else:
                if instrumental is None:
                    instrumental = audio_int16.astype(np.int32)
                else:
                    instrumental = instrumental + audio_int16.astype(np.int32)

            percent_complete = int(((i + 1) / steps) * 100)
            progress.progress(percent_complete, text=f"Processing... {percent_complete}%")

        instrumental = np.clip(instrumental, -32768, 32767).astype(np.int16)

        def np_to_audseg(np_audio, sr):
            if np_audio.ndim == 1:
                channels = 1
            else:
                channels = np_audio.shape[0]
                np_audio = np_audio.T.flatten()
            return AudioSegment(
                np_audio.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=channels
            )

        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)

        vocals_seg = np_to_audseg(vocals, sr)
        vocals_file = os.path.join(output_folder, "vocals.mp3")
        vocals_seg.export(vocals_file, format="mp3", bitrate="192k")

        instr_seg = np_to_audseg(instrumental, sr)
        instr_file = os.path.join(output_folder, "instrumental.mp3")
        instr_seg.export(instr_file, format="mp3", bitrate="192k")

        st.success("✅ Done! Files are ready.")

        with open(vocals_file, "rb") as f:
            st.download_button("⬇️ Download Vocals", f, file_name="vocals.mp3", mime="audio/mp3")

        with open(instr_file, "rb") as f:
            st.download_button("⬇️ Download Instrumental", f, file_name="instrumental.mp3", mime="audio/mp3")

# -----------------------
# Main
# -----------------------
if st.session_state.logged_in:
    app_page()
else:
    login_page()
