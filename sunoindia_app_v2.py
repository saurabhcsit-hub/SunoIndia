import os
# Install FFmpeg for audio processing
os.system("apt-get update && apt-get install -y ffmpeg")

import torch
import numpy as np
import streamlit as st
import librosa
import soundfile as sf
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

    # Initialize session state keys
    if "login_user" not in st.session_state:
        st.session_state.login_user = ""
    if "login_pass" not in st.session_state:
        st.session_state.login_pass = ""
    if "signup_user" not in st.session_state:
        st.session_state.signup_user = ""
    if "signup_pass" not in st.session_state:
        st.session_state.signup_pass = ""

    tab1, tab2, tab3 = st.tabs(["🔑 Login", "📝 Signup", "🚀 Continue without Login"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")

    with tab2:
        new_user = st.text_input("Choose Username", key="signup_user")
        new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        if st.button("Signup", key="signup_btn"):
            if new_user in st.session_state.users:
                st.error("⚠️ Username already exists")
            elif new_user and new_pass:
                st.session_state.users[new_user] = new_pass
                st.success("✅ Account created! Please login now.")
            else:
                st.warning("Please fill all fields")

    with tab3:
        if st.button("Continue without Login", key="continue_btn"):
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

        st.info("🎵 Loading audio...")
        y, sr = librosa.load(input_song, sr=None, mono=False)

        st.info("🎵 Loading Demucs model...")
        model = get_model("htdemucs")

        st.info("🎵 Separating vocals and instrumental...")
        progress = st.progress(0, text="Processing... Please wait")

        wav = torch.tensor(y, dtype=torch.float32).view(y.shape[0], -1) if y.ndim > 1 else torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        out = apply_model(model, wav.unsqueeze(0), device="cpu", split=True)
        sources = model.sources

        vocals = None
        instrumental = None

        steps = len(sources)
        for i, (source, audio_tensor) in enumerate(zip(sources, out[0])):
            audio_np = audio_tensor.cpu().numpy()
            if source == "vocals":
                vocals = audio_np
            else:
                instrumental = audio_np if instrumental is None else instrumental + audio_np

            percent_complete = int(((i + 1) / steps) * 100)
            progress.progress(percent_complete, text=f"Processing... {percent_complete}%")

        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)

        vocals_file = os.path.join(output_folder, "vocals.wav")
        instr_file = os.path.join(output_folder, "instrumental.wav")

        sf.write(vocals_file, vocals.T, sr)
        sf.write(instr_file, instrumental.T, sr)

        st.success("✅ Done! Files are ready.")

        with open(vocals_file, "rb") as f:
            st.download_button("⬇️ Download Vocals", f, file_name="vocals.wav", mime="audio/wav")

        with open(instr_file, "rb") as f:
            st.download_button("⬇️ Download Instrumental", f, file_name="instrumental.wav", mime="audio/wav")

# -----------------------
# Main
# -----------------------
if st.session_state.logged_in:
    app_page()
else:
    login_page()
