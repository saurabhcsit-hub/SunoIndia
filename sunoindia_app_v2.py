import os
import torch
import numpy as np
import streamlit as st
from pydub import AudioSegment
from demucs.pretrained import get_model
from demucs.apply import apply_model

# 🎨 Streamlit Page Config
st.set_page_config(page_title="SunoIndia - Vocal & Instrumental Splitter", 
                   page_icon="🎵", 
                   layout="centered")

# -------------------
# Session state init
# -------------------
if "users" not in st.session_state:
    # Default one user
    st.session_state["users"] = {"admin": "123"}

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "guest" not in st.session_state:
    st.session_state["guest"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

# -------------------
# Landing Page
# -------------------
def landing_page():
    st.markdown(
        """
        <div style="text-align:center; padding:30px;">
            <h1 style="font-size:50px; color:#ff4081; font-weight:bold;">🎶 SunoIndia 🎶</h1>
            <h3 style="color:#00bcd4;">Separate Vocals & Instrumentals from any song instantly!</h3>
            <p style="color:gray;">Your AI-powered music companion</p>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-top:40px;">
                <div style="background:#ff8a80; padding:30px; border-radius:20px; color:white;">🎤 Vocals</div>
                <div style="background:#82b1ff; padding:30px; border-radius:20px; color:white;">🎸 Instrumentals</div>
                <div style="background:#a7ffeb; padding:30px; border-radius:20px; color:black;">🥁 Drums</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("👉 Please login, signup, or continue without login")

    tab1, tab2, tab3 = st.tabs(["🔑 Login", "📝 Signup", "🚀 Guest"])

    # Login form
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if username in st.session_state["users"] and st.session_state["users"][username] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = username
                    st.success(f"✅ Welcome back {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

    # Signup form
    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")
            signup_btn = st.form_submit_button("Signup")

            if signup_btn:
                if new_user in st.session_state["users"]:
                    st.warning("⚠️ Username already exists. Please choose another.")
                elif new_user.strip() == "" or new_pass.strip() == "":
                    st.error("❌ Username and password cannot be empty")
                else:
                    st.session_state["users"][new_user] = new_pass
                    st.success("🎉 Signup successful! You can now log in.")

    # Guest access
    with tab3:
        if st.button("Continue as Guest"):
            st.session_state["guest"] = True
            st.session_state["current_user"] = "Guest"
            st.rerun()

# -------------------
# Music Splitter Page
# -------------------
def music_splitter():
    st.title("🎶 SunoIndia")

    if st.session_state["guest"]:
        st.subheader("Welcome Guest 👋 (Limited access)")
    else:
        st.subheader(f"Welcome {st.session_state['current_user']} 👋")

    # Logout button
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["guest"] = False
        st.session_state["current_user"] = None
        st.rerun()

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

        vocals, instrumental = None, None
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
                    instrumental += audio_int16.astype(np.int32)

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


# -------------------
# Routing
# -------------------
if st.session_state["logged_in"] or st.session_state["guest"]:
    music_splitter()
else:
    landing_page()
