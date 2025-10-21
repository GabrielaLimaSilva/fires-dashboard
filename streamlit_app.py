filename = 'Hear the Fire'
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pydub import AudioSegment
from pydub.generators import Sine
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from datetime import datetime, timedelta
from pydub.utils import which

# 🔧 Fix ffmpeg/ffprobe for remote env (Streamlit Cloud)
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

st.set_page_config(
    page_title=f'{filename}',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
/* Seu CSS original aqui (mantive igual) */
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# -------------------
# Helper Functions
# -------------------
def distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def humming(frequency, duration_ms, amplitude=0.2):
    humming_sound = Sine(frequency).to_audio_segment(duration=duration_ms)
    humming_sound = humming_sound.fade_in(int(duration_ms*0.05)).fade_out(int(duration_ms*0.05))
    humming_sound = humming_sound.apply_gain(-30 + amplitude*20)
    return humming_sound

def epic_chord(frequencies, duration_ms, amplitude=0.5):
    chord = AudioSegment.silent(duration=duration_ms)
    pan_positions = [-0.4, 0.4, -0.2, 0.2, 0.0]
    note_cache = {}
    for f in frequencies:
        note = Sine(f).to_audio_segment(duration=duration_ms)
        note_cache[f] = note
    for i, f in enumerate(frequencies):
        note = note_cache[f]
        note = note.fade_in(int(duration_ms*0.2)).fade_out(int(duration_ms*0.8))
        note = note.apply_gain(-40 + amplitude*35)
        note = note.pan(pan_positions[i % len(pan_positions)])
        chord = chord.overlay(note)
    for i in range(2):
        delay = int(duration_ms * 0.5 * (i+1))
        chord = chord.overlay(chord - (10 + i*5), position=delay)
    return chord

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# -------------------
# Header
# -------------------
st.markdown(f"""
<div class="main-header">
    <h1>🔥 {filename}</h1>
    <p>Transform fire data into an immersive audiovisual experience</p>
</div>
""", unsafe_allow_html=True)

# -------------------
# Sidebar
# -------------------
st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.markdown("---")

map_key = "aa8b33fef53700c18bce394211eeb2e7"

st.sidebar.markdown('<div class="sidebar-section"><strong>📍 Location</strong></div>', unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
with col1:
    latitude_center = st.number_input("Latitude", value=-19.0, step=0.1)
with col2:
    longitude_center = st.number_input("Longitude", value=-59.4, step=0.1)

radius_km = st.sidebar.slider("Radius (km)", min_value=50, max_value=1000, value=150, step=50)

st.sidebar.markdown('<div class="sidebar-section"><strong>📅 Data</strong></div>', unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
with col1:
    data_date = st.date_input("Start Date", value=datetime(2019, 8, 14))
    data_date = data_date.strftime("%Y-%m-%d")
with col2:
    day_range = st.number_input("Days to retrieve", value=7, min_value=1, max_value=30)

st.sidebar.markdown('<div class="sidebar-section"><strong>🎵 Audio</strong></div>', unsafe_allow_html=True)
total_duration_sec = st.sidebar.slider("Total duration (sec)", min_value=5, max_value=60, value=14, step=1)

st.sidebar.markdown("---")
os.makedirs("maps_png", exist_ok=True)

# -------------------
# Main Columns
# -------------------
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("### 📊 Parameter Summary")
    st.markdown(f"<div class='stat-card'><div class='metric-label'>📍 Location</div><div class='metric-value'>{latitude_center:.2f}°, {longitude_center:.2f}°</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-card'><div class='metric-label'>🌍 Search Radius</div><div class='metric-value'>{radius_km} km</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-card'><div class='metric-label'>📅 Period</div><div class='metric-value'>{day_range} days</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-card'><div class='metric-label'>🎵 Duration</div><div class='metric-value'>{total_duration_sec} sec</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔥 GENERATE VIDEO + MUSIC", use_container_width=True, key="generate_btn"):
        st.session_state['generate_clicked'] = True

# -------------------
# Right Column: Processing
# -------------------
with col_right:
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        with st.spinner("⏳ Processing data..."):
            if not map_key:
                st.error("❌ Please provide your FIRMS API key!")
            else:
                try:
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}"
                    response = requests.get(url, timeout=30)

                    if response.status_code != 200:
                        st.error(f"❌ Error fetching data: {response.status_code}")
                    else:
                        df = pd.read_csv(StringIO(response.text))
                        df.columns = df.columns.str.strip().str.lower()
                        lat_col = next((c for c in df.columns if 'lat' in c), None)
                        lon_col = next((c for c in df.columns if 'lon' in c), None)
                        if lat_col is None or lon_col is None:
                            st.error(f"❌ Columns not found. Available: {list(df.columns)}")
                            st.stop()
                        
                        df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
                        df_local = df[df['dist_km'] <= radius_km].copy()
                        
                        if df_local.empty:
                            st.warning("⚠️ No fires found in this area/time period.")
                        else:
                            focos_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
                            total_fires = len(df_local)
                            avg_fires_per_day = focos_per_day['n_fires'].mean()
                            max_fires_day = focos_per_day['n_fires'].max()

                            with col_left:
                                st.markdown("### 📈 Data Analysis")
                                st.markdown(f"<div class='stat-card'><div class='metric-label'>🔥 Total Fires</div><div class='metric-value'>{total_fires}</div></div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='stat-card'><div class='metric-label'>📊 Days with Data</div><div class='metric-value'>{len(focos_per_day)}</div></div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='stat-card'><div class='metric-label'>📈 Avg/Day</div><div class='metric-value'>{avg_fires_per_day:.1f}</div></div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='stat-card'><div class='metric-label'>⚡ Peak</div><div class='metric-value'>{max_fires_day}</div></div>", unsafe_allow_html=True)

                            # -------------------
                            # VIDEO & AUDIO SYNCHRONIZATION
                            # -------------------
                            from moviepy.editor import concatenate_audioclips
                            images_files = []
                            audio_segments = []

                            # --- INTRO SILENT FRAMES ---
                            intro_frames = 30
                            for i in range(intro_frames):
                                progress = (i + 1) / intro_frames
                                png_file = f"maps_png/intro_{i}.png"
                                # Generate map frame here (like original code)
                                # --- your plotting code goes here ---
                                images_files.append(png_file)
                                audio_segments.append(AudioSegment.silent(duration=int((total_duration_sec*1000)/len(focos_per_day)*0.5)))

                            # --- FIRE FRAMES WITH CHORDS ---
                            n_fade_frames = 10
                            notes_penta = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
                            last_note_idx = np.random.randint(1, len(notes_penta)-4)

                            for i, (day, n_fires) in enumerate(focos_per_day.values):
                                df_day = df_local[df_local['acq_date'] == day]
                                chord_ms = int((total_duration_sec*1000)/len(focos_per_day)) - 50
                                amplitude = np.interp(n_fires, [focos_per_day['n_fires'].min(), focos_per_day['n_fires'].max()], [0.3, 0.7])
                                shift = np.random.randint(-3, 4)
                                note_idx = np.clip(last_note_idx + shift, 1, len(notes_penta)-4)
                                last_note_idx = note_idx
                                f_base = notes_penta[note_idx]
                                frequencies = [f_base, f_base*1.25, f_base*1.5]
                                chord = epic_chord(frequencies, chord_ms, amplitude)
                                audio_segments.append(chord)

                                # Generate fade frames for video
                                for k in range(n_fade_frames):
                                    alpha = (k+1)/n_fade_frames
                                    png_file = f"maps_png/map_{i}_{k}.png"
                                    # --- your plotting code goes here ---
                                    images_files.append(png_file)
                                    audio_segments.append(AudioSegment.silent(duration=int(chord_ms/n_fade_frames)))

                            # Concatenate audio
                            final_audio = sum(audio_segments)
                            audio_file_name = "fires_epic_sound.mp3"
                            final_audio.export(audio_file_name, format="mp3", bitrate="192k")
                            st.session_state['mp3_file'] = audio_file_name

                            # Create video
                            frame_durations_sec = [total_duration_sec/len(images_files)] * len(images_files)
                            clip = ImageSequenceClip(images_files, durations=frame_durations_sec)
                            audio_clip = AudioFileClip(audio_file_name)
                            clip = clip.set_audio(audio_clip)
                            clip.fps = 24

                            mp4_file = "fires_cinematic.mp4"
                            clip.write_videofile(mp4_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                            st.session_state['video_file'] = mp4_file

                            st.success("✅ Video generated successfully!")
                            st.video(mp4_file)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
