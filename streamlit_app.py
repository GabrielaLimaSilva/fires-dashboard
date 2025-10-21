filename = 'Hear the Fire'
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pydub import AudioSegment
from pydub.generators import Sine, Square, Sawtooth, Triangle
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips, AudioClip
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from datetime import datetime, timedelta
from pydub.utils import which

# 🔧 Fix ffmpeg and ffprobe path in remote environment (like Streamlit Cloud)
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# -------------------
# Streamlit Configuration
# -------------------
st.set_page_config(
    page_title=f'{filename}',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for fire-themed design
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        :root {
            --fire-red: #ff4444;
            --fire-orange: #ff8c00;
            --fire-yellow: #ffd700;
            --dark-smoke: #0f0f1e;
            --darker-smoke: #0a0a14;
        }

        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        body {
            background: #0a0a14;
            color: #f5f5f5;
            overflow: hidden;
        }

        .main-header {
            background: linear-gradient(135deg, #ff4444 0%, #ff8c00 50%, #ffd700 100%);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px rgba(255, 68, 68, 0.4);
            position: relative;
            overflow: hidden;
        }

        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.1) 0%, transparent 50%);
            pointer-events: none;
        }

        .main-header h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            color: white;
            font-size: 28px;
            font-weight: 700;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            letter-spacing: -0.5px;
        }

        .main-header p {
            position: relative;
            z-index: 1;
            margin: 0.3rem 0 0 0;
            color: rgba(255,255,255,0.9);
            font-size: 13px;
            font-weight: 400;
        }

        .stat-card {
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.12) 0%, rgba(255, 140, 0, 0.08) 100%);
            padding: 0.6rem 0.8rem;
            border-radius: 10px;
            border-left: 3px solid #ff4444;
            box-shadow: 0 2px 8px rgba(255, 68, 68, 0.2);
            backdrop-filter: blur(10px);
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
        }

        .stat-card:hover {
            transform: translateX(3px);
            box-shadow: 0 4px 12px rgba(255, 68, 68, 0.3);
        }

        .metric-label {
            font-size: 9px;
            color: #ff8c00;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }

        .metric-value {
            font-size: 16px;
            color: #ffd700;
            font-weight: 700;
            text-shadow: 0 0 8px rgba(255, 212, 0, 0.4);
        }

        .video-container {
            background: #000;
            border-radius: 16px;
            overflow: visible;
            box-shadow: 0 12px 40px rgba(255, 68, 68, 0.5);
            border: 2px solid rgba(255, 140, 0, 0.3);
            height: calc(100vh - 220px);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.5rem;
        }

        .video-container video {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 12px;
        }
        
        /* Garantir que os controles do vídeo fiquem visíveis */
        video::-webkit-media-controls-panel {
            background: rgba(0, 0, 0, 0.8) !important;
        }

        .stButton>button {
            background: linear-gradient(135deg, #ff4444 0%, #ff8c00 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 12px !important;
            box-shadow: 0 4px 15px rgba(255, 68, 68, 0.4) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255, 68, 68, 0.6) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%) !important;
            border-right: 1px solid rgba(255, 140, 0, 0.2);
        }

        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
        }

        .sidebar-section {
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.1) 0%, rgba(255, 140, 0, 0.05) 100%);
            padding: 0.8rem;
            border-radius: 10px;
            border-left: 3px solid #ff4444;
            margin-bottom: 0.8rem;
        }

        .stNumberInput>div>div>input,
        .stSlider>div>div>div>div {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 140, 0, 0.3) !important;
            border-radius: 8px !important;
            color: white !important;
        }

        .info-box {
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%);
            border-left: 4px solid #ff4444;
            padding: 0.8rem;
            border-radius: 10px;
            margin: 0.8rem 0;
            font-size: 12px;
            line-height: 1.4;
        }

        .info-box strong {
            color: #ffd700;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 0.8rem;
        }

        .stSpinner > div {
            border-top-color: #ff4444 !important;
        }

        .element-container {
            margin-bottom: 0 !important;
        }

        [data-testid="stStatusWidget"] {
            background: rgba(255, 68, 68, 0.1) !important;
            border-radius: 10px !important;
            border-left: 3px solid #ff8c00 !important;
        }
    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# -------------------
# IMPROVED AUDIO FUNCTIONS
# -------------------

def generate_tone(frequency, duration_ms, waveform='sine', amplitude=0.5):
    """Gera diferentes timbres baseados na forma de onda."""
    if waveform == 'sine':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    elif waveform == 'square':
        tone = Square(frequency).to_audio_segment(duration=duration_ms)
    elif waveform == 'sawtooth':
        tone = Sawtooth(frequency).to_audio_segment(duration=duration_ms)
    elif waveform == 'triangle':
        tone = Triangle(frequency).to_audio_segment(duration=duration_ms)
    elif waveform == 'complex':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        tone = tone.overlay(Sine(frequency * 2).to_audio_segment(duration=duration_ms) - 15)
        tone = tone.overlay(Sine(frequency * 3).to_audio_segment(duration=duration_ms) - 22)
    elif waveform == 'pad':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        tone = tone.overlay(Sine(frequency * 1.01).to_audio_segment(duration=duration_ms) - 8)
        tone = tone.overlay(Sine(frequency * 0.99).to_audio_segment(duration=duration_ms) - 8)
        tone = tone.overlay(Sine(frequency * 2).to_audio_segment(duration=duration_ms) - 18)
    else:
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    
    tone = tone.apply_gain(-50 + amplitude * 25)
    return tone


def create_ambient_layer(duration_ms, intensity=0.3):
    """Cria uma camada de som ambiente que evolui lentamente."""
    drone1 = Sine(55).to_audio_segment(duration=duration_ms)
    drone2 = Sine(82.4).to_audio_segment(duration=duration_ms)
    
    noise = AudioSegment.silent(duration=duration_ms)
    for _ in range(3):
        freq = np.random.uniform(150, 250)
        noise_tone = Sine(freq).to_audio_segment(duration=duration_ms)
        noise_tone = noise_tone.apply_gain(-55 + np.random.uniform(-3, 3))
        noise = noise.overlay(noise_tone)
    
    ambient = drone1.apply_gain(-45 + intensity * 10)
    ambient = ambient.overlay(drone2.apply_gain(-48 + intensity * 10))
    ambient = ambient.overlay(noise.apply_gain(-50))
    
    ambient = ambient.fade_in(int(duration_ms * 0.4)).fade_out(int(duration_ms * 0.4))
    
    return ambient


def create_bass_line(root_freq, duration_ms, pattern='pulse'):
    """Cria uma linha de baixo rítmica."""
    bass = AudioSegment.silent(duration=duration_ms)
    
    if pattern == 'pulse':
        pulse_duration = min(150, duration_ms // 4)
        for i in range(3):
            pos = int(i * duration_ms / 3)
            note = Sine(root_freq / 2).to_audio_segment(duration=pulse_duration)
            note = note.apply_gain(-35).fade_in(10).fade_out(100)
            bass = bass.overlay(note, position=pos)
    
    elif pattern == 'walking':
        notes = [root_freq / 2, root_freq / 2 * 1.125, root_freq / 2 * 1.25, root_freq / 2 * 1.125]
        note_duration = duration_ms // len(notes)
        for i, freq in enumerate(notes):
            note = Sine(freq).to_audio_segment(duration=note_duration)
            note = note.apply_gain(-38).fade_in(20).fade_out(50)
            bass = bass.overlay(note, position=i * note_duration)
    
    elif pattern == 'sustained':
        note = Sine(root_freq / 2).to_audio_segment(duration=duration_ms)
        note = note.apply_gain(-40).fade_in(100).fade_out(200)
        bass = bass.overlay(note)
    
    return bass


def create_rhythm_layer(duration_ms, intensity=0.5, pattern='ambient'):
    """Cria camada rítmica sutil."""
    rhythm = AudioSegment.silent(duration=duration_ms)
    
    if pattern == 'ambient':
        num_hits = int(3 + intensity * 2)
        for i in range(num_hits):
            pos = int(i * duration_ms / num_hits)
            hit = Sine(800 + i * 100).to_audio_segment(duration=60)
            hit = hit.apply_gain(-45 + intensity * 5).fade_out(50)
            rhythm = rhythm.overlay(hit, position=pos)
    
    elif pattern == 'groove':
        beat_duration = duration_ms // 4
        for i in range(4):
            pos = i * beat_duration
            if i % 2 == 0:
                kick = Sine(60).to_audio_segment(duration=80)
                kick = kick.apply_gain(-40).fade_out(60)
                rhythm = rhythm.overlay(kick, position=pos)
            hat = Sine(3000).to_audio_segment(duration=30)
            hat = hat.apply_gain(-48).fade_out(25)
            rhythm = rhythm.overlay(hat, position=pos)
    
    return rhythm


def create_melodic_phrase(base_freq, duration_ms, scale_notes, phrase_type='ascending'):
    """Cria uma frase melódica."""
    melody = AudioSegment.silent(duration=duration_ms)
    
    if phrase_type == 'ascending':
        note_indices = [0, 2, 4, 6]
    elif phrase_type == 'descending':
        note_indices = [6, 4, 2, 0]
    elif phrase_type == 'wave':
        note_indices = [0, 3, 6, 3]
    else:
        note_indices = [0, 2, 3, 5]
    
    note_duration = duration_ms // len(note_indices)
    
    for i, idx in enumerate(note_indices):
        freq = base_freq * (scale_notes[idx % len(scale_notes)] / scale_notes[0])
        note = generate_tone(freq, note_duration, 'sine', 0.4)
        note = note.fade_in(50).fade_out(100)
        melody = melody.overlay(note, position=i * note_duration)
    
    return melody


def compose_fire_symphony(fires_per_day_df, total_duration_sec=14):
    """Compõe a trilha sonora completa com estrutura musical."""
    n_days = len(fires_per_day_df)
    duration_per_day_ms = int((total_duration_sec * 1000) / n_days)
    
    scale_notes = [
        261.63, 293.66, 329.63, 392.00, 440.00,
        523.25, 587.33, 659.25, 783.99, 880.00
    ]
    
    max_fires = fires_per_day_df['n_fires'].max()
    min_fires = fires_per_day_df['n_fires'].min()
    mean_fires = fires_per_day_df['n_fires'].mean()
    
    intro_days = min(2, n_days // 4)
    outro_days = min(2, n_days // 4)
    
    ambient_layer = create_ambient_layer(total_duration_sec * 1000, intensity=0.25)
    
    melody_segments = []
    bass_segments = []
    rhythm_segments = []
    
    root_note_idx = 0 if mean_fires < (max_fires * 0.4) else 2
    
    for day_idx, (day, n_fires) in enumerate(fires_per_day_df.values):
        intensity = np.interp(n_fires, [min_fires, max_fires], [0.2, 0.8])
        
        if day_idx < intro_days:
            section = 'intro'
        elif day_idx >= n_days - outro_days:
            section = 'outro'
        else:
            section = 'main'
        
        note_idx = int(np.interp(intensity, [0, 1], [0, len(scale_notes) - 3]))
        note_idx = np.clip(note_idx, 0, len(scale_notes) - 3)
        
        base_freq = scale_notes[note_idx]
        
        if section == 'intro':
            intervals = [1, 1.5, 2]
            waveform = 'pad'
            chord_amplitude = 0.25 + intensity * 0.15
        elif section == 'outro':
            intervals = [1, 1.25, 1.5]
            waveform = 'sine'
            chord_amplitude = 0.3 - (day_idx - (n_days - outro_days)) * 0.05
        else:
            if intensity < 0.4:
                intervals = [1, 1.25, 1.5]
            elif intensity < 0.7:
                intervals = [1, 1.25, 1.5, 2]
            else:
                intervals = [1, 1.2, 1.5, 1.8, 2]
            waveform = 'pad' if intensity > 0.5 else 'sine'
            chord_amplitude = 0.3 + intensity * 0.2
        
        chord = AudioSegment.silent(duration=duration_per_day_ms)
        pan_positions = [-0.3, 0, 0.3, -0.15, 0.15]
        
        frequencies = [base_freq * x for x in intervals]
        
        for i, freq in enumerate(frequencies):
            note = generate_tone(freq, duration_per_day_ms, waveform, chord_amplitude)
            attack = int(duration_per_day_ms * 0.15)
            release = int(duration_per_day_ms * 0.6)
            note = note.fade_in(attack).fade_out(release)
            note = note.pan(pan_positions[i % len(pan_positions)])
            chord = chord.overlay(note)
        
        if intensity > 0.6 and section == 'main':
            delay_ms = int(duration_per_day_ms * 0.4)
            chord = chord.overlay(chord - 10, position=delay_ms)
        
        melody_segments.append(chord)
        
        if section == 'intro':
            bass_pattern = 'sustained'
        elif section == 'outro':
            bass_pattern = 'sustained'
        elif intensity > 0.6:
            bass_pattern = 'walking'
        else:
            bass_pattern = 'pulse'
        
        bass = create_bass_line(base_freq, duration_per_day_ms, bass_pattern)
        bass_segments.append(bass)
        
        if section == 'intro' or section == 'outro':
            rhythm_pattern = 'ambient'
        else:
            rhythm_pattern = 'groove' if intensity > 0.5 else 'ambient'
        
        rhythm = create_rhythm_layer(duration_per_day_ms, intensity, rhythm_pattern)
        rhythm_segments.append(rhythm)
        
        if day_idx > 0 and day_idx % 3 == 0 and section == 'main':
            prev_intensity = np.interp(
                fires_per_day_df.iloc[day_idx - 1]['n_fires'],
                [min_fires, max_fires],
                [0.2, 0.8]
            )
            
            if intensity > prev_intensity:
                phrase = create_melodic_phrase(base_freq * 2, duration_per_day_ms, scale_notes, 'ascending')
                phrase = phrase - 12
                chord = chord.overlay(phrase)
            elif intensity < prev_intensity - 0.2:
                phrase = create_melodic_phrase(base_freq * 2, duration_per_day_ms, scale_notes, 'descending')
                phrase = phrase - 12
                chord = chord.overlay(phrase)
    
    melody_track = sum(melody_segments)
    bass_track = sum(bass_segments)
    rhythm_track = sum(rhythm_segments)
    
    final_mix = melody_track
    final_mix = final_mix.overlay(bass_track - 2)
    final_mix = final_mix.overlay(rhythm_track - 5)
    final_mix = final_mix.overlay(ambient_layer - 6)
    
    intro_fade = int(total_duration_sec * 1000 * 0.08)
    outro_fade = int(total_duration_sec * 1000 * 0.15)
    
    final_mix = final_mix.fade_in(intro_fade).fade_out(outro_fade)
    final_mix = final_mix.apply_gain(-2)
    final_mix = final_mix.normalize(headroom=0.5)
    
    reverb = final_mix - 20
    final_mix = final_mix.overlay(reverb, position=80)
    
    return final_mix


# -------------------
# Helper Functions
# -------------------
def distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# -------------------
# Header
# -------------------
st.markdown("""
    <div class="main-header">
        <h1>🔥 Hear the Fire</h1>
        <p>Transform fire data into an immersive audiovisual experience</p>
    </div>
""", unsafe_allow_html=True)

# -------------------
# Sidebar Configuration
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
    data_date = st.date_input("Start date", value=datetime(2019, 8, 14))
    data_date = data_date.strftime("%Y-%m-%d")
with col2:
    day_range = st.number_input("Days to retrieve", value=7, min_value=1, max_value=30)

st.sidebar.markdown('<div class="sidebar-section"><strong>🎵 Audio</strong></div>', unsafe_allow_html=True)
total_duration_sec = st.sidebar.slider("Total duration (sec)", min_value=5, max_value=60, value=14, step=1)

st.sidebar.markdown("---")

st.sidebar.markdown("#### 📋 Summary")
st.sidebar.markdown(f"""
    <div class="stat-card">
        <div class="metric-label">Location</div>
        <div class="metric-value" style="font-size: 13px;">{latitude_center:.1f}°, {longitude_center:.1f}°</div>
    </div>
    <div class="stat-card">
        <div class="metric-label">Radius</div>
        <div class="metric-value" style="font-size: 13px;">{radius_km} km</div>
    </div>
    <div class="stat-card">
        <div class="metric-label">Period</div>
        <div class="metric-value" style="font-size: 13px;">{day_range} days</div>
    </div>
    <div class="stat-card">
        <div class="metric-label">Duration</div>
        <div class="metric-value" style="font-size: 13px;">{total_duration_sec} sec</div>
    </div>
""", unsafe_allow_html=True)

os.makedirs("maps_png", exist_ok=True)

# -------------------
# Main Content - Layout Otimizado
# -------------------
col_left, col_right = st.columns([1, 3], gap="medium")

with col_left:
    st.markdown("""
        <div class="info-box">
            <strong>🎵 How it works:</strong> Each day becomes a musical chord. 
            More fires = richer sound with bass and rhythm. 
            <strong>Listen to the data.</strong>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔥 GENERATE", key="generate_btn"):
        st.session_state['generate_clicked'] = True
    
    if 'video_file' in st.session_state:
        st.markdown("#### 📊 Stats")
        
        if 'stats_data' in st.session_state:
            stats = st.session_state['stats_data']
            
            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stat-card">
                    <div class="metric-label">🔥 Total Fires</div>
                    <div class="metric-value">{stats['total']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stat-card">
                    <div class="metric-label">📊 Days</div>
                    <div class="metric-value">{stats['days']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stat-card">
                    <div class="metric-label">📈 Avg/Day</div>
                    <div class="metric-value">{stats['avg']:.0f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stat-card">
                    <div class="metric-label">⚡ Peak</div>
                    <div class="metric-value">{stats['peak']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("#### 💾 Download")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            if 'mp3_file' in st.session_state:
                with open(st.session_state['mp3_file'], "rb") as f:
                    st.download_button(
                        label="🎵 MP3",
                        data=f.read(),
                        file_name=st.session_state['mp3_file'],
                        mime="audio/mpeg",
                        use_container_width=True
                    )
        
        with col_d2:
            with open(st.session_state['video_file'], "rb") as f:
                st.download_button(
                    label="🎬 MP4",
                    data=f.read(),
                    file_name=st.session_state['video_file'],
                    mime="video/mp4",
                    use_container_width=True
                )

with col_right:
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        # Mostrar status de geração na área do vídeo
        st.markdown("""
            <div class="video-container">
                <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.8);">
                    <h2 style="color: #ffd700; margin-bottom: 1rem;">⏳ Generating Your Experience...</h2>
                    <p style="font-size: 14px;">This may take a few minutes. Please wait.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif 'video_file' in st.session_state and os.path.exists(st.session_state['video_file']):
        st.markdown("### 🎬 Your Creation")
        st.video(st.session_state['video_file'])
    else:
        st.markdown("""
            <div class="video-container">
                <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.5);">
                    <h2 style="color: #ffd700; margin-bottom: 1rem;">🎬 Your Video Will Appear Here</h2>
                    <p style="font-size: 14px;">Configure the parameters in the sidebar and click "GENERATE" to create your audiovisual experience.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -------------------
# Processing
# -------------------
if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
    if not map_key:
        with col_right:
            st.error("❌ Please enter your FIRMS API key!")
    else:
        try:
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}"
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                with col_right:
                    st.error(f"❌ Error fetching data: {response.status_code}")
            else:
                df = pd.read_csv(StringIO(response.text))
                df.columns = df.columns.str.strip().str.lower()

                lat_col = None
                lon_col = None

                for col in df.columns:
                    if 'lat' in col:
                        lat_col = col
                    if 'lon' in col:
                        lon_col = col

                if lat_col is None or lon_col is None:
                    with col_right:
                        st.error(f"❌ Columns not found. Available columns: {list(df.columns)}")
                    st.stop()

                df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
                df_local = df[df['dist_km'] <= radius_km].copy()

                if df_local.empty:
                    with col_right:
                        st.warning("⚠️ No fires found in this area and period.")
                else:
                    fires_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
                    total_fires = len(df_local)
                    avg_fires_per_day = df_local.groupby('acq_date').size().mean()
                    max_fires_day = fires_per_day['n_fires'].max()

                    st.session_state['stats_data'] = {
                        'total': total_fires,
                        'days': len(fires_per_day),
                        'avg': avg_fires_per_day,
                        'peak': max_fires_day
                    }

                    all_days = fires_per_day['acq_date'].tolist()
                    n_days = len(fires_per_day)
                    n_fade_frames = 10

                    melody = compose_fire_symphony(fires_per_day, total_duration_sec)
                    file_name = "fires_epic_sound.mp3"
                    melody.export(file_name, format="mp3", bitrate="192k")
                    st.session_state['mp3_file'] = file_name

                            lon_min = longitude_center - radius_km/100
                            lon_max = longitude_center + radius_km/100
                            lat_min = latitude_center - radius_km/100
                            lat_max = latitude_center + radius_km/100
                            images_files = []
                            
                    intro_frames = 30
                    
                                        
                    for i in range(intro_frames):
                        progress = (i + 1) / intro_frames
                    
                        fig = plt.figure(figsize=(20, 15), dpi=200)
                        fig.patch.set_facecolor('black')
                        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
                        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
                        ax_bar = fig.add_subplot(gs[1])
                    
                        fig.patch.set_facecolor('#000000')
                        ax_map.set_facecolor('black')
                        ax_bar.set_facecolor('black')
                    
                        ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                        ax_map.add_feature(cfeature.LAND, facecolor='none', edgecolor='gray', linewidth=0.8)
                        ax_map.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
                        ax_map.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5)
                        ax_map.set_xticks([])
                        ax_map.set_yticks([])
                    
                        ax_map.plot(longitude_center, latitude_center, 'ro', markersize=15,
                                    transform=ccrs.PlateCarree(), alpha=0.8)
                    
                        current_radius_km = radius_km * progress
                        lat_deg_radius = current_radius_km / 111
                        lon_deg_radius = current_radius_km / (111 * np.cos(np.radians(latitude_center)))
                    
                        theta = np.linspace(0, 2*np.pi, 100)
                        lat_circle = latitude_center + lat_deg_radius * np.sin(theta)
                        lon_circle = longitude_center + lon_deg_radius * np.cos(theta)
                    
                        ax_map.plot(lon_circle, lat_circle, 'r-', linewidth=2,
                                    transform=ccrs.PlateCarree(), alpha=0.7)
                    
                        if progress > 0.7:
                            lat_end = latitude_center + lat_deg_radius * np.sin(np.pi/4)
                            lon_end = longitude_center + lon_deg_radius * np.cos(np.pi/4)
                    
                            ax_map.plot([longitude_center, lon_end], [latitude_center, lat_end],
                                        'y-', linewidth=3, transform=ccrs.PlateCarree(), alpha=0.8)
                    
                            mid_lat = (latitude_center + lat_end)/2
                            mid_lon = (longitude_center + lon_end)/2
                            ax_map.text(mid_lon, mid_lat, f'{radius_km} km',
                                        color='white', fontsize=16, fontweight='bold',
                                        transform=ccrs.PlateCarree(), ha='center', va='center',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
                    
                        ax_bar.set_facecolor('black')
                        ax_bar.set_xlim(0, 1)
                        ax_bar.set_ylim(0, 1)
                        ax_bar.set_xticks([])
                        ax_bar.set_yticks([])
                        for spine in ax_bar.spines.values():
                            spine.set_visible(False)
                        for spine in ax_map.spines.values():
                            spine.set_visible(False)
                    
                        png_file = f"maps_png/intro_{i}.png"
                        fig.savefig(png_file, facecolor='#000000', dpi=100, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                    
                        img = Image.open(png_file).convert("RGB")
                        final_img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0,0,0))
                        img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                        offset = ((TARGET_WIDTH - img.width)//2, (TARGET_HEIGHT - img.height)//2)
                        final_img.paste(img, offset)
                        final_img.save(png_file, quality=95)
                                                images_files.append(png_file)
                    
                    for i, (day, n_fires) in enumerate(fires_per_day.values):
                        df_day = df_local[df_local['acq_date'] == day]
                        frp_norm = np.zeros(len(df_day))
                        if 'frp' in df_day.columns and not df_day['frp'].isna().all():
                            frp_norm = (df_day['frp'] - df_day['frp'].min()) / (df_day['frp'].max() - df_day['frp'].min() + 1e-6)
                    
                        for k in range(n_fade_frames):
                            alpha = (k+1)/n_fade_frames
                    
                            fig = plt.figure(figsize=(20, 15), dpi=200)
                            fig.patch.set_facecolor('black')
                            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
                            ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
                            ax_bar = fig.add_subplot(gs[1])
                    
                            fig.patch.set_facecolor('#000000')
                            ax_map.set_facecolor('black')
                            ax_bar.set_facecolor('black')
                    
                            ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                            ax_map.add_feature(cfeature.LAND, facecolor='none', edgecolor='gray', linewidth=0.8)
                            ax_map.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
                            ax_map.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5)
                            ax_map.set_xticks([])
                            ax_map.set_yticks([])
                    
                            scatter = ax_map.scatter(
                                df_day[lon_col],
                                df_day[lat_col],
                                c=frp_norm,
                                cmap='hot',
                                s=200 + 100 * np.sin(alpha * np.pi),
                                alpha=0.7 + 0.3*alpha,
                                linewidths=2,
                                edgecolors='yellow',
                                transform=ccrs.PlateCarree(),
                                marker='o'
                            )
                    
                            if len(df_day) > 0:
                                high_intensity = df_day[df_day['frp'] > df_day['frp'].quantile(0.7)] if 'frp' in df_day.columns else df_day
                                if len(high_intensity) > 0:
                                    ax_map.scatter(
                                        high_intensity[lon_col],
                                        high_intensity[lat_col],
                                        c='white',
                                        s=300,
                                        alpha=0.3*alpha,
                                        linewidths=1,
                                        edgecolors='orange',
                                        transform=ccrs.PlateCarree(),
                                        marker='*'
                                    )
                    
                            bar_heights = [
                                fires_per_day.loc[fires_per_day['acq_date']==d,'n_fires'].values[0]
                                if d<=day else 0
                                for d in all_days
                            ]
                            colors = ['orangered' if d<=day else 'gray' for d in all_days]
                            bars = ax_bar.bar(all_days, bar_heights, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
                            
                            for bar, height in zip(bars, bar_heights):
                                if height > 0:
                                    bar.set_linewidth(1.5)
                                    bar.set_edgecolor('#ffd700')
                            
                            ax_bar.tick_params(colors='white', labelsize=12)
                            ax_bar.set_ylabel('Number of Fires', color='white', fontsize=14, fontweight='bold')
                            ax_bar.set_xlabel('Date', color='white', fontsize=14, fontweight='bold')
                            ax_bar.set_ylim(0, fires_per_day['n_fires'].max()*1.2)
                            ax_bar.grid(axis='y', alpha=0.2, linestyle='--', color='gray')
                            ax_bar.set_facecolor('#0a0a0a')
                            plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')
                            for spine in ax_bar.spines.values():
                                spine.set_color('#ff8c00')
                                spine.set_linewidth(1.5)
                            for spine in ax_map.spines.values():
                                spine.set_visible(False)
                            ax_map.tick_params(left=False, right=False, top=False, bottom=False)
                    
                            png_file = f"maps_png/map_{i}_{k}.png"
                            fig.savefig(png_file, facecolor='#000000', bbox_inches='tight', pad_inches=0)
                            plt.close(fig)
                    
                            img = Image.open(png_file).convert("RGB")
                            final_img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0,0,0))
                            img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                            offset = ((TARGET_WIDTH - img.width)//2, (TARGET_HEIGHT - img.height)//2)
                            final_img.paste(img, offset)
                            final_img.save(png_file)
                            images_files.append(png_file)

                    intro_duration = 4.0
                    fires_duration = total_duration_sec

                    intro_frame_duration = intro_duration / intro_frames
                    fires_frame_count = len(images_files) - intro_frames
                    fires_frame_duration = fires_duration / fires_frame_count if fires_frame_count > 0 else 0.1

                    frame_durations = [intro_frame_duration] * intro_frames + [fires_frame_duration] * fires_frame_count

                    clip = ImageSequenceClip(images_files, durations=frame_durations)
                    clip = clip.on_color(size=(1920,1080), color=(0,0,0))
                    
                    audio_clip = AudioFileClip(file_name)
                    
                    def make_frame(t):
                        return [0, 0]
                    
                    silent_audio = AudioClip(make_frame, duration=intro_duration, fps=44100)
                    
                    full_audio = concatenate_audioclips([silent_audio, audio_clip])
                    clip = clip.set_audio(full_audio)
                    clip.fps = 24

                    mp4_file = "fires_cinematic.mp4"
                    clip.write_videofile(mp4_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                    st.session_state['video_file'] = mp4_file
                    st.session_state['generate_clicked'] = False

                    st.rerun()

        except Exception as e:
            with col_right:
                st.error(f"❌ Error: {str(e)}")Carree(),
                                        marker='o'
                                    )
                            
                                    if len(df_day) > 0:
                                        high_intensity = df_day[df_day['frp'] > df_day['frp'].quantile(0.7)] if 'frp' in df_day.columns else df_day
                                        if len(high_intensity) > 0:
                                            ax_map.scatter(
                                                high_intensity[lon_col],
                                                high_intensity[lat_col],
                                                c='white',
                                                s=300,
                                                alpha=0.3*alpha,
                                                linewidths=1,
                                                edgecolors='orange',
                                                transform=ccrs.PlateCarree(),
                                                marker='*'
                                            )
                            
                                    bar_heights = [
                                        fires_per_day.loc[fires_per_day['acq_date']==d,'n_fires'].values[0]
                                        if d<=day else 0
                                        for d in all_days
                                    ]
                                    colors = ['orangered' if d<=day else 'gray' for d in all_days]
                                    bars = ax_bar.bar(all_days, bar_heights, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
                                    
                                    for bar, height in zip(bars, bar_heights):
                                        if height > 0:
                                            bar.set_linewidth(1.5)
                                            bar.set_edgecolor('#ffd700')
                                    
                                    ax_bar.tick_params(colors='white', labelsize=12)
                                    ax_bar.set_ylabel('Number of Fires', color='white', fontsize=14, fontweight='bold')
                                    ax_bar.set_xlabel('Date', color='white', fontsize=14, fontweight='bold')
                                    ax_bar.set_ylim(0, fires_per_day['n_fires'].max()*1.2)
                                    ax_bar.grid(axis='y', alpha=0.2, linestyle='--', color='gray')
                                    ax_bar.set_facecolor('#0a0a0a')
                                    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')
                                    for spine in ax_bar.spines.values():
                                        spine.set_color('#ff8c00')
                                        spine.set_linewidth(1.5)
                                    for spine in ax_map.spines.values():
                                        spine.set_visible(False)
                                    ax_map.tick_params(left=False, right=False, top=False, bottom=False)
                            
                                    png_file = f"maps_png/map_{i}_{k}.png"
                                    fig.savefig(png_file, facecolor='#000000', bbox_inches='tight', pad_inches=0)
                                    plt.close(fig)
                            
                                    img = Image.open(png_file).convert("RGB")
                                    final_img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0,0,0))
                                    img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                                    offset = ((TARGET_WIDTH - img.width)//2, (TARGET_HEIGHT - img.height)//2)
                                    final_img.paste(img, offset)
                                    final_img.save(png_file)
                                    images_files.append(png_file)

                            status.update(label="🎬 Compiling video...", state="running")

                            intro_duration = 4.0
                            fires_duration = total_duration_sec

                            intro_frame_duration = intro_duration / intro_frames
                            fires_frame_count = len(images_files) - intro_frames
                            fires_frame_duration = fires_duration / fires_frame_count if fires_frame_count > 0 else 0.1

                            frame_durations = [intro_frame_duration] * intro_frames + [fires_frame_duration] * fires_frame_count

                            clip = ImageSequenceClip(images_files, durations=frame_durations)
                            clip = clip.on_color(size=(1920,1080), color=(0,0,0))
                            
                            audio_clip = AudioFileClip(file_name)
                            
                            def make_frame(t):
                                return [0, 0]
                            
                            silent_audio = AudioClip(make_frame, duration=intro_duration, fps=44100)
                            
                            full_audio = concatenate_audioclips([silent_audio, audio_clip])
                            clip = clip.set_audio(full_audio)
                            clip.fps = 24

                            mp4_file = "fires_cinematic.mp4"
                            clip.write_videofile(mp4_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                            st.session_state['video_file'] = mp4_file
                            st.session_state['generate_clicked'] = False  # Limpar flag

                            status.update(label="✅ Complete!", state="complete")

                        st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
