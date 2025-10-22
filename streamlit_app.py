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
from datetime import datetime
from pydub.utils import which
import hashlib
import json

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# ============= SISTEMA DE CACHE =============
CACHE_DIR = "video_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_cache_key(params):
    """Gera hash único dos parâmetros"""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()

def get_cached_files(cache_key):
    """Verifica se existem arquivos em cache"""
    video_path = os.path.join(CACHE_DIR, f"video_{cache_key}.mp4")
    audio_path = os.path.join(CACHE_DIR, f"audio_{cache_key}.mp3")
    if os.path.exists(video_path) and os.path.exists(audio_path):
        return video_path, audio_path
    return None, None

def save_to_cache(cache_key, video_src, audio_src):
    """Salva arquivos no cache"""
    import shutil
    video_dst = os.path.join(CACHE_DIR, f"video_{cache_key}.mp4")
    audio_dst = os.path.join(CACHE_DIR, f"audio_{cache_key}.mp3")
    shutil.copy2(video_src, video_dst)
    shutil.copy2(audio_src, audio_dst)
    return video_dst, audio_dst
# ============= FIM SISTEMA DE CACHE =============

st.set_page_config(page_title=f'{filename}', layout="wide", initial_sidebar_state="expanded")

# Layout moderno e compacto
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .main .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 100% !important; }
        #MainMenu, footer, header { visibility: hidden; }
        body { background: #0a0a14; color: #f5f5f5; }
        
        .main-header { background: linear-gradient(135deg, #ff4444 0%, #ff8c00 50%, #ffd700 100%); padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(255, 68, 68, 0.4); }
        .main-header h1 { margin: 0; color: white; font-size: 28px; font-weight: 700; }
        .main-header p { margin: 0.3rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 13px; }
        
        .stat-card { background: linear-gradient(135deg, rgba(255, 68, 68, 0.12) 0%, rgba(255, 140, 0, 0.08) 100%); padding: 0.6rem 0.8rem; border-radius: 10px; border-left: 3px solid #ff4444; margin-bottom: 0.5rem; }
        .metric-label { font-size: 9px; color: #ff8c00; font-weight: 600; text-transform: uppercase; }
        .metric-value { font-size: 16px; color: #ffd700; font-weight: 700; }
        
        .video-container { background: #000; border-radius: 16px; overflow: visible; box-shadow: 0 12px 40px rgba(255, 68, 68, 0.5); border: 2px solid rgba(255, 140, 0, 0.3); height: calc(100vh - 220px); display: flex; align-items: center; justify-content: center; padding: 0.5rem; }
        
        .stButton>button { background: linear-gradient(135deg, #ff4444 0%, #ff8c00 100%) !important; color: white !important; border: none !important; padding: 0.6rem 1.2rem !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important; }
        
        .info-box { background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%); border-left: 4px solid #ff4444; padding: 0.8rem; border-radius: 10px; margin: 0.8rem 0; font-size: 12px; }
        .info-box strong { color: #ffd700; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# Funções de áudio originais
def generate_tone(frequency, duration_ms, waveform='sine', amplitude=0.5):
    if waveform == 'sine':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    elif waveform == 'pad':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        tone = tone.overlay(Sine(frequency * 1.01).to_audio_segment(duration=duration_ms) - 8)
        tone = tone.overlay(Sine(frequency * 0.99).to_audio_segment(duration=duration_ms) - 8)
        tone = tone.overlay(Sine(frequency * 2).to_audio_segment(duration=duration_ms) - 18)
    elif waveform == 'triangle':
        tone = Triangle(frequency).to_audio_segment(duration=duration_ms)
    elif waveform == 'complex':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        tone = tone.overlay(Sine(frequency * 2).to_audio_segment(duration=duration_ms) - 15)
        tone = tone.overlay(Sine(frequency * 3).to_audio_segment(duration=duration_ms) - 22)
    else:
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    tone = tone.apply_gain(-50 + amplitude * 25)
    return tone

def create_ambient_layer(duration_ms, intensity=0.3):
    drone1 = Sine(55).to_audio_segment(duration=duration_ms).apply_gain(-45 + intensity * 10)
    drone2 = Sine(82.4).to_audio_segment(duration=duration_ms).apply_gain(-48 + intensity * 10)
    noise = AudioSegment.silent(duration=duration_ms)
    for _ in range(3):
        freq = np.random.uniform(150, 250)
        noise_tone = Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-55 + np.random.uniform(-3, 3))
        noise = noise.overlay(noise_tone)
    ambient = drone1.overlay(drone2).overlay(noise.apply_gain(-50))
    return ambient.fade_in(int(duration_ms * 0.4)).fade_out(int(duration_ms * 0.4))

def create_bass_line(root_freq, duration_ms, pattern='pulse'):
    bass = AudioSegment.silent(duration=duration_ms)
    if pattern == 'pulse':
        for i in range(3):
            pos = int(i * duration_ms / 3)
            note = Sine(root_freq / 2).to_audio_segment(duration=150).apply_gain(-35).fade_in(10).fade_out(100)
            bass = bass.overlay(note, position=pos)
    elif pattern == 'walking':
        notes = [root_freq / 2, root_freq / 2 * 1.125, root_freq / 2 * 1.25, root_freq / 2 * 1.125]
        note_duration = duration_ms // len(notes)
        for i, freq in enumerate(notes):
            note = Sine(freq).to_audio_segment(duration=note_duration).apply_gain(-38).fade_in(20).fade_out(50)
            bass = bass.overlay(note, position=i * note_duration)
    else:
        note = Sine(root_freq / 2).to_audio_segment(duration=duration_ms).apply_gain(-40).fade_in(100).fade_out(200)
        bass = bass.overlay(note)
    return bass

def create_rhythm_layer(duration_ms, intensity=0.5, pattern='ambient'):
    rhythm = AudioSegment.silent(duration=duration_ms)
    if pattern == 'ambient':
        num_hits = int(3 + intensity * 2)
        for i in range(num_hits):
            pos = int(i * duration_ms / num_hits)
            hit = Sine(800 + i * 100).to_audio_segment(duration=60).apply_gain(-45 + intensity * 5).fade_out(50)
            rhythm = rhythm.overlay(hit, position=pos)
    elif pattern == 'groove':
        beat_duration = duration_ms // 4
        for i in range(4):
            pos = i * beat_duration
            if i % 2 == 0:
                kick = Sine(60).to_audio_segment(duration=80).apply_gain(-40).fade_out(60)
                rhythm = rhythm.overlay(kick, position=pos)
            hat = Sine(3000).to_audio_segment(duration=30).apply_gain(-48).fade_out(25)
            rhythm = rhythm.overlay(hat, position=pos)
    return rhythm

def create_melodic_phrase(base_freq, duration_ms, scale_notes, phrase_type='ascending'):
    melody = AudioSegment.silent(duration=duration_ms)
    if phrase_type == 'ascending':
        note_indices = [0, 2, 4, 6]
    elif phrase_type == 'descending':
        note_indices = [6, 4, 2, 0]
    else:
        note_indices = [0, 2, 3, 5]
    note_duration = duration_ms // len(note_indices)
    for i, idx in enumerate(note_indices):
        freq = base_freq * (scale_notes[idx % len(scale_notes)] / scale_notes[0])
        note = generate_tone(freq, note_duration, 'sine', 0.4).fade_in(50).fade_out(100)
        melody = melody.overlay(note, position=i * note_duration)
    return melody

def compose_fire_symphony(fires_per_day_df, total_duration_sec=14):
    n_days = len(fires_per_day_df)
    duration_per_day_ms = int((total_duration_sec * 1000) / n_days)
    scale_notes = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00]
    max_fires = fires_per_day_df['n_fires'].max()
    min_fires = fires_per_day_df['n_fires'].min()
    mean_fires = fires_per_day_df['n_fires'].mean()
    intro_days = min(2, n_days // 4)
    outro_days = min(2, n_days // 4)
    ambient_layer = create_ambient_layer(total_duration_sec * 1000, intensity=0.25)
    melody_segments = []
    bass_segments = []
    rhythm_segments = []
    
    for day_idx, (day, n_fires) in enumerate(fires_per_day_df.values):
        intensity = np.interp(n_fires, [min_fires, max_fires], [0.2, 0.8])
        if day_idx < intro_days:
            section = 'intro'
        elif day_idx >= n_days - outro_days:
            section = 'outro'
        else:
            section = 'main'
        
        note_idx = int(np.interp(intensity, [0, 1], [0, len(scale_notes) - 3]))
        base_freq = scale_notes[note_idx]
        
        if section == 'intro':
            intervals = [1, 1.5, 2]
            waveform = 'pad'
            chord_amplitude = 0.25 + intensity * 0.15
        elif section == 'outro':
            intervals = [1, 1.25, 1.5]
            waveform = 'sine'
            chord_amplitude = 0.3
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
            note = note.fade_in(attack).fade_out(release).pan(pan_positions[i % len(pan_positions)])
            chord = chord.overlay(note)
        
        if intensity > 0.6 and section == 'main':
            delay_ms = int(duration_per_day_ms * 0.4)
            chord = chord.overlay(chord - 10, position=delay_ms)
        
        melody_segments.append(chord)
        bass_segments.append(create_bass_line(base_freq, duration_per_day_ms, 'walking' if intensity > 0.6 else 'pulse'))
        rhythm_segments.append(create_rhythm_layer(duration_per_day_ms, intensity, 'groove' if intensity > 0.5 else 'ambient'))
        
        if day_idx > 0 and day_idx % 3 == 0 and section == 'main':
            prev_intensity = np.interp(fires_per_day_df.iloc[day_idx - 1]['n_fires'], [min_fires, max_fires], [0.2, 0.8])
            if intensity > prev_intensity:
                phrase = create_melodic_phrase(base_freq * 2, duration_per_day_ms, scale_notes, 'ascending') - 12
                chord = chord.overlay(phrase)
    
    melody_track = sum(melody_segments)
    bass_track = sum(bass_segments)
    rhythm_track = sum(rhythm_segments)
    final_mix = melody_track.overlay(bass_track - 2).overlay(rhythm_track - 5).overlay(ambient_layer - 6)
    intro_fade = int(total_duration_sec * 1000 * 0.08)
    outro_fade = int(total_duration_sec * 1000 * 0.15)
    final_mix = final_mix.fade_in(intro_fade).fade_out(outro_fade).apply_gain(-2).normalize(headroom=0.5)
    reverb = final_mix - 20
    final_mix = final_mix.overlay(reverb, position=80)
    return final_mix

def distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

st.markdown('<div class="main-header"><h1>🔥 Hear the Fire</h1><p>Transform fire data into an immersive audiovisual experience</p></div>', unsafe_allow_html=True)

# BARRA DE PROGRESSO NO TOPO - criar placeholders sempre
progress_placeholder = st.empty()
status_placeholder = st.empty()

st.sidebar.markdown("### ⚙️ Settings")

# API Key segura - usar secrets do Streamlit
try:
    map_key = st.secrets["NASA_FIRMS_KEY"]
except:
    # Fallback para desenvolvimento local - criar arquivo .streamlit/secrets.toml
    map_key = "a4abee84e580a96ff5ba9bd54cd11a8d"


col1, col2 = st.sidebar.columns(2)
with col1:
    latitude_center = st.number_input("Latitude", value=-19.0, step=0.1)
with col2:
    longitude_center = st.number_input("Longitude", value=-59.4, step=0.1)

radius_km = st.sidebar.slider("Radius (km)", 50, 500, 150, 50)

col1, col2 = st.sidebar.columns(2)
with col1:
    data_date = st.date_input("Start date", value=datetime(2019, 8, 14)).strftime("%Y-%m-%d")
with col2:
    day_range = st.slider("Days", min_value=1, max_value=10, value=10)

total_duration_sec = 1.2*day_range

os.makedirs("maps_png", exist_ok=True)

col_left, col_right = st.columns([1, 3], gap="medium")

with col_left:
    st.markdown('<div class="info-box"><strong>🎵 How it works:</strong> Each day becomes a musical chord. More fires = richer sound. <strong>Listen to the data.</strong></div>', unsafe_allow_html=True)
    
    # Gerar cache key com parâmetros atuais
    cache_params = {
        'lat': latitude_center,
        'lon': longitude_center,
        'radius': radius_km,
        'date': data_date,
        'days': day_range
    }
    current_cache_key = generate_cache_key(cache_params)
    cached_video, cached_audio = get_cached_files(current_cache_key)
    
    # Se tem cache, carregar stats automaticamente
    if cached_video:
        stats_file = os.path.join(CACHE_DIR, f"stats_{current_cache_key}.json")
        if os.path.exists(stats_file) and 'stats_data' not in st.session_state:
            try:
                with open(stats_file, 'r') as f:
                    st.session_state['stats_data'] = json.load(f)
            except (json.JSONDecodeError, Exception):
                pass
    
    # Mostrar se tem cache disponível
    if cached_video:
        st.markdown('<div class="info-box" style="border-left-color: #00ff88;"><strong>⚡ Cache found!</strong> Video ready to load instantly.</div>', unsafe_allow_html=True)
    
    if st.button("🔥 GENERATE", key="generate_btn"):
        # Se tem cache, carrega direto
        if cached_video:
            st.session_state['video_file'] = cached_video
            st.session_state['mp3_file'] = cached_audio
            st.session_state['generate_clicked'] = False
            # Carregar stats definitivamente
            stats_file = os.path.join(CACHE_DIR, f"stats_{current_cache_key}.json")
            if os.path.exists(stats_file):
                try:
                    with open(stats_file, 'r') as f:
                        file_content = f.read()
                        if file_content.strip():  # Verificar se não está vazio
                            loaded_stats = json.loads(file_content)
                            st.session_state['stats_data'] = loaded_stats
                        else:
                            st.warning("⚠️ Stats file is empty")
                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"⚠️ Could not load stats: {e}")
            else:
                st.warning(f"⚠️ Stats file not found: {stats_file}")
            st.rerun()
        else:
            st.session_state['current_cache_key'] = current_cache_key
            st.session_state['generate_clicked'] = True
    
    if 'video_file' in st.session_state and st.session_state.get('video_file') and os.path.exists(st.session_state['video_file']):
        st.markdown("#### 📊 Stats")
        if 'stats_data' in st.session_state:
            stats = st.session_state['stats_data']
            st.markdown(f'<div class="stats-grid"><div class="stat-card"><div class="metric-label">🔥 Total</div><div class="metric-value">{stats["total"]}</div></div><div class="stat-card"><div class="metric-label">📊 Days</div><div class="metric-value">{stats["days"]}</div></div><div class="stat-card"><div class="metric-label">📈 Avg</div><div class="metric-value">{stats["avg"]:.0f}</div></div><div class="stat-card"><div class="metric-label">⚡ Peak</div><div class="metric-value">{stats["peak"]}</div></div></div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Stats not loaded in session_state")
            # Mostrar o que tem no arquivo
            stats_file = os.path.join(CACHE_DIR, f"stats_{current_cache_key}.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    content = f.read()
                    st.code(f"Stats file content:\n{content}")
            else:
                st.error(f"Stats file not found: {stats_file}")
        
        st.markdown("#### 💾 Download")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if 'mp3_file' in st.session_state and st.session_state.get('mp3_file') and os.path.exists(st.session_state['mp3_file']):
                with open(st.session_state['mp3_file'], "rb") as f:
                    st.download_button("🎵 MP3", f.read(), st.session_state['mp3_file'], "audio/mpeg", use_container_width=True)
        with col_d2:
            with open(st.session_state['video_file'], "rb") as f:
                st.download_button("🎬 MP4", f.read(), st.session_state['video_file'], "video/mp4", use_container_width=True)

with col_right:
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        st.markdown('<div class="video-container"><div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.8);"><h2 style="color: #ffd700;">⏳ Generating...</h2><p>Please wait.</p></div></div>', unsafe_allow_html=True)
    elif 'video_file' in st.session_state and st.session_state.get('video_file') and os.path.exists(st.session_state['video_file']):
        st.markdown("### 🎬 Your Creation")
        st.video(st.session_state['video_file'])
    else:
        st.markdown('<div class="video-container"><div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.5);"><h2 style="color: #ffd700;">🎬 Your Video Will Appear Here</h2><p>Configure parameters and click GENERATE.</p></div></div>', unsafe_allow_html=True)

if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
    progress_bar = progress_placeholder.progress(0)
    status_text = status_placeholder.empty()
    
    try:
        status_text.text("🔍 Fetching fire data from NASA...")
        progress_bar.progress(5)
        response = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}", timeout=30)
        progress_bar.progress(10)
        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.strip().str.lower()
        lat_col = next((c for c in df.columns if 'lat' in c), None)
        lon_col = next((c for c in df.columns if 'lon' in c), None)
        
        status_text.text("📊 Processing fire data...")
        progress_bar.progress(15)
        df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
        df_local = df[df['dist_km'] <= radius_km].copy()
        progress_bar.progress(20)
        
        if not df_local.empty:
            fires_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
            # Criar stats convertendo tipos numpy/pandas para Python nativos
            st.session_state['stats_data'] = {
                'total': int(len(df_local)), 
                'days': int(len(fires_per_day)), 
                'avg': float(fires_per_day['n_fires'].mean()), 
                'peak': int(fires_per_day['n_fires'].max())
            }
            
            status_text.text("🎵 Composing fire symphony...")
            progress_bar.progress(25)
            melody = compose_fire_symphony(fires_per_day, total_duration_sec)
            progress_bar.progress(35)
            melody.export("fires_sound.mp3", format="mp3", bitrate="192k")
            st.session_state['mp3_file'] = "fires_sound.mp3"
            progress_bar.progress(40)
            
            lon_min = longitude_center - radius_km/100
            lon_max = longitude_center + radius_km/100
            lat_min = latitude_center - radius_km/100
            lat_max = latitude_center + radius_km/100
            images_files = []
            all_days = fires_per_day['acq_date'].tolist()
            n_days = len(fires_per_day)
            n_fade_frames = 5  # Reduzido de 10 para 5
            intro_frames = 15  # Reduzido de 30 para 15
            
            status_text.text("🎬 Creating intro animation...")
            for i in range(intro_frames):
                progress = (i + 1) / intro_frames
                progress_bar.progress(40 + int(10 * progress))
                fig = plt.figure(figsize=(16, 9), dpi=100)  # Aumentado para 16:9
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
                ax_map.plot(longitude_center, latitude_center, 'ro', markersize=15, transform=ccrs.PlateCarree(), alpha=0.8)
                current_radius_km = radius_km * progress
                lat_deg_radius = current_radius_km / 111
                lon_deg_radius = current_radius_km / (111 * np.cos(np.radians(latitude_center)))
                theta = np.linspace(0, 2*np.pi, 100)
                lat_circle = latitude_center + lat_deg_radius * np.sin(theta)
                lon_circle = longitude_center + lon_deg_radius * np.cos(theta)
                ax_map.plot(lon_circle, lat_circle, 'r-', linewidth=2, transform=ccrs.PlateCarree(), alpha=0.7)
                if progress > 0.7:
                    lat_end = latitude_center + lat_deg_radius * np.sin(np.pi/4)
                    lon_end = longitude_center + lon_deg_radius * np.cos(np.pi/4)
                    ax_map.plot([longitude_center, lon_end], [latitude_center, lat_end], 'y-', linewidth=3, transform=ccrs.PlateCarree(), alpha=0.8)
                    mid_lat = (latitude_center + lat_end)/2
                    mid_lon = (longitude_center + lon_end)/2
                    ax_map.text(mid_lon, mid_lat, f'{radius_km} km', color='white', fontsize=16, fontweight='bold', transform=ccrs.PlateCarree(), ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
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
                fig.savefig(png_file, facecolor='#000000', dpi=100, bbox_inches='tight', pad_inches=0.1)  # DPI 100 + padding mínimo
                plt.close(fig)
                img = Image.open(png_file).convert("RGB")
                # Redimensionar para preencher completamente
                img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                img.save(png_file, quality=85, optimize=True)
                images_files.append(png_file)
            
            status_text.text("🔥 Rendering fire visualizations...")
            total_fire_frames = n_days * n_fade_frames
            for i, (day, n_fires) in enumerate(fires_per_day.values):
                status_text.text(f"🔥 Rendering day {i+1}/{n_days}: {day} ({n_fires} fires)")
                df_day = df_local[df_local['acq_date'] == day]
                frp_norm = np.zeros(len(df_day))
                if 'frp' in df_day.columns and not df_day['frp'].isna().all():
                    frp_norm = (df_day['frp'] - df_day['frp'].min()) / (df_day['frp'].max() - df_day['frp'].min() + 1e-6)
                for k in range(n_fade_frames):
                    frame_progress = (i * n_fade_frames + k) / total_fire_frames
                    progress_bar.progress(50 + int(40 * frame_progress))
                    alpha = (k+1)/n_fade_frames
                    fig = plt.figure(figsize=(16, 9), dpi=100)  # 16:9 aspect ratio
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
                    
                    # VISUALIZAÇÃO CINEMATOGRÁFICA DE FOGO
                    if len(df_day) > 0:
                        # Camada 1: Glow externo (vermelho escuro)
                        glow_sizes = 400 + 100 * np.sin(alpha * np.pi * 2)  # Reduzido
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#8B0000', s=glow_sizes, alpha=0.15 * alpha,
                                     transform=ccrs.PlateCarree())
                        
                        # Camada 2: Halo alaranjado médio
                        halo_sizes = 250 + 80 * np.sin(alpha * np.pi * 2)  # Reduzido
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#FF4500', s=halo_sizes, alpha=0.25 * alpha,
                                     transform=ccrs.PlateCarree())
                        
                        # Camada 3: Core laranja brilhante
                        core_sizes = 150 + 60 * np.sin(alpha * np.pi * 2)  # Reduzido
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#FF8C00', s=core_sizes, alpha=0.6 * alpha,
                                     linewidths=0, transform=ccrs.PlateCarree())
                        
                        # Camada 4: Centro amarelo intenso (variação por intensidade)
                        center_colors = plt.cm.YlOrRd(frp_norm * 0.7 + 0.3)
                        center_sizes = 80 + 50 * np.sin(alpha * np.pi * 3) * (1 + frp_norm)  # Reduzido
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c=center_colors, s=center_sizes, alpha=0.85 * alpha,
                                     edgecolors='#FFD700', linewidths=1,  # Linewidth reduzido
                                     transform=ccrs.PlateCarree())
                        
                        # Camada 5: Núcleo branco brilhante para focos intensos
                        high_intensity = df_day[df_day['frp'] > df_day['frp'].quantile(0.7)] if 'frp' in df_day.columns else df_day.head(int(len(df_day)*0.3))
                        if len(high_intensity) > 0:
                            white_sizes = 60 + 40 * np.sin(alpha * np.pi * 4)  # Reduzido
                            ax_map.scatter(high_intensity[lon_col], high_intensity[lat_col], 
                                         c='white', s=white_sizes, alpha=0.9 * alpha,
                                         edgecolors='#FFFF00', linewidths=1.5,
                                         transform=ccrs.PlateCarree(), marker='*', zorder=10)
                            
                            # Partículas ascendentes (simulando fagulhas) - REMOVIDO para otimizar
                        
                        # Efeito de pulsação - reduzido
                        if k % 2 == 0:  # A cada 2 frames (ao invés de 3)
                            burst_indices = np.random.choice(len(df_day), size=min(3, len(df_day)), replace=False)  # 3 ao invés de 5
                            burst_points = df_day.iloc[burst_indices]
                            ax_map.scatter(burst_points[lon_col], burst_points[lat_col],
                                         c='#FF0000', s=500, alpha=0.2,  # Tamanho reduzido
                                         transform=ccrs.PlateCarree())
                    
                    bar_heights = [fires_per_day.loc[fires_per_day['acq_date']==d,'n_fires'].values[0] if d<=day else 0 for d in all_days]
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
                    fig.savefig(png_file, facecolor='#000000', dpi=100, bbox_inches='tight', pad_inches=0.1)  # DPI 100 + padding
                    plt.close(fig)
                    img = Image.open(png_file).convert("RGB")
                    # Redimensionar mantendo aspect ratio e preenchendo o frame
                    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    img.save(png_file, quality=85, optimize=True)
                    images_files.append(png_file)
            
            status_text.text("🎬 Assembling video...")
            progress_bar.progress(90)
            
            intro_duration = 4.0
            fires_duration = total_duration_sec
            intro_frame_duration = intro_duration / intro_frames
            fires_frame_count = len(images_files) - intro_frames
            fires_frame_duration = fires_duration / fires_frame_count if fires_frame_count > 0 else 0.1
            frame_durations = [intro_frame_duration] * intro_frames + [fires_frame_duration] * fires_frame_count
            
            clip = ImageSequenceClip(images_files, durations=frame_durations)
            clip = clip.on_color(size=(1280, 720), color=(0,0,0))  # Resolução ajustada
            audio_clip = AudioFileClip("fires_sound.mp3")
            
            def make_frame(t):
                return [0, 0]
            
            silent_audio = AudioClip(make_frame, duration=intro_duration, fps=44100)
            full_audio = concatenate_audioclips([silent_audio, audio_clip])
            clip = clip.set_audio(full_audio)
            clip.fps = 24
            
            status_text.text("💾 Exporting final video...")
            progress_bar.progress(95)
            clip.write_videofile("fires_video.mp4", codec="libx264", audio_codec="aac", verbose=False, logger=None)
            
            # Salvar no cache
            cache_key = st.session_state.get('current_cache_key')
            if cache_key:
                status_text.text("💾 Saving to cache...")
                cached_video, cached_audio = save_to_cache(cache_key, "fires_video.mp4", "fires_sound.mp3")
                st.session_state['video_file'] = cached_video
                st.session_state['mp3_file'] = cached_audio
                
                # Salvar stats DEPOIS de definir os arquivos
                if 'stats_data' in st.session_state:
                    try:
                        stats_file = os.path.join(CACHE_DIR, f"stats_{cache_key}.json")
                        stats_to_save = st.session_state['stats_data']
                        
                        # Debug: mostrar o que vai salvar
                        status_text.text(f"💾 Saving stats: {stats_to_save}")
                        
                        # Salvar com sync completo
                        with open(stats_file, 'w') as f:
                            json.dump(stats_to_save, f, indent=2)
                            f.flush()  # Flush do buffer Python
                            os.fsync(f.fileno())  # Flush do buffer do OS
                        
                        # Verificar se salvou corretamente lendo de volta
                        with open(stats_file, 'r') as f:
                            verify = json.load(f)
                            if verify == stats_to_save:
                                status_text.text(f"✅ Stats verified OK!")
                            else:
                                st.warning(f"⚠️ Stats verification failed: {verify}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not save stats: {e}")
                else:
                    st.warning("⚠️ No stats_data in session_state to save")
            else:
                st.session_state['video_file'] = "fires_video.mp4"
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            st.session_state['generate_clicked'] = False
            
            # Garantir que tudo foi salvo antes do rerun
            import time
            time.sleep(0.5)  # Pequeno delay para garantir flush
            
            progress_placeholder.empty()
            status_placeholder.empty()
            st.rerun()
        else:
            progress_placeholder.empty()
            status_placeholder.empty()
            st.error("⚠️ No fires found.")
            st.session_state['generate_clicked'] = False
    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Error: {str(e)}")
        st.session_state['generate_clicked'] = False
