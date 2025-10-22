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
        
        .cache-badge { display: inline-block; background: #00ff88; color: #000; padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 11px; font-weight: 700; margin-left: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# ============= SISTEMA DE CACHE =============
CACHE_DIR = "video_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_cache_key(params):
    """Gera um hash único baseado nos parâmetros"""
    # Serializa os parâmetros de forma consistente
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()

def get_cached_video(cache_key):
    """Verifica se existe vídeo em cache e retorna o caminho"""
    video_path = os.path.join(CACHE_DIR, f"video_{cache_key}.mp4")
    audio_path = os.path.join(CACHE_DIR, f"audio_{cache_key}.mp3")
    
    if os.path.exists(video_path) and os.path.exists(audio_path):
        return video_path, audio_path
    return None, None

def save_to_cache(cache_key, video_path, audio_path):
    """Salva o vídeo e áudio no cache"""
    cached_video = os.path.join(CACHE_DIR, f"video_{cache_key}.mp4")
    cached_audio = os.path.join(CACHE_DIR, f"audio_{cache_key}.mp3")
    
    # Copia os arquivos para o cache
    import shutil
    shutil.copy2(video_path, cached_video)
    shutil.copy2(audio_path, cached_audio)
    
    return cached_video, cached_audio

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

def create_fire_soundscape(fire_data, base_duration_sec=30.0):
    n = len(fire_data)
    if n == 0:
        return AudioSegment.silent(duration=int(base_duration_sec * 1000))
    
    total_frp = fire_data['frp'].sum() if 'frp' in fire_data.columns else n * 10.0
    avg_frp = total_frp / n
    max_frp = fire_data['frp'].max() if 'frp' in fire_data.columns else 100.0
    intensity = min(avg_frp / 50.0, 2.0)
    duration_scale = min(1.0 + (intensity - 1.0) * 0.5, 2.0)
    total_duration_sec = base_duration_sec * duration_scale
    total_duration_ms = int(total_duration_sec * 1000)
    
    root_freq = 110
    scale_notes = [root_freq, root_freq*9/8, root_freq*5/4, root_freq*4/3, root_freq*3/2, root_freq*5/3, root_freq*15/8, root_freq*2]
    
    layers = []
    ambient = create_ambient_layer(total_duration_ms, intensity=min(intensity * 0.6, 1.0))
    layers.append(ambient)
    
    if intensity > 0.3:
        bass = create_bass_line(root_freq, total_duration_ms, pattern='pulse' if intensity > 0.7 else 'sustained')
        layers.append(bass)
    
    segment_count = max(3, int(n / 20))
    segment_duration_ms = total_duration_ms // segment_count
    
    for i in range(segment_count):
        segment_intensity = intensity * (1.0 + 0.3 * np.sin(2 * np.pi * i / segment_count))
        freq = root_freq * (1.2 ** (i % 5))
        waveform_choice = 'pad' if segment_intensity > 0.7 else 'sine'
        tone = generate_tone(freq, segment_duration_ms, waveform=waveform_choice, amplitude=min(segment_intensity, 1.0))
        tone = tone.fade_in(int(segment_duration_ms * 0.2)).fade_out(int(segment_duration_ms * 0.3))
        layers.append(tone)
    
    if intensity > 0.5:
        rhythm = create_rhythm_layer(total_duration_ms, intensity=min(intensity * 0.8, 1.0), pattern='groove' if intensity > 0.8 else 'ambient')
        layers.append(rhythm)
    
    if intensity > 0.6:
        melody_duration_ms = total_duration_ms // 3
        phrase1 = create_melodic_phrase(root_freq * 2, melody_duration_ms, scale_notes, 'ascending')
        phrase2 = create_melodic_phrase(root_freq * 2, melody_duration_ms, scale_notes, 'arpeggio')
        phrase3 = create_melodic_phrase(root_freq * 2, melody_duration_ms, scale_notes, 'descending')
        melody = phrase1 + phrase2 + phrase3
        melody = melody.apply_gain(-10)
        layers.append(melody)
    
    soundscape = AudioSegment.silent(duration=total_duration_ms)
    for layer in layers:
        soundscape = soundscape.overlay(layer)
    
    soundscape = soundscape.normalize()
    max_gain = -3.0
    current_dBFS = soundscape.dBFS
    gain_adjustment = max_gain - current_dBFS
    soundscape = soundscape.apply_gain(min(gain_adjustment, 6.0))
    
    return soundscape, total_duration_sec

# Header
st.markdown("""
    <div class="main-header">
        <h1>🔥 Hear the Fire - Visualize and Sonify Global Wildfires</h1>
        <p>Transform NASA FIRMS fire data into immersive audiovisual experiences</p>
    </div>
""", unsafe_allow_html=True)

# Inicializa session state
if 'video_file' not in st.session_state:
    st.session_state['video_file'] = None
if 'generate_clicked' not in st.session_state:
    st.session_state['generate_clicked'] = False
if 'is_cached' not in st.session_state:
    st.session_state['is_cached'] = False

col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown("### 🎛️ Configuration")
    
    country = st.text_input("🌍 Country", value="Brazil", help="Country name for fire data")
    
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2024-01-01"))
    with date_col2:
        end_date = st.date_input("📅 End Date", value=pd.to_datetime("2024-01-31"))
    
    st.markdown("---")
    
    # Criar dicionário de parâmetros para o cache
    current_params = {
        'country': country,
        'start_date': str(start_date),
        'end_date': str(end_date)
    }
    
    # Gerar cache key
    cache_key = generate_cache_key(current_params)
    
    # Verificar se existe em cache
    cached_video, cached_audio = get_cached_video(cache_key)
    
    if cached_video:
        st.markdown("""
            <div class="info-box">
                ✨ <strong>Cache Found!</strong><br>
                Video already generated for these parameters. Loading instantly!
            </div>
        """, unsafe_allow_html=True)
        st.session_state['is_cached'] = True
    else:
        st.session_state['is_cached'] = False
    
    if st.button("🎬 Generate Visualization", type="primary"):
        st.session_state['generate_clicked'] = True
        
        # Se está em cache, carrega diretamente
        if cached_video:
            st.session_state['video_file'] = cached_video
            st.rerun()

with col_right:
    if st.session_state['video_file'] and os.path.exists(st.session_state['video_file']):
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        
        # Badge de cache se aplicável
        cache_badge = '<span class="cache-badge">⚡ CACHED</span>' if st.session_state.get('is_cached', False) else ''
        st.markdown(f"### 🎥 Fire Visualization {cache_badge}", unsafe_allow_html=True)
        
        st.video(st.session_state['video_file'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with open(st.session_state['video_file'], 'rb') as f:
                st.download_button(
                    label="📥 Download Video",
                    data=f,
                    file_name=f"fires_{country}_{start_date}_{end_date}.mp4",
                    mime="video/mp4"
                )
        with col2:
            if st.button("🔄 Generate New"):
                st.session_state['video_file'] = None
                st.session_state['generate_clicked'] = False
                st.rerun()
    else:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; color: #888; padding: 3rem;'>
                <div style='font-size: 64px; margin-bottom: 1rem;'>🔥</div>
                <h3 style='color: #ff8c00;'>Ready to Visualize Fires</h3>
                <p>Configure parameters and click "Generate Visualization"</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Processamento
if st.session_state.get('generate_clicked', False) and not st.session_state.get('video_file'):
    try:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with progress_placeholder.container():
            progress_bar = st.progress(0)
        with status_placeholder.container():
            status_text = st.empty()
        
        # Verifica cache novamente antes de processar
        cached_video, cached_audio = get_cached_video(cache_key)
        if cached_video:
            status_text.text("⚡ Loading from cache...")
            st.session_state['video_file'] = cached_video
            st.session_state['is_cached'] = True
            st.session_state['generate_clicked'] = False
            progress_placeholder.empty()
            status_placeholder.empty()
            st.rerun()
        
        status_text.text("🌐 Fetching fire data from NASA FIRMS...")
        progress_bar.progress(5)
        
        url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/f1eb3eb0e0d5f2be5dd87e9f7e3ddd65/VIIRS_SNPP_NRT/{country}/1"
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to fetch data: HTTP {response.status_code}")
            st.session_state['generate_clicked'] = False
            st.stop()
        
        df = pd.read_csv(StringIO(response.text))
        
        status_text.text("🔄 Processing fire data...")
        progress_bar.progress(15)
        
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['acq_date'] >= start_dt) & (df['acq_date'] <= end_dt)].copy()
        
        if len(df) > 0:
            status_text.text("🎵 Generating fire soundscape...")
            progress_bar.progress(25)
            
            # Gera áudio e salva com cache key
            soundscape, total_duration_sec = create_fire_soundscape(df, base_duration_sec=30.0)
            audio_file = f"fires_sound_{cache_key}.mp3"
            soundscape.export(audio_file, format="mp3", bitrate="192k")
            
            status_text.text("🗺️ Creating map visualizations...")
            progress_bar.progress(35)
            
            lon_col = 'longitude'
            lat_col = 'latitude'
            lon_min, lon_max = df[lon_col].min() - 2, df[lon_col].max() + 2
            lat_min, lat_max = df[lat_col].min() - 2, df[lat_col].max() + 2
            
            all_days = pd.date_range(start=df['acq_date'].min(), end=df['acq_date'].max(), freq='D')
            fires_per_day = df.groupby('acq_date').size().reset_index(name='n_fires')
            
            os.makedirs("maps_png", exist_ok=True)
            images_files = []
            
            TARGET_WIDTH = 1280
            TARGET_HEIGHT = 720
            
            intro_frames = 24
            status_text.text("🎨 Creating intro sequence...")
            for k in range(intro_frames):
                alpha = (k + 1) / intro_frames
                fig = plt.figure(figsize=(16, 9), dpi=100)
                fig.patch.set_facecolor('black')
                ax = fig.add_subplot(111)
                ax.set_facecolor('black')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                title_alpha = min(alpha * 1.5, 1.0)
                subtitle_alpha = max(0, (alpha - 0.3) * 1.5)
                ax.text(0.5, 0.58, 'HEAR THE FIRE', ha='center', va='center', fontsize=60, 
                       color='white', fontweight='bold', alpha=title_alpha)
                ax.text(0.5, 0.42, f'{country} • {start_date} to {end_date}', ha='center', va='center',
                       fontsize=24, color='#ff8c00', alpha=subtitle_alpha)
                
                png_file = f"maps_png/intro_{k}.png"
                fig.savefig(png_file, facecolor='#000000', dpi=100, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                img = Image.open(png_file).convert("RGB")
                img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                img.save(png_file, quality=85, optimize=True)
                images_files.append(png_file)
            
            status_text.text("🔥 Rendering fire animations...")
            total_days = len(all_days)
            for i, day in enumerate(all_days):
                progress = 35 + int(50 * (i / total_days))
                progress_bar.progress(progress)
                status_text.text(f"🔥 Rendering day {i+1}/{total_days}...")
                
                df_day = df[df['acq_date'] == day].copy()
                if 'frp' in df_day.columns:
                    frp_norm = (df_day['frp'] - df_day['frp'].min()) / (df_day['frp'].max() - df_day['frp'].min() + 1e-6)
                else:
                    frp_norm = pd.Series([0.5]*len(df_day))
                
                n_fade_frames = 3
                for k in range(n_fade_frames):
                    alpha = (k+1)/n_fade_frames
                    fig = plt.figure(figsize=(16, 9), dpi=100)
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
                    
                    if len(df_day) > 0:
                        glow_sizes = 400 + 100 * np.sin(alpha * np.pi * 2)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#8B0000', s=glow_sizes, alpha=0.15 * alpha,
                                     transform=ccrs.PlateCarree())
                        
                        halo_sizes = 250 + 80 * np.sin(alpha * np.pi * 2)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#FF4500', s=halo_sizes, alpha=0.25 * alpha,
                                     transform=ccrs.PlateCarree())
                        
                        core_sizes = 150 + 60 * np.sin(alpha * np.pi * 2)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#FF8C00', s=core_sizes, alpha=0.6 * alpha,
                                     linewidths=0, transform=ccrs.PlateCarree())
                        
                        center_colors = plt.cm.YlOrRd(frp_norm * 0.7 + 0.3)
                        center_sizes = 80 + 50 * np.sin(alpha * np.pi * 3) * (1 + frp_norm)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c=center_colors, s=center_sizes, alpha=0.85 * alpha,
                                     edgecolors='#FFD700', linewidths=1,
                                     transform=ccrs.PlateCarree())
                        
                        high_intensity = df_day[df_day['frp'] > df_day['frp'].quantile(0.7)] if 'frp' in df_day.columns else df_day.head(int(len(df_day)*0.3))
                        if len(high_intensity) > 0:
                            white_sizes = 60 + 40 * np.sin(alpha * np.pi * 4)
                            ax_map.scatter(high_intensity[lon_col], high_intensity[lat_col], 
                                         c='white', s=white_sizes, alpha=0.9 * alpha,
                                         edgecolors='#FFFF00', linewidths=1.5,
                                         transform=ccrs.PlateCarree(), marker='*', zorder=10)
                        
                        if k % 2 == 0:
                            burst_indices = np.random.choice(len(df_day), size=min(3, len(df_day)), replace=False)
                            burst_points = df_day.iloc[burst_indices]
                            ax_map.scatter(burst_points[lon_col], burst_points[lat_col],
                                         c='#FF0000', s=500, alpha=0.2,
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
                    fig.savefig(png_file, facecolor='#000000', dpi=100, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)
                    img = Image.open(png_file).convert("RGB")
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
            clip = clip.on_color(size=(1280, 720), color=(0,0,0))
            audio_clip = AudioFileClip(audio_file)
            
            def make_frame(t):
                return [0, 0]
            
            silent_audio = AudioClip(make_frame, duration=intro_duration, fps=44100)
            full_audio = concatenate_audioclips([silent_audio, audio_clip])
            clip = clip.set_audio(full_audio)
            clip.fps = 24
            
            status_text.text("💾 Exporting final video...")
            progress_bar.progress(95)
            video_file = f"fires_video_{cache_key}.mp4"
            clip.write_videofile(video_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            
            # Salva no cache
            status_text.text("💾 Saving to cache...")
            cached_video, cached_audio = save_to_cache(cache_key, video_file, audio_file)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            st.session_state['video_file'] = cached_video
            st.session_state['is_cached'] = False  # Era novo, agora está em cache
            st.session_state['generate_clicked'] = False
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
