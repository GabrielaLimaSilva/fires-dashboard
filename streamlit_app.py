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
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        with st.spinner("⏳ Processing..."):
            if not map_key:
                st.error("❌ Please enter your FIRMS API key!")
            else:
                try:
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}"
                    response = requests.get(url, timeout=30)

                    if response.status_code != 200:
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
                            st.error(f"❌ Columns not found. Available columns: {list(df.columns)}")
                            st.stop()

                        df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
                        df_local = df[df['dist_km'] <= radius_km].copy()

                        if df_local.empty:
                            st.warning("⚠️ No fires found in this area and period.")
                        else:
                            fires_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
                            total_fires = len(df_local)
                            avg_fires_per_day = df_local.groupby('acq_date').size().mean()
                            max_fires_day = fires_per_day['n_fires'].max()

                            # Salvar stats para exibir depois
                            st.session_state['stats_data'] = {
                                'total': total_fires,
                                'days': len(fires_per_day),
                                'avg': avg_fires_per_day,
                                'peak': max_fires_day
                            }

                            with st.status("🎬 Generating...") as status:
                                status.update(label="🎵 Creating soundtrack...", state="running")

                                all_days = fires_per_day['acq_date'].tolist()
                                n_days = len(fires_per_day)
                                n_fade_frames = 10

                                # IMPROVED MUSIC GENERATION
                                melody = compose_fire_symphony(fires_per_day, total_duration_sec)
                                file_name = "fires_epic_sound.mp3"
                                melody.export(file_name, format="mp3", bitrate="192k")
                                st.session_state['mp3_file'] = file_name

                                status.update(label="🗺️ Generating maps...", state="running")

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

                                status.update(label="✅ Complete!", state="complete")

                            # Rerun para atualizar layout com vídeo
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")page_title=f'{filename}',
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

        /* Remove padding padrão do Streamlit */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        /* Ocultar elementos do Streamlit */
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

        /* Compact stat cards */
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
            overflow: hidden;
            box-shadow: 0 12px 40px rgba(255, 68, 68, 0.5);
            border: 2px solid rgba(255, 140, 0, 0.3);
            height: calc(100vh - 180px);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-container video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Botões modernos */
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

        /* Download buttons */
        .download-section {
            margin-top: 0.8rem;
            display: flex;
            gap: 0.5rem;
        }

        /* Sidebar moderna */
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

        /* Inputs modernos */
        .stNumberInput>div>div>input,
        .stSlider>div>div>div>div {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 140, 0, 0.3) !important;
            border-radius: 8px !important;
            color: white !important;
        }

        /* Info box compacta */
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

        /* Compact stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 0.8rem;
        }

        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #ff4444 !important;
        }

        /* Remove espaços extras */
        .element-container {
            margin-bottom: 0 !important;
        }

        /* Status moderna */
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
        # Pad synth style - rico e envolvente
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        tone = tone.overlay(Sine(frequency * 1.01).to_audio_segment(duration=duration_ms) - 8)  # Detuning
        tone = tone.overlay(Sine(frequency * 0.99).to_audio_segment(duration=duration_ms) - 8)
        tone = tone.overlay(Sine(frequency * 2).to_audio_segment(duration=duration_ms) - 18)
    else:
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    
    tone = tone.apply_gain(-50 + amplitude * 25)
    return tone


def create_bass_line(root_freq, duration_ms, pattern='pulse'):
    """Cria uma linha de baixo rítmica."""
    bass = AudioSegment.silent(duration=duration_ms)
    
    if pattern == 'pulse':
        # Pulso rítmico - minimalista
        pulse_duration = min(150, duration_ms // 4)
        for i in range(3):
            pos = int(i * duration_ms / 3)
            note = Sine(root_freq / 2).to_audio_segment(duration=pulse_duration)
            note = note.apply_gain(-35).fade_in(10).fade_out(100)
            bass = bass.overlay(note, position=pos)
    
    elif pattern == 'walking':
        # Linha de baixo andante
        notes = [root_freq / 2, root_freq / 2 * 1.125, root_freq / 2 * 1.25, root_freq / 2 * 1.125]
        note_duration = duration_ms // len(notes)
        for i, freq in enumerate(notes):
            note = Sine(freq).to_audio_segment(duration=note_duration)
            note = note.apply_gain(-38).fade_in(20).fade_out(50)
            bass = bass.overlay(note, position=i * note_duration)
    
    elif pattern == 'sustained':
        # Baixo sustentado
        note = Sine(root_freq / 2).to_audio_segment(duration=duration_ms)
        note = note.apply_gain(-40).fade_in(100).fade_out(200)
        bass = bass.overlay(note)
    
    return bass


def create_rhythm_layer(duration_ms, intensity=0.5, pattern='ambient'):
    """Cria camada rítmica sutil."""
    rhythm = AudioSegment.silent(duration=duration_ms)
    
    if pattern == 'ambient':
        # Pulsos sutis de percussão
        num_hits = int(3 + intensity * 2)
        for i in range(num_hits):
            pos = int(i * duration_ms / num_hits)
            # Tom percussivo agudo
            hit = Sine(800 + i * 100).to_audio_segment(duration=60)
            hit = hit.apply_gain(-45 + intensity * 5).fade_out(50)
            rhythm = rhythm.overlay(hit, position=pos)
    
    elif pattern == 'groove':
        # Padrão rítmico mais presente
        beat_duration = duration_ms // 4
        for i in range(4):
            pos = i * beat_duration
            # Kick sutil
            if i % 2 == 0:
                kick = Sine(60).to_audio_segment(duration=80)
                kick = kick.apply_gain(-40).fade_out(60)
                rhythm = rhythm.overlay(kick, position=pos)
            # Hi-hat
            hat = Sine(3000).to_audio_segment(duration=30)
            hat = hat.apply_gain(-48).fade_out(25)
            rhythm = rhythm.overlay(hat, position=pos)
    
    return rhythm


def create_melodic_phrase(base_freq, duration_ms, scale_notes, phrase_type='ascending'):
    """Cria uma frase melódica."""
    melody = AudioSegment.silent(duration=duration_ms)
    
    if phrase_type == 'ascending':
        # Frase ascendente
        note_indices = [0, 2, 4, 6]
    elif phrase_type == 'descending':
        # Frase descendente
        note_indices = [6, 4, 2, 0]
    elif phrase_type == 'wave':
        # Onda melódica
        note_indices = [0, 3, 6, 3]
    else:
        # Aleatório suave
        note_indices = [0, 2, 3, 5]
    
    note_duration = duration_ms // len(note_indices)
    
    for i, idx in enumerate(note_indices):
        freq = base_freq * (scale_notes[idx % len(scale_notes)] / scale_notes[0])
        note = generate_tone(freq, note_duration, 'sine', 0.4)
        note = note.fade_in(50).fade_out(100)
        melody = melody.overlay(note, position=i * note_duration)
    
    return melody


def create_ambient_layer(duration_ms, intensity=0.3):
    """Cria uma camada de som ambiente que evolui lentamente."""
    drone1 = Sine(55).to_audio_segment(duration=duration_ms)
    drone2 = Sine(82.4).to_audio_segment(duration=duration_ms)
    
    # Reduzir ruído para som mais limpo
    noise = AudioSegment.silent(duration=duration_ms)
    for _ in range(3):  # Reduzido de 5 para 3
        freq = np.random.uniform(150, 250)
        noise_tone = Sine(freq).to_audio_segment(duration=duration_ms)
        noise_tone = noise_tone.apply_gain(-55 + np.random.uniform(-3, 3))
        noise = noise.overlay(noise_tone)
    
    # Volume mais baixo e suave
    ambient = drone1.apply_gain(-45 + intensity * 10)
    ambient = ambient.overlay(drone2.apply_gain(-48 + intensity * 10))
    ambient = ambient.overlay(noise.apply_gain(-50))
    
    # Fade mais longo para suavidade
    ambient = ambient.fade_in(int(duration_ms * 0.4)).fade_out(int(duration_ms * 0.4))
    
    return ambient


def epic_chord_v2(frequencies, duration_ms, fire_intensity=0.5, day_index=0, total_days=1):
    """Versão melhorada dos acordes com mais diferenciação e impacto."""
    chord = AudioSegment.silent(duration=duration_ms)
    
    # Usar timbres mais suaves em geral
    if fire_intensity < 0.3:
        waveforms = ['sine', 'sine', 'sine', 'sine']
        attack_time = 0.4
        release_time = 0.8
    elif fire_intensity < 0.6:
        waveforms = ['sine', 'sine', 'triangle', 'sine']
        attack_time = 0.3
        release_time = 0.7
    else:
        waveforms = ['sine', 'triangle', 'complex', 'sine']  # Removido sawtooth e square
        attack_time = 0.2
        release_time = 0.6
    
    pan_positions = [-0.4, -0.15, 0.15, 0.4, 0.0]
    
    for i, freq in enumerate(frequencies):
        waveform = waveforms[i % len(waveforms)]
        
        # Amplitude mais controlada
        base_amplitude = 0.3 + fire_intensity * 0.3
        if i == 0:
            amplitude = base_amplitude * 1.0
        else:
            amplitude = base_amplitude * (0.6 + i * 0.08)
        
        note = generate_tone(freq, duration_ms, waveform, amplitude)
        
        fade_in_ms = int(duration_ms * attack_time)
        fade_out_ms = int(duration_ms * release_time)
        note = note.fade_in(fade_in_ms).fade_out(fade_out_ms)
        
        pan = pan_positions[i % len(pan_positions)]
        note = note.pan(pan)
        
        chord = chord.overlay(note)
    
    # Efeitos reduzidos
    if fire_intensity > 0.5:  # Aumentado threshold
        num_echoes = 1
        for i in range(num_echoes):
            delay_ms = int(duration_ms * 0.4 * (i + 1))
            decay_db = -(12 + i * 8)  # Mais suave
            chord = chord.overlay(chord + decay_db, position=delay_ms)
    
    # Remover transiente agressivo, deixar apenas em intensidade muito alta
    if fire_intensity > 0.8:
        attack_freq = frequencies[0] * 3
        attack = Sine(attack_freq).to_audio_segment(duration=40)
        attack = attack.apply_gain(-40 + fire_intensity * 8)
        attack = attack.fade_out(35)
        chord = chord.overlay(attack, position=0)
    
    # Sub-bass mais suave
    if fire_intensity > 0.7:
        sub_freq = frequencies[0] / 2
        sub_bass = Sine(sub_freq).to_audio_segment(duration=duration_ms)
        sub_bass = sub_bass.apply_gain(-45 + fire_intensity * 8)
        sub_bass = sub_bass.fade_in(150).fade_out(int(duration_ms * 0.7))
        chord = chord.overlay(sub_bass)
    
    return chord


def create_transition(duration_ms, transition_type='rise', intensity=0.5):
    """Cria efeitos de transição entre dias."""
    if transition_type == 'silence':
        return AudioSegment.silent(duration=duration_ms)
    
    elif transition_type == 'rise':
        start_freq = 130
        end_freq = 350 + intensity * 300
        sweep = AudioSegment.silent(duration=duration_ms)
        
        steps = 15  # Menos steps para som mais suave
        for i in range(steps):
            progress = i / steps
            freq = start_freq + (end_freq - start_freq) * progress
            segment_duration = duration_ms // steps
            tone = Sine(freq).to_audio_segment(duration=segment_duration)
            tone = tone.apply_gain(-45 + intensity * 8)  # Muito mais suave
            sweep = sweep.overlay(tone, position=i * segment_duration)
        
        return sweep.fade_out(int(duration_ms * 0.5))
    
    elif transition_type == 'fall':
        start_freq = 600 - intensity * 200
        end_freq = 130
        sweep = AudioSegment.silent(duration=duration_ms)
        
        steps = 12
        for i in range(steps):
            progress = i / steps
            freq = start_freq - (start_freq - end_freq) * progress
            segment_duration = duration_ms // steps
            tone = Sine(freq).to_audio_segment(duration=segment_duration)
            tone = tone.apply_gain(-45 + intensity * 8)
            sweep = sweep.overlay(tone, position=i * segment_duration)
        
        return sweep.fade_in(int(duration_ms * 0.4))
    
    elif transition_type == 'impact':
        impact = AudioSegment.silent(duration=duration_ms)
        
        # Impact mais sutil
        low = Sine(60).to_audio_segment(duration=duration_ms)
        low = low.apply_gain(-40 + intensity * 10)
        low = low.fade_out(int(duration_ms * 0.9))
        
        mid = Sine(250).to_audio_segment(duration=duration_ms // 2)  # Sine ao invés de Square
        mid = mid.apply_gain(-42 + intensity * 8)
        mid = mid.fade_out(int(duration_ms * 0.5))
        
        high = Sine(800).to_audio_segment(duration=duration_ms // 4)  # Freq mais baixa
        high = high.apply_gain(-45 + intensity * 10)
        high = high.fade_out(int(duration_ms * 0.4))
        
        impact = impact.overlay(low).overlay(mid).overlay(high)
        return impact


def compose_fire_symphony(fires_per_day_df, total_duration_sec=14):
    """Compõe a trilha sonora completa com estrutura musical."""
    n_days = len(fires_per_day_df)
    duration_per_day_ms = int((total_duration_sec * 1000) / n_days)
    
    # Escala pentatônica maior (alegre e melodiosa)
    scale_notes = [
        261.63,  # C4
        293.66,  # D4
        329.63,  # E4
        392.00,  # G4
        440.00,  # A4
        523.25,  # C5
        587.33,  # D5
        659.25,  # E5
        783.99,  # G5
        880.00   # A5
    ]
    
    max_fires = fires_per_day_df['n_fires'].max()
    min_fires = fires_per_day_df['n_fires'].min()
    mean_fires = fires_per_day_df['n_fires'].mean()
    
    # Definir seções musicais baseadas no número de dias
    intro_days = min(2, n_days // 4)
    outro_days = min(2, n_days // 4)
    main_days = n_days - intro_days - outro_days
    
    # Camada ambiente constante
    ambient_layer = create_ambient_layer(
        total_duration_sec * 1000, 
        intensity=0.25
    )
    
    melody_segments = []
    bass_segments = []
    rhythm_segments = []
    
    # Escolher tonalidade baseada na intensidade média
    root_note_idx = 0 if mean_fires < (max_fires * 0.4) else 2
    
    for day_idx, (day, n_fires) in enumerate(fires_per_day_df.values):
        intensity = np.interp(n_fires, [min_fires, max_fires], [0.2, 0.8])
        
        # === DETERMINAR SEÇÃO MUSICAL ===
        if day_idx < intro_days:
            section = 'intro'
        elif day_idx >= n_days - outro_days:
            section = 'outro'
        else:
            section = 'main'
        
        # === ESCOLHER NOTA DA MELODIA ===
        # Usar intensidade para escolher nota na escala
        note_idx = int(np.interp(intensity, [0, 1], [0, len(scale_notes) - 3]))
        note_idx = np.clip(note_idx, 0, len(scale_notes) - 3)
        
        base_freq = scale_notes[note_idx]
        
        # === CONSTRUIR ACORDE ===
        if section == 'intro':
            # Intro: acordes simples e etéreos
            intervals = [1, 1.5, 2]  # Quinta + oitava
            waveform = 'pad'
            chord_amplitude = 0.25 + intensity * 0.15
        elif section == 'outro':
            # Outro: resolução suave
            intervals = [1, 1.25, 1.5]  # Tríade maior
            waveform = 'sine'
            chord_amplitude = 0.3 - (day_idx - (n_days - outro_days)) * 0.05
        else:
            # Main: acordes mais ricos
            if intensity < 0.4:
                intervals = [1, 1.25, 1.5]  # Tríade
            elif intensity < 0.7:
                intervals = [1, 1.25, 1.5, 2]  # Tríade + oitava
            else:
                intervals = [1, 1.2, 1.5, 1.8, 2]  # Acorde mais cheio
            waveform = 'pad' if intensity > 0.5 else 'sine'
            chord_amplitude = 0.3 + intensity * 0.2
        
        # Criar acorde principal
        chord = AudioSegment.silent(duration=duration_per_day_ms)
        pan_positions = [-0.3, 0, 0.3, -0.15, 0.15]
        
        frequencies = [base_freq * x for x in intervals]
        
        for i, freq in enumerate(frequencies):
            note = generate_tone(freq, duration_per_day_ms, waveform, chord_amplitude)
            
            # Envelope musical
            attack = int(duration_per_day_ms * 0.15)
            release = int(duration_per_day_ms * 0.6)
            note = note.fade_in(attack).fade_out(release)
            
            # Panorâmica
            note = note.pan(pan_positions[i % len(pan_positions)])
            chord = chord.overlay(note)
        
        # Adicionar delay sutil em acordes intensos
        if intensity > 0.6 and section == 'main':
            delay_ms = int(duration_per_day_ms * 0.4)
            chord = chord.overlay(chord - 10, position=delay_ms)
        
        melody_segments.append(chord)
        
        # === LINHA DE BAIXO ===
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
        
        # === CAMADA RÍTMICA ===
        if section == 'intro' or section == 'outro':
            rhythm_pattern = 'ambient'
        else:
            rhythm_pattern = 'groove' if intensity > 0.5 else 'ambient'
        
        rhythm = create_rhythm_layer(duration_per_day_ms, intensity, rhythm_pattern)
        rhythm_segments.append(rhythm)
        
        # === FRASES MELÓDICAS (ocasionais) ===
        # Adicionar melodia em momentos específicos
        if day_idx > 0 and day_idx % 3 == 0 and section == 'main':
            prev_intensity = np.interp(
                fires_per_day_df.iloc[day_idx - 1]['n_fires'],
                [min_fires, max_fires],
                [0.2, 0.8]
            )
            
            if intensity > prev_intensity:
                phrase = create_melodic_phrase(
                    base_freq * 2,  # Oitava acima
                    duration_per_day_ms,
                    scale_notes,
                    'ascending'
                )
                phrase = phrase - 12  # Mais suave
                chord = chord.overlay(phrase)
            elif intensity < prev_intensity - 0.2:
                phrase = create_melodic_phrase(
                    base_freq * 2,
                    duration_per_day_ms,
                    scale_notes,
                    'descending'
                )
                phrase = phrase - 12
                chord = chord.overlay(phrase)
    
    # === MIXAGEM FINAL ===
    
    # Combinar todas as camadas
    melody_track = sum(melody_segments)
    bass_track = sum(bass_segments)
    rhythm_track = sum(rhythm_segments)
    
    # Balancear volumes
    final_mix = melody_track  # Base
    final_mix = final_mix.overlay(bass_track - 2)  # Baixo um pouco mais baixo
    final_mix = final_mix.overlay(rhythm_track - 5)  # Ritmo sutil
    final_mix = final_mix.overlay(ambient_layer - 6)  # Ambiente no fundo
    
    # === MASTERIZAÇÃO ===
    
    # Fade musical
    intro_fade = int(total_duration_sec * 1000 * 0.08)  # 8% do total
    outro_fade = int(total_duration_sec * 1000 * 0.15)  # 15% do total
    
    final_mix = final_mix.fade_in(intro_fade).fade_out(outro_fade)
    
    # Compressão suave (simulada)
    final_mix = final_mix.apply_gain(-2)
    final_mix = final_mix.normalize(headroom=0.5)
    
    # Adicionar reverb sutil (via delay longo)
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
# Main Content - Layout Otimizado
# -------------------
col_left, col_right = st.columns([1, 3], gap="medium")

# LEFT SIDE - Controls compactos
with col_left:
    
    # Info box compacta
    st.markdown("""
        <div class="info-box">
            <strong>🎵 How it works:</strong> Each day becomes a musical chord. 
            More fires = richer sound with bass and rhythm. 
            <strong>Listen to the data.</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # Botão de geração
    if st.button("🔥 GENERATE", key="generate_btn"):
        st.session_state['generate_clicked'] = True
    
    # Stats compactas (aparecem depois de gerar)
    if 'video_file' in st.session_state:
        st.markdown("#### 📊 Stats")
        
        if 'stats_data' in st.session_state:
            stats = st.session_state['stats_data']
            
            # Grid 2x2 para stats
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
        
        # Downloads compactos
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

# RIGHT SIDE - Vídeo grande
with col_right:
    if 'video_file' in st.session_state:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(st.session_state['video_file'])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder quando não há vídeo
        st.markdown("""
            <div class="video-container">
                <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.5);">
                    <h2 style="color: #ffd700; margin-bottom: 1rem;">🎬 Your Video Will Appear Here</h2>
                    <p style="font-size: 14px;">Configure the parameters in the sidebar and click "GENERATE" to create your audiovisual experience.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        with st.spinner("⏳ Processing data... This operation may take a few minutes"):
            if not map_key:
                st.error("❌ Please enter your FIRMS API key!")
            else:
                try:
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}"
                    response = requests.get(url, timeout=30)

                    if response.status_code != 200:
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
                            st.error(f"❌ Columns not found. Available columns: {list(df.columns)}")
                            st.stop()

                        df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
                        df_local = df[df['dist_km'] <= radius_km].copy()

                        if df_local.empty:
                            st.warning("⚠️ No fires found in this area and period.")
                        else:
                            fires_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
                            total_fires = len(df_local)
                            avg_fires_per_day = df_local.groupby('acq_date').size().mean()
                            max_fires_day = fires_per_day['n_fires'].max()

                            with col_left:
                                st.markdown("### 📈 Data Analysis")

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">🔥 Total Fire Spots</div>
                                        <div class="metric-value">{total_fires}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">📊 Days with Data</div>
                                        <div class="metric-value">{len(fires_per_day)}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">📈 Average/Day</div>
                                        <div class="metric-value">{avg_fires_per_day:.1f}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">⚡ Peak</div>
                                        <div class="metric-value">{max_fires_day}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            with st.status("🎬 Generating artifacts...") as status:
                                status.update(label="🎵 Creating soundtrack...", state="running")

                                all_days = fires_per_day['acq_date'].tolist()
                                n_days = len(fires_per_day)
                                n_fade_frames = 10

                                # IMPROVED MUSIC GENERATION
                                melody = compose_fire_symphony(fires_per_day, total_duration_sec)
                                file_name = "fires_epic_sound.mp3"
                                melody.export(file_name, format="mp3", bitrate="192k")
                                st.session_state['mp3_file'] = file_name

                                status.update(label="🗺️ Generating maps...", state="running")

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

                                status.update(label="✅ Complete!", state="complete")

                            st.markdown("""
                                <div class="success-box">
                                    <strong>✨ Success!</strong> Your audiovisual experience has been generated successfully!
                                </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

if 'video_file' in st.session_state:
    with col_right:
        st.markdown("### 🎬 Your Creation")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(st.session_state['video_file'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="download-buttons">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
