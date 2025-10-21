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

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

st.set_page_config(page_title=f'{filename}', layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .main .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 100% !important; }
        #MainMenu, footer, header { visibility: hidden; }
        body { background: #0a0a14; color: #f5f5f5; overflow: hidden; }
        .main-header { background: linear-gradient(135deg, #ff4444 0%, #ff8c00 50%, #ffd700 100%); padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(255, 68, 68, 0.4); position: relative; overflow: hidden; }
        .main-header h1 { position: relative; z-index: 1; margin: 0; color: white; font-size: 28px; font-weight: 700; }
        .main-header p { position: relative; z-index: 1; margin: 0.3rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 13px; }
        .stat-card { background: linear-gradient(135deg, rgba(255, 68, 68, 0.12) 0%, rgba(255, 140, 0, 0.08) 100%); padding: 0.6rem 0.8rem; border-radius: 10px; border-left: 3px solid #ff4444; margin-bottom: 0.5rem; }
        .metric-label { font-size: 9px; color: #ff8c00; font-weight: 600; text-transform: uppercase; }
        .metric-value { font-size: 16px; color: #ffd700; font-weight: 700; }
        .video-container { background: #000; border-radius: 16px; overflow: visible; box-shadow: 0 12px 40px rgba(255, 68, 68, 0.5); border: 2px solid rgba(255, 140, 0, 0.3); height: calc(100vh - 220px); display: flex; align-items: center; justify-content: center; padding: 0.5rem; }
        .video-container video { width: 100%; height: 100%; object-fit: contain; border-radius: 12px; }
        .stButton>button { background: linear-gradient(135deg, #ff4444 0%, #ff8c00 100%) !important; color: white !important; border: none !important; padding: 0.6rem 1.2rem !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important; }
        .info-box { background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%); border-left: 4px solid #ff4444; padding: 0.8rem; border-radius: 10px; margin: 0.8rem 0; font-size: 12px; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

def generate_tone(frequency, duration_ms, waveform='sine', amplitude=0.5):
    if waveform == 'sine':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    elif waveform == 'pad':
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        tone = tone.overlay(Sine(frequency * 1.01).to_audio_segment(duration=duration_ms) - 8)
        tone = tone.overlay(Sine(frequency * 0.99).to_audio_segment(duration=duration_ms) - 8)
    else:
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
    tone = tone.apply_gain(-50 + amplitude * 25)
    return tone

def create_ambient_layer(duration_ms, intensity=0.3):
    drone1 = Sine(55).to_audio_segment(duration=duration_ms).apply_gain(-45 + intensity * 10)
    drone2 = Sine(82.4).to_audio_segment(duration=duration_ms).apply_gain(-48 + intensity * 10)
    ambient = drone1.overlay(drone2)
    return ambient.fade_in(int(duration_ms * 0.4)).fade_out(int(duration_ms * 0.4))

def create_bass_line(root_freq, duration_ms, pattern='pulse'):
    bass = AudioSegment.silent(duration=duration_ms)
    if pattern == 'pulse':
        for i in range(3):
            pos = int(i * duration_ms / 3)
            note = Sine(root_freq / 2).to_audio_segment(duration=150).apply_gain(-35).fade_in(10).fade_out(100)
            bass = bass.overlay(note, position=pos)
    return bass

def compose_fire_symphony(fires_per_day_df, total_duration_sec=14):
    n_days = len(fires_per_day_df)
    duration_per_day_ms = int((total_duration_sec * 1000) / n_days)
    scale_notes = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99, 880.00]
    max_fires = fires_per_day_df['n_fires'].max()
    min_fires = fires_per_day_df['n_fires'].min()
    ambient_layer = create_ambient_layer(total_duration_sec * 1000, intensity=0.25)
    melody_segments = []
    
    for day_idx, (day, n_fires) in enumerate(fires_per_day_df.values):
        intensity = np.interp(n_fires, [min_fires, max_fires], [0.2, 0.8])
        note_idx = int(np.interp(intensity, [0, 1], [0, len(scale_notes) - 3]))
        base_freq = scale_notes[note_idx]
        intervals = [1, 1.25, 1.5] if intensity < 0.5 else [1, 1.25, 1.5, 2]
        
        chord = AudioSegment.silent(duration=duration_per_day_ms)
        for i, interval in enumerate(intervals):
            freq = base_freq * interval
            note = generate_tone(freq, duration_per_day_ms, 'pad' if intensity > 0.5 else 'sine', 0.3 + intensity * 0.2)
            note = note.fade_in(int(duration_per_day_ms * 0.15)).fade_out(int(duration_per_day_ms * 0.6))
            chord = chord.overlay(note)
        
        melody_segments.append(chord)
    
    melody = sum(melody_segments)
    final_mix = melody.overlay(ambient_layer - 6)
    final_mix = final_mix.fade_in(1000).fade_out(2000).apply_gain(-2).normalize(headroom=0.5)
    return final_mix

def distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

st.markdown('<div class="main-header"><h1>🔥 Hear the Fire</h1><p>Transform fire data into an immersive audiovisual experience</p></div>', unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Settings")
map_key = "aa8b33fef53700c18bce394211eeb2e7"

col1, col2 = st.sidebar.columns(2)
with col1:
    latitude_center = st.number_input("Latitude", value=-19.0, step=0.1)
with col2:
    longitude_center = st.number_input("Longitude", value=-59.4, step=0.1)

radius_km = st.sidebar.slider("Radius (km)", 50, 1000, 150, 50)

col1, col2 = st.sidebar.columns(2)
with col1:
    data_date = st.date_input("Start date", value=datetime(2019, 8, 14)).strftime("%Y-%m-%d")
with col2:
    day_range = st.number_input("Days", value=7, min_value=1, max_value=30)

total_duration_sec = st.sidebar.slider("Duration (sec)", 5, 60, 14, 1)

os.makedirs("maps_png", exist_ok=True)

col_left, col_right = st.columns([1, 3], gap="medium")

with col_left:
    st.markdown('<div class="info-box"><strong>🎵 How it works:</strong> Each day becomes a musical chord. More fires = richer sound. <strong>Listen to the data.</strong></div>', unsafe_allow_html=True)
    
    if st.button("🔥 GENERATE", key="generate_btn"):
        st.session_state['generate_clicked'] = True
    
    if 'video_file' in st.session_state:
        st.markdown("#### 📊 Stats")
        if 'stats_data' in st.session_state:
            stats = st.session_state['stats_data']
            st.markdown(f'<div class="stats-grid"><div class="stat-card"><div class="metric-label">🔥 Total</div><div class="metric-value">{stats["total"]}</div></div><div class="stat-card"><div class="metric-label">📊 Days</div><div class="metric-value">{stats["days"]}</div></div><div class="stat-card"><div class="metric-label">📈 Avg</div><div class="metric-value">{stats["avg"]:.0f}</div></div><div class="stat-card"><div class="metric-label">⚡ Peak</div><div class="metric-value">{stats["peak"]}</div></div></div>', unsafe_allow_html=True)
        
        st.markdown("#### 💾 Download")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if 'mp3_file' in st.session_state and os.path.exists(st.session_state['mp3_file']):
                with open(st.session_state['mp3_file'], "rb") as f:
                    st.download_button("🎵 MP3", f.read(), st.session_state['mp3_file'], "audio/mpeg", use_container_width=True)
        with col_d2:
            if os.path.exists(st.session_state['video_file']):
                with open(st.session_state['video_file'], "rb") as f:
                    st.download_button("🎬 MP4", f.read(), st.session_state['video_file'], "video/mp4", use_container_width=True)

with col_right:
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        st.markdown('<div class="video-container"><div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.8);"><h2 style="color: #ffd700;">⏳ Generating...</h2><p>Please wait.</p></div></div>', unsafe_allow_html=True)
    elif 'video_file' in st.session_state and os.path.exists(st.session_state['video_file']):
        st.markdown("### 🎬 Your Creation")
        st.video(st.session_state['video_file'])
    else:
        st.markdown('<div class="video-container"><div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.5);"><h2 style="color: #ffd700;">🎬 Your Video Will Appear Here</h2><p>Configure parameters and click GENERATE.</p></div></div>', unsafe_allow_html=True)

if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
    try:
        response = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}", timeout=30)
        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.strip().str.lower()
        
        lat_col = next((c for c in df.columns if 'lat' in c), None)
        lon_col = next((c for c in df.columns if 'lon' in c), None)
        
        df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
        df_local = df[df['dist_km'] <= radius_km].copy()
        
        if not df_local.empty:
            fires_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
            st.session_state['stats_data'] = {
                'total': len(df_local),
                'days': len(fires_per_day),
                'avg': fires_per_day['n_fires'].mean(),
                'peak': fires_per_day['n_fires'].max()
            }
            
            # Generate music
            melody = compose_fire_symphony(fires_per_day, total_duration_sec)
            melody.export("fires_sound.mp3", format="mp3", bitrate="192k")
            st.session_state['mp3_file'] = "fires_sound.mp3"
            
            # Generate maps with proper settings
            lon_min = longitude_center - radius_km/100
            lon_max = longitude_center + radius_km/100
            lat_min = latitude_center - radius_km/100
            lat_max = latitude_center + radius_km/100
            images_files = []
            all_days = fires_per_day['acq_date'].tolist()
            n_days = len(fires_per_day)
            
            # Intro frames (30 frames = 4 seconds at slower pace)
            intro_frames = 30
            for i in range(intro_frames):
                progress = (i + 1) / intro_frames
                fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
                fig.patch.set_facecolor('black')
                ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
                ax.set_facecolor('black')
                ax.set_extent([lon_min, lon_max, lat_min, lat_max])
                ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='gray', linewidth=0.8)
                ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
                ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Growing circle
                ax.plot(longitude_center, latitude_center, 'ro', markersize=15, transform=ccrs.PlateCarree(), alpha=0.8)
                current_radius_km = radius_km * progress
                lat_deg_radius = current_radius_km / 111
                lon_deg_radius = current_radius_km / (111 * np.cos(np.radians(latitude_center)))
                theta = np.linspace(0, 2*np.pi, 100)
                lat_circle = latitude_center + lat_deg_radius * np.sin(theta)
                lon_circle = longitude_center + lon_deg_radius * np.cos(theta)
                ax.plot(lon_circle, lat_circle, 'r-', linewidth=2, transform=ccrs.PlateCarree(), alpha=0.7)
                
                png_file = f"maps_png/intro_{i}.png"
                fig.savefig(png_file, facecolor='black', dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                # Force RGB and exact dimensions
                img = Image.open(png_file).convert("RGB")
                final_img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0,0,0))
                img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                offset = ((TARGET_WIDTH - img.width)//2, (TARGET_HEIGHT - img.height)//2)
                final_img.paste(img, offset)
                final_img.save(png_file, quality=95)
                images_files.append(png_file)
            
            # Fire frames - 10 frames per day for smooth animation
            n_frames_per_day = 10
            for idx, (day, n_fires) in enumerate(fires_per_day.values):
                df_day = df_local[df_local['acq_date'] == day]
                
                for frame in range(n_frames_per_day):
                    alpha = (frame + 1) / n_frames_per_day
                    
                    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
                    fig.patch.set_facecolor('black')
                    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
                    ax.set_facecolor('black')
                    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
                    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='gray', linewidth=0.8)
                    ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
                    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Scatter fire points with animation
                    if len(df_day) > 0:
                        ax.scatter(
                            df_day[lon_col], 
                            df_day[lat_col], 
                            c='red', 
                            s=200 + 100 * np.sin(alpha * np.pi), 
                            alpha=0.7 + 0.3 * alpha,
                            linewidths=2,
                            edgecolors='yellow',
                            transform=ccrs.PlateCarree(),
                            marker='o'
                        )
                    
                    png_file = f"maps_png/day_{idx}_frame_{frame}.png"
                    fig.savefig(png_file, facecolor='black', dpi=100, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    
                    # Force RGB and exact dimensions
                    img = Image.open(png_file).convert("RGB")
                    final_img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (0,0,0))
                    img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    offset = ((TARGET_WIDTH - img.width)//2, (TARGET_HEIGHT - img.height)//2)
                    final_img.paste(img, offset)
                    final_img.save(png_file, quality=95)
                    images_files.append(png_file)
            
            # Create video with proper timing
            intro_duration = 4.0
            fires_duration = total_duration_sec
            
            intro_frame_duration = intro_duration / intro_frames
            fires_frame_count = len(images_files) - intro_frames
            fires_frame_duration = fires_duration / fires_frame_count if fires_frame_count > 0 else 0.1
            
            frame_durations = [intro_frame_duration] * intro_frames + [fires_frame_duration] * fires_frame_count
            
            # Create clip
            clip = ImageSequenceClip(images_files, durations=frame_durations)
            clip = clip.on_color(size=(TARGET_WIDTH, TARGET_HEIGHT), color=(0,0,0))
            
            # Add audio (silent intro + music)
            audio_clip = AudioFileClip("fires_sound.mp3")
            
            def make_frame(t):
                return [0, 0]
            
            silent_audio = AudioClip(make_frame, duration=intro_duration, fps=44100)
            full_audio = concatenate_audioclips([silent_audio, audio_clip])
            clip = clip.set_audio(full_audio)
            clip.fps = 24
            
            # Write video
            clip.write_videofile("fires_video.mp4", codec="libx264", audio_codec="aac", bitrate="5000k", fps=24, verbose=False, logger=None)
            
            st.session_state['video_file'] = "fires_video.mp4"
            st.session_state['generate_clicked'] = False
            st.rerun()
        else:
            st.error("⚠️ No fires found in this area.")
            st.session_state['generate_clicked'] = False
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state['generate_clicked'] = False
