filename = 'Hear the Fire'
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pydub.utils import which
from pydub import AudioSegment
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips, AudioClip
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from datetime import datetime

# Import the audio generator module
from audio_generator import create_audio_generator

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
        :root {
            --fire-red: #ff4444;
            --fire-orange: #ff8c00;
            --fire-yellow: #ffd700;
            --dark-smoke: #1a1a2e;
            --darker-smoke: #0f0f1e;
            --light: #f5f5f5;
        }

        body {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #2d1b1b 100%);
            color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #ff4444 0%, #ff8c00 50%, #ffd700 100%);
            background-attachment: fixed;
            padding: 30px;
            border-radius: 0px;
            margin: -30px -30px 20px -30px;
            box-shadow: 0 8px 40px rgba(255, 68, 68, 0.6), inset 0 0 30px rgba(255, 212, 0, 0.2);
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
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120"><path d="M0,50 Q300,0 600,50 T1200,50 L1200,120 L0,120 Z" fill="rgba(255,68,68,0.1)"/></svg>');
            background-size: cover;
            opacity: 0.3;
        }

        .main-header h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            color: white;
            font-size: 36px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
        }

        .main-header p {
            position: relative;
            z-index: 1;
            margin: 8px 0 0 0;
            color: rgba(0,0,0,0.8);
            font-size: 16px;
            font-weight: 600;
            text-shadow: 1px 1px 4px rgba(255,255,255,0.3);
        }

        .stat-card {
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%);
            padding: 8px 10px;
            border-radius: 8px;
            border-left: 3px solid #ff4444;
            border-top: 1px solid #ff8c00;
            box-shadow: 0 2px 12px rgba(255, 68, 68, 0.25);
            backdrop-filter: blur(10px);
            margin-bottom: 8px;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(255, 68, 68, 0.35);
        }

        .metric-label {
            font-size: 10px;
            color: #ff8c00;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 18px;
            color: #ffd700;
            font-weight: 700;
            margin: 4px 0;
            text-shadow: 0 0 8px rgba(255, 212, 0, 0.4);
        }

        .success-box {
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%);
            border-left: 5px solid #ff4444;
            border-top: 2px solid #ff8c00;
            padding: 12px;
            border-radius: 8px;
            margin: 12px 0;
            box-shadow: 0 4px 15px rgba(255, 68, 68, 0.2);
        }

        .success-box strong {
            color: #ffd700;
        }

        .video-container {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(255, 68, 68, 0.4);
            border: 2px solid rgba(255, 140, 0, 0.5);
        }

        .stButton>button {
            background: linear-gradient(135deg, #ff4444 0%, #ff8c00 100%) !important;
            color: white !important;
            border: none !important;
            padding: 8px 16px !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            box-shadow: 0 3px 12px rgba(255, 68, 68, 0.4) !important;
            transition: all 0.3s ease !important;
        }

        .stButton>button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(255, 68, 68, 0.5) !important;
        }

        .sidebar-section {
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.1) 0%, rgba(255, 140, 0, 0.05) 100%);
            padding: 12px;
            border-radius: 10px;
            border-left: 4px solid #ff4444;
            margin-bottom: 12px;
        }
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

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

# -------------------
# Header
# -------------------
st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; color: white; font-size: 36px;">🔥 Hear the Fire</h1>
        <p style="margin: 8px 0 0 0; color: rgba(0,0,0,0.8); font-size: 16px;">
            Transform fire data into an immersive audiovisual experience
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="success-box" style="margin: 15px 0; padding: 10px;">
        <p style="color: #f5f5f5; line-height: 1.5; margin: 0; font-size: 14px;">
            🎵 <strong style="color: #ffd700;">How it works:</strong> Each day generates a unique chord. 
            More fires = louder sounds. Pentatonic scale creates natural melodic flow. <strong style="color: #ffd700;">Hear the fires burning.</strong> 🔥
        </p>
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
os.makedirs("maps_png", exist_ok=True)

# -------------------
# Main Content
# -------------------
col_left, col_right = st.columns([1, 2], gap="medium")

with col_left:
    st.markdown("### 📊 Parameter Summary")

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">📍 Location</div>
            <div class="metric-value" style="font-size: 18px;">{latitude_center:.2f}°, {longitude_center:.2f}°</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">🌍 Search Radius</div>
            <div class="metric-value">{radius_km} km</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">📅 Period</div>
            <div class="metric-value">{day_range} days</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">🎵 Duration</div>
            <div class="metric-value">{total_duration_sec} sec</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔥 GENERATE VIDEO + MUSIC", use_container_width=True, key="generate_btn"):
        st.session_state['generate_clicked'] = True

with col_right:
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        with st.spinner("⏳ Processing data..."):
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
                                # === AUDIO GENERATION USING MODULE ===
                                status.update(label="🎵 Creating soundtrack...", state="running")
                                
                                audio_gen = create_audio_generator()
                                melody = audio_gen.generate_melody(fires_per_day, total_duration_sec)
                                file_name = audio_gen.export_audio(melody)
                                st.session_state['mp3_file'] = file_name

                                # === VIDEO GENERATION (unchanged) ===
                                status.update(label="🗺️ Generating maps...", state="running")

                                lon_min = longitude_center - radius_km/100
                                lon_max = longitude_center + radius_km/100
                                lat_min = latitude_center - radius_km/100
                                lat_max = latitude_center + radius_km/100

                                intro_frames = 30
                                images_files = []
                                all_days = fires_per_day['acq_date'].tolist()
                                
                                # Introduction frames
                                for i in range(intro_frames):
                                    progress = (i + 1) / intro_frames
                                    fig = plt.figure(figsize=(20, 15), dpi=200)
                                    fig.patch.set_facecolor('black')
                                    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
                                    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
                                    ax_bar = fig.add_subplot(gs[1])
                                
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
                                        ax_map.text(mid_lon, mid_lat, f'{radius_km} km', color='white', fontsize=16, fontweight='bold',
                                                    transform=ccrs.PlateCarree(), ha='center', va='center',
                                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
                                
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
                                
                                # Fire frames
                                n_fade_frames = 10
                                
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

# -------------------
# Display Video and Downloads
# -------------------
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
