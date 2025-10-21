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

# -------------------
# Streamlit Configuration
# -------------------
st.set_page_config(
    page_title=f'{filename}',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS para design com tema de incêndios
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
            padding: 12px;
            border-radius: 10px;
            border-left: 4px solid #ff4444;
            border-top: 2px solid #ff8c00;
            box-shadow: 0 4px 20px rgba(255, 68, 68, 0.3), inset 0 1px 0 rgba(255, 212, 0, 0.2);
            backdrop-filter: blur(10px);
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(255, 68, 68, 0.4), inset 0 1px 0 rgba(255, 212, 0, 0.3);
        }

        .metric-label {
            font-size: 11px;
            color: #ff8c00;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-value {
            font-size: 22px;
            color: #ffd700;
            font-weight: 800;
            margin: 6px 0;
            text-shadow: 0 0 10px rgba(255, 212, 0, 0.5);
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

        .control-section {
            background: linear-gradient(135deg, rgba(255, 68, 68, 0.08) 0%, rgba(255, 140, 0, 0.05) 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 140, 0, 0.3);
            margin-bottom: 15px;
        }

        .control-section h3 {
            color: #ffd700;
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 16px;
        }

        .video-container {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(255, 68, 68, 0.4);
            border: 2px solid rgba(255, 140, 0, 0.5);
        }

        .download-buttons {
            display: flex;
            gap: 10px;
            margin-top: 12px;
        }

        .stButton>button {
            background: linear-gradient(135deg, #ff4444 0%, #ff8c00 100%) !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(255, 68, 68, 0.4) !important;
            transition: all 0.3s ease !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255, 68, 68, 0.6) !important;
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

# -------------------
# Header
# -------------------
st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; color: white; font-size: 36px;">🔥 Hear the Fire</h1>
        <p style="margin: 8px 0 0 0; color: rgba(0,0,0,0.8); font-size: 16px;">
            Transforme dados de incêndios em uma experiência audiovisual imersiva
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------
# Sidebar Configuration
# -------------------
st.sidebar.markdown("### ⚙️ Configurações")
st.sidebar.markdown("---")

# API Key fixa (não mostrada no dashboard)
map_key = "aa8b33fef53700c18bce394211eeb2e7"

st.sidebar.markdown('<div class="sidebar-section"><strong>📍 Localização</strong></div>', unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
with col1:
    latitude_center = st.number_input("Latitude", value=-19.0, step=0.1)
with col2:
    longitude_center = st.number_input("Longitude", value=-59.4, step=0.1)

radius_km = st.sidebar.slider("Raio (km)", min_value=50, max_value=1000, value=300, step=50)

st.sidebar.markdown('<div class="sidebar-section"><strong>📅 Dados</strong></div>', unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
with col1:
    data_date = st.date_input("Data inicial", value=datetime(2019, 8, 14))
    data_date = data_date.strftime("%Y-%m-%d")
with col2:
    day_range = st.number_input("Dias a recuperar", value=3, min_value=1, max_value=30)

st.sidebar.markdown('<div class="sidebar-section"><strong>🎵 Áudio</strong></div>', unsafe_allow_html=True)
total_duration_sec = st.sidebar.slider("Duração total (seg)", min_value=5, max_value=60, value=8, step=1)

st.sidebar.markdown("---")
os.makedirs("maps_png", exist_ok=True)

# -------------------
# Main Content - Split Layout
# -------------------
col_left, col_right = st.columns([1, 1], gap="medium")

# LEFT SIDE - Controls & Stats
with col_left:
    st.markdown("### 📊 Resumo de Parâmetros")

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">📍 Localização</div>
            <div class="metric-value" style="font-size: 18px;">{latitude_center:.2f}°, {longitude_center:.2f}°</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">🌍 Raio de Busca</div>
            <div class="metric-value">{radius_km} km</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">📅 Período</div>
            <div class="metric-value">{day_range} dias</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">🎵 Duração</div>
            <div class="metric-value">{total_duration_sec} seg</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔥 GERAR VÍDEO + MÚSICA", use_container_width=True, key="generate_btn"):
        st.session_state['generate_clicked'] = True

# RIGHT SIDE - Video & Results
with col_right:
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        with st.spinner("⏳ Processando dados... Esta operação pode levar alguns minutos"):
            if not map_key:
                st.error("❌ Por favor, insira sua chave de API FIRMS!")
            else:
                try:
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}"
                    response = requests.get(url, timeout=30)

                    if response.status_code != 200:
                        st.error(f"❌ Erro ao buscar dados: {response.status_code}")
                    else:
                        df = pd.read_csv(StringIO(response.text))

                        # Normalizar nomes de coluna
                        df.columns = df.columns.str.strip().str.lower()

                        # Verificar quais colunas de latitude/longitude existem
                        lat_col = None
                        lon_col = None

                        for col in df.columns:
                            if 'lat' in col:
                                lat_col = col
                            if 'lon' in col:
                                lon_col = col

                        if lat_col is None or lon_col is None:
                            st.error(f"❌ Colunas não encontradas. Colunas disponíveis: {list(df.columns)}")
                            st.stop()

                        df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
                        df_local = df[df['dist_km'] <= radius_km].copy()

                        if df_local.empty:
                            st.warning("⚠️ Nenhum incêndio encontrado nesta área e período.")
                        else:
                            # Estatísticas
                            focos_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
                            total_fires = len(df_local)
                            avg_fires_per_day = df_local.groupby('acq_date').size().mean()
                            max_fires_day = focos_per_day['n_fires'].max()

                            # Display Stats na coluna esquerda - movido para cima
                            with col_left:
                                st.markdown("### 📈 Análise de Dados")

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">🔥 Total de Focos</div>
                                        <div class="metric-value">{total_fires}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">📊 Dias com Dados</div>
                                        <div class="metric-value">{len(focos_per_day)}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">📈 Média/Dia</div>
                                        <div class="metric-value">{avg_fires_per_day:.1f}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="stat-card">
                                        <div class="metric-label">⚡ Pico</div>
                                        <div class="metric-value">{max_fires_day}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            # Processamento do vídeo
                            with st.status("🎬 Gerando artefatos...") as status:
                                status.update(label="🎵 Criando trilha sonora...", state="running")

                                all_days = focos_per_day['acq_date'].tolist()
                                n_days = len(focos_per_day)
                                duration_per_day_ms = int((total_duration_sec * 1000) / n_days)
                                pause_ms = 50
                                chord_ms = duration_per_day_ms - pause_ms
                                n_fade_frames = 10

                                # Música
                                notes_penta = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00,
                                               246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
                                melody_segments = []
                                max_fires = focos_per_day['n_fires'].max()
                                min_fires = focos_per_day['n_fires'].min()
                                last_note_idx = np.random.randint(1, len(notes_penta)-4)

                                for day, n_fires in focos_per_day.values:
                                    amplitude = np.interp(n_fires, [min_fires, max_fires], [0.3, 0.7])
                                    shift = np.random.randint(-3, 4)
                                    note_idx = np.clip(last_note_idx + shift, 1, len(notes_penta)-4)
                                    last_note_idx = note_idx
                                    f_base = notes_penta[note_idx]
                                    intervals = [1, 1.25, 1.5, 2]
                                    frequencies = [f_base*x for x in intervals]
                                    chord = epic_chord(frequencies, chord_ms, amplitude)
                                    melody_segments.append(chord)
                                    melody_segments.append(AudioSegment.silent(duration=pause_ms))

                                melody = sum(melody_segments)
                                melody = melody.overlay(humming(130.81, len(melody)))
                                file_name = "fires_epic_sound.mp3"
                                melody.export(file_name, format="mp3", bitrate="192k")
                                st.session_state['mp3_file'] = file_name

                                status.update(label="🗺️ Gerando mapas...", state="running")

                                # Mapa
                                lon_min = longitude_center - radius_km/100
                                lon_max = longitude_center + radius_km/100
                                lat_min = latitude_center - radius_km/100
                                lat_max = latitude_center + radius_km/100
                                images_files = []

                                # INTRODUÇÃO: Círculo crescendo e linha com raio
                               # INTRODUÇÃO: Círculo crescendo e linha com raio
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

                                    # Ponto central
                                    ax_map.plot(longitude_center, latitude_center, 'ro', markersize=15,
                                              transform=ccrs.PlateCarree(), alpha=0.8)

                                    # Círculo crescendo
                                    current_radius_km = radius_km * progress
                                    lat_deg_radius = current_radius_km / 111
                                    lon_deg_radius = current_radius_km / (111 * np.cos(np.radians(latitude_center)))

                                    theta = np.linspace(0, 2*np.pi, 100)
                                    lat_circle = latitude_center + lat_deg_radius * np.sin(theta)
                                    lon_circle = longitude_center + lon_deg_radius * np.cos(theta)

                                    ax_map.plot(lon_circle, lat_circle, 'r-', linewidth=2,
                                              transform=ccrs.PlateCarree(), alpha=0.7)

                                    # Linha do centro até a borda (apenas nos últimos frames)
                                    if progress > 0.7:
                                        # Linha em ângulo de 45 graus
                                        lat_end = latitude_center + lat_deg_radius * np.sin(np.pi/4)
                                        lon_end = longitude_center + lon_deg_radius * np.cos(np.pi/4)

                                        ax_map.plot([longitude_center, lon_end], [latitude_center, lat_end],
                                                  'y-', linewidth=3, transform=ccrs.PlateCarree(), alpha=0.8)

                                        # Texto do raio
                                        mid_lat = (latitude_center + lat_end) / 2
                                        mid_lon = (longitude_center + lon_end) / 2
                                        ax_map.text(mid_lon, mid_lat, f'{radius_km} km',
                                                  color='white', fontsize=16, fontweight='bold',
                                                  transform=ccrs.PlateCarree(), ha='center', va='center',
                                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))

                                    # Subplot inferior VAZIO - mantém o mesmo layout do gráfico principal
                                    ax_bar.set_facecolor('black')
                                    ax_bar.set_xlim(0, 1)
                                    ax_bar.set_ylim(0, 1)
                                    ax_bar.set_xticks([])
                                    ax_bar.set_yticks([])

                                    # Remover bordas do subplot inferior
                                    for spine in ax_bar.spines.values():
                                        spine.set_visible(False)

                                    # Remover bordas do subplot do mapa
                                    for spine in ax_map.spines.values():
                                        spine.set_visible(False)

                                    ax_map.tick_params(left=False, right=False, top=False, bottom=False)

                                    png_file = f"maps_png/intro_{i}.png"
                                    fig.savefig(png_file, facecolor='#000000', bbox_inches='tight', pad_inches=0)
                                    plt.close(fig)

                                    img = Image.open(png_file)
                                    img = img.resize((1920, 1080))
                                    img.save(png_file)
                                    images_files.append(png_file)

                                # SEQUÊNCIA PRINCIPAL: Focos de incêndio
                                for i in range(len(focos_per_day)):
                                    df_day = df_local[df_local['acq_date'] == focos_per_day['acq_date'].iloc[i]]
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

                                        # Focos com efeitos expressivos
                                        scatter = ax_map.scatter(
                                            df_day[lon_col],
                                            df_day[lat_col],
                                            c=frp_norm,
                                            cmap='hot',
                                            s=200 + 100 * np.sin(alpha * np.pi),  # Tamanho pulsante
                                            alpha=0.7 + 0.3 * alpha,
                                            linewidths=2,
                                            edgecolors='yellow',
                                            transform=ccrs.PlateCarree(),
                                            marker='o'
                                        )

                                        # Adicionar alguns efeitos de brilho para focos mais intensos
                                        if len(df_day) > 0:
                                            high_intensity = df_day[df_day['frp'] > df_day['frp'].quantile(0.7)] if 'frp' in df_day.columns else df_day
                                            if len(high_intensity) > 0:
                                                ax_map.scatter(
                                                    high_intensity[lon_col],
                                                    high_intensity[lat_col],
                                                    c='white',
                                                    s=300,
                                                    alpha=0.3 * alpha,
                                                    linewidths=1,
                                                    edgecolors='orange',
                                                    transform=ccrs.PlateCarree(),
                                                    marker='*'
                                                )

                                        # Barra de progresso temporal
                                        bar_heights = [
                                            focos_per_day.loc[focos_per_day['acq_date']==d,'n_fires'].values[0]
                                            if d<=focos_per_day['acq_date'].iloc[i] else 0
                                            for d in all_days
                                        ]
                                        colors = ['orangered' if d<=focos_per_day['acq_date'].iloc[i] else 'gray'
                                                 for d in all_days]
                                        ax_bar.bar(all_days, bar_heights, color=colors, alpha=0.8)
                                        ax_bar.tick_params(colors='white', labelsize=14)
                                        ax_bar.set_ylabel('Número de Incêndios', color='white', fontsize=16)
                                        ax_bar.set_xlabel('Data', color='white', fontsize=16)
                                        ax_bar.set_ylim(0, focos_per_day['n_fires'].max()*1.2)
                                        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')
                                        for spine in ax_bar.spines.values():
                                            spine.set_visible(False)
                                        for spine in ax_map.spines.values():
                                            spine.set_visible(False)
                                        ax_map.tick_params(left=False, right=False, top=False, bottom=False)

                                        png_file = f"maps_png/map_{i}_{k}.png"
                                        fig.savefig(png_file, facecolor='#000000', bbox_inches='tight', pad_inches=0)
                                        plt.close(fig)

                                        img = Image.open(png_file)
                                        img = img.resize((1920, 1080))
                                        img.save(png_file)
                                        images_files.append(png_file)

                                status.update(label="🎬 Compilando vídeo...", state="running")

                                # Ajustar durações - introdução mais rápida, focos no tempo da música
                                total_frames = len(images_files)
                                intro_duration = 4.0  # 2 segundos para introdução
                                fires_duration = total_duration_sec - intro_duration

                                intro_frame_duration = intro_duration / intro_frames
                                fires_frame_duration = fires_duration / (total_frames - intro_frames)

                                frame_durations = [intro_frame_duration] * intro_frames + [fires_frame_duration] * (total_frames - intro_frames)

                                clip = ImageSequenceClip(images_files, durations=frame_durations)
                                clip = clip.on_color(size=(1920,1080), color=(0,0,0))
                                audio_clip = AudioFileClip(file_name)
                                clip = clip.set_audio(audio_clip)
                                clip.fps = 24

                                mp4_file = "fires_cinematic.mp4"
                                clip.write_videofile(mp4_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                                st.session_state['video_file'] = mp4_file

                                status.update(label="✅ Concluído!", state="complete")

                            st.markdown("""
                                <div class="success-box">
                                    <strong>✨ Sucesso!</strong> Sua experiência audiovisual foi gerada com sucesso!
                                </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

# -------------------
# Display Video and Downloads - Right Column
# -------------------
if 'video_file' in st.session_state:
    with col_right:
        st.markdown("### 🎬 Sua Criação")
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
