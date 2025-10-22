frp'].quantile(0.7)] if 'frp' in df_day.columns else df_day.head(int(len(df_day)*0.3))
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
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            st.session_state['video_file'] = "fires_video.mp4"
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
