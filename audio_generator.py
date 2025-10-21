"""
Audio Generator Module for Hear the Fire
==========================================
This module handles all music generation logic.
Modify this file to experiment with different sound designs
without changing the main application code.

"""

from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np


class FireAudioGenerator:
    """
    Generates music based on fire data.
    
    Each day with fire activity creates a unique chord.
    The intensity of fires controls the amplitude/volume.
    """
    
    def __init__(self):
        """Initialize the audio generator with default settings."""
        # Pentatonic scale notes (frequencies in Hz)
        self.notes_penta = [
            130.81, 146.83, 164.81, 174.61, 196.00, 220.00,
            246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 
            440.00, 493.88
        ]
        
        # Stereo panning positions for spatial audio
        self.pan_positions = [-0.4, 0.4, -0.2, 0.2, 0.0]
        
        # Base frequency for humming/drone
        self.humming_frequency = 130.81
        
    def humming(self, frequency, duration_ms, amplitude=0.2):
        """
        Create a subtle humming/drone sound for atmospheric depth.
        
        Args:
            frequency (float): Base frequency in Hz
            duration_ms (int): Duration in milliseconds
            amplitude (float): Volume level (0.0 to 1.0)
            
        Returns:
            AudioSegment: The humming sound
        """
        humming_sound = Sine(frequency).to_audio_segment(duration=duration_ms)
        humming_sound = humming_sound.fade_in(int(duration_ms * 0.05))
        humming_sound = humming_sound.fade_out(int(duration_ms * 0.05))
        humming_sound = humming_sound.apply_gain(-30 + amplitude * 20)
        return humming_sound
    
    def epic_chord(self, frequencies, duration_ms, amplitude=0.5):
        """
        Create a rich, epic chord with multiple frequencies.
        
        Args:
            frequencies (list): List of frequencies to combine
            duration_ms (int): Duration in milliseconds
            amplitude (float): Volume level (0.0 to 1.0)
            
        Returns:
            AudioSegment: The chord sound
        """
        chord = AudioSegment.silent(duration=duration_ms)
        note_cache = {}
        
        # Generate base notes
        for f in frequencies:
            note = Sine(f).to_audio_segment(duration=duration_ms)
            note_cache[f] = note
        
        # Combine notes with panning and envelope
        for i, f in enumerate(frequencies):
            note = note_cache[f]
            note = note.fade_in(int(duration_ms * 0.2))
            note = note.fade_out(int(duration_ms * 0.8))
            note = note.apply_gain(-40 + amplitude * 35)
            note = note.pan(self.pan_positions[i % len(self.pan_positions)])
            chord = chord.overlay(note)
        
        # Add reverb/echo effects
        for i in range(2):
            delay = int(duration_ms * 0.5 * (i + 1))
            chord = chord.overlay(chord - (10 + i * 5), position=delay)
        
        return chord
    
    def generate_melody(self, fires_per_day_data, total_duration_sec, pause_ms=50):
        """
        Generate the complete melody based on fire data.
        
        Args:
            fires_per_day_data (pd.DataFrame): DataFrame with 'acq_date' and 'n_fires' columns
            total_duration_sec (int): Total duration in seconds
            pause_ms (int): Pause between chords in milliseconds
            
        Returns:
            AudioSegment: Complete melody with all days
        """
        n_days = len(fires_per_day_data)
        duration_per_day_ms = int((total_duration_sec * 1000) / n_days)
        chord_ms = duration_per_day_ms - pause_ms
        
        melody_segments = []
        max_fires = fires_per_day_data['n_fires'].max()
        min_fires = fires_per_day_data['n_fires'].min()
        
        # Start at a random position in the scale
        last_note_idx = np.random.randint(1, len(self.notes_penta) - 4)
        
        for day, n_fires in fires_per_day_data.values:
            # Map fire intensity to amplitude
            amplitude = np.interp(n_fires, [min_fires, max_fires], [0.3, 0.7])
            
            # Create melodic progression (random walk through scale)
            shift = np.random.randint(-3, 4)
            note_idx = np.clip(last_note_idx + shift, 1, len(self.notes_penta) - 4)
            last_note_idx = note_idx
            
            # Build chord with harmonic intervals
            f_base = self.notes_penta[note_idx]
            intervals = [1, 1.25, 1.5, 2]  # Root, major third, fifth, octave
            frequencies = [f_base * x for x in intervals]
            
            # Generate and add chord
            chord = self.epic_chord(frequencies, chord_ms, amplitude)
            melody_segments.append(chord)
            melody_segments.append(AudioSegment.silent(duration=pause_ms))
        
        # Combine all segments
        melody = sum(melody_segments)
        
        # Add atmospheric humming
        melody = melody.overlay(self.humming(self.humming_frequency, len(melody)))
        
        return melody
    
    def export_audio(self, melody, filename="fires_epic_sound.mp3", bitrate="192k"):
        """
        Export the melody to an audio file.
        
        Args:
            melody (AudioSegment): The audio to export
            filename (str): Output filename
            bitrate (str): Audio bitrate (e.g., "192k", "320k")
            
        Returns:
            str: Path to the exported file
        """
        melody.export(filename, format="mp3", bitrate=bitrate)
        return filename
