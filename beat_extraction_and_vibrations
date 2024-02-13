import librosa
import numpy as np
import sounddevice as sd
import serial
import time
import threading

def play_audio(data, sr):
    """Play audio with sounddevice."""
    sd.play(data, samplerate=sr)
    sd.wait()

def calculate_intensity(energy, min_amp=50, max_amp=255):
    """Scale the energy value to a suitable amplitude range for vibrations."""
    normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    return (normalized_energy * (max_amp - min_amp) + min_amp).astype(int)

def beat_sync_vibrate(serial_conn, beat_times, energies, offset=0.0):
    """Send vibration commands in sync with beats, varying intensity based on audio energy."""
    start_time = time.time()
    amplitudes = calculate_intensity(energies)
    for beat_time, amplitude in zip(beat_times, amplitudes):
        wait_time = beat_time + offset - (time.time() - start_time)
        if wait_time > 0:
            time.sleep(wait_time)
        command = f"V100,{amplitude}\n".encode()
        serial_conn.write(command)

def process_song(serial_conn, filename):
    """Process a single song for playback and vibration synchronization."""
    y, sr = librosa.load(filename, sr=None)
    y_slow = librosa.effects.time_stretch(y, rate=0.8)  # Slow down the song

    tempo, beats = librosa.beat.beat_track(y=y_slow, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    rms_energy = librosa.feature.rms(y=y_slow)[0]
    beat_frames = librosa.frames_to_time(beats, sr=sr)
    energies = rms_energy[librosa.time_to_frames(beat_times, sr=sr)]

    audio_thread = threading.Thread(target=play_audio, args=(y_slow, sr))
    vibration_thread = threading.Thread(target=beat_sync_vibrate, args=(serial_conn, beat_times, energies, -0.1))

    audio_thread.start()
    vibration_thread.start()

    audio_thread.join()
    vibration_thread.join()

# List of songs
song_filenames = [
    r'C:\Users\Kasun_PC\Downloads\song1.mp3',
    r'C:\Users\Kasun_PC\Downloads\song2.mp3',
    r'C:\Users\Kasun_PC\Downloads\song3.mp3',
    r'C:\Users\Kasun_PC\Downloads\song4.mp3',
    r'C:\Users\Kasun_PC\Downloads\song5.mp3'
]

arduino_port = 'COM6'  # Update to your Arduino's COM port
baud_rate = 9600

# Establish serial connection
serial_conn = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Allow time for the serial connection to initialize

# Process each song in sequence
for filename in song_filenames:
    process_song(serial_conn, filename)

serial_conn.close()
