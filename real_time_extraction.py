import librosa
import numpy as np
from google.colab import files
from IPython.display import Audio
import time

def process_audio_realtime(file_path, chunk_size=2048, delay=0.1):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Get the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Extract beats using librosa
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Convert beat frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Extract melody/pitch information
    harmonic_y = librosa.effects.harmonic(y)
    pitches, magnitudes = librosa.core.magphase(librosa.core.cqt(harmonic_y, sr=sr))
    pitch_max = pitches[:, magnitudes.argmax(axis=0)]
    pitch_times = librosa.frames_to_time(range(len(pitch_max)), sr=sr)

    # Extract amplitude/intensity
    amplitude = np.abs(librosa.stft(y))
    amplitude_times = librosa.frames_to_time(range(amplitude.shape[1]), sr=sr)

    # Extract frequency spectrum
    spec = np.abs(librosa.stft(y))
    spec_times = librosa.frames_to_time(range(spec.shape[1]), sr=sr)

    # Determine the number of chunks
    num_chunks = int(len(y) / chunk_size)

    for i in range(num_chunks):
        # Extract a chunk of audio
        chunk = y[i * chunk_size : (i + 1) * chunk_size]

        # Process the chunk in real-time
        print(f"\nProcessing Chunk {i + 1}/{num_chunks} (Delay: {delay} seconds)")
        
        time.sleep(delay)
      
        # Calculate amplitude mean for the current chunk
        amplitude_chunk = np.abs(librosa.stft(chunk))
        amplitude_mean_chunk = np.mean(amplitude_chunk, axis=0)

        # Print or use the real-time results
        print("Real-Time Results:")
        print("Beat Times (in seconds):", beat_times)
        print("Melody Pitch (in Hz):", pitch_max)
        print("Amplitude Mean:", amplitude_mean_chunk)
        print("Frequency Spectrum:")
        print(spec)

        # Play the audio chunk in real-time
        Audio(chunk, rate=sr, autoplay=True)

uploaded = files.upload()
file_path = list(uploaded.keys())[0]
process_audio_realtime(file_path, delay=0.1)
