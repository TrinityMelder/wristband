!pip install pydub
import librosa
import numpy as np
from google.colab import files

def process_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Get the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Extract beats using librosa
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Convert beat frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Print the beat times
    print("Beat Times (in seconds):", beat_times)

    # Extract melody/pitch information
    harmonic_y = librosa.effects.harmonic(y)
    pitches, magnitudes = librosa.core.magphase(librosa.core.cqt(harmonic_y, sr=sr))
    pitch_max = pitches[:, magnitudes.argmax(axis=0)]
    pitch_times = librosa.frames_to_time(range(len(pitch_max)), sr=sr)
    print("Melody Pitch (in Hz):", pitch_max)

    # Extract amplitude/intensity
    amplitude = np.abs(librosa.stft(y))
    amplitude_times = librosa.frames_to_time(range(amplitude.shape[1]), sr=sr)
    amplitude_mean = np.mean(amplitude, axis=0)
    print("Amplitude Mean:", amplitude_mean)

    # Extract frequency spectrum
    spec = np.abs(librosa.stft(y))
    spec_times = librosa.frames_to_time(range(spec.shape[1]), sr=sr)
    print("Frequency Spectrum:")
    print(spec)

# Example usage with an uploaded audio file
uploaded = files.upload()
file_path = list(uploaded.keys())[0]

# Process the audio and extract features
process_audio(file_path)
