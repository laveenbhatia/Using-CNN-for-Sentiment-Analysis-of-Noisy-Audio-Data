import librosa
import os
import numpy as np
from librosa import display
from matplotlib import pyplot as plt


def add_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    noisy_signal = signal + noise * noise_factor
    return noisy_signal


# Directory path for raw audio files
DATA_PATH = "../data/Audio"

# Directory path to save the converted spectrogram
SPEC_PATH = "../data/Spectrogram/Training"

categories = os.listdir(DATA_PATH)

noise_factors = np.array([0.1, 0.3, 0.6, 0.9])

count = 0
print("Extracting Mel Spectrograms")
for category in categories:
    audio_files_path = os.path.join(DATA_PATH, category)
    audio_files = os.listdir(audio_files_path)
    for file in audio_files:
        if file.endswith(".wav"):
            file_path = os.path.join(audio_files_path, file)
            mel_path = SPEC_PATH + "/" + category + "/" + file[0:-3] + "png"
            dir = SPEC_PATH + "/" + category
            os.makedirs(dir, exist_ok=True)
            # Reading the audio file
            signal, sr = librosa.load(file_path, sr=22050)
            # Shuffling the noise factors array so that we pick a different noise factor every time
            np.random.shuffle(noise_factors)
            # Adding the noise
            signal = add_noise(signal, noise_factors[0])
            # Extracting the mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            librosa.display.specshow(log_mel_spectrogram)
            plt.savefig(mel_path)
            print(count)
