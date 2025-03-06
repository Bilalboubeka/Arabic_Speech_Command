import librosa
import sounddevice as sd
import numpy as np
import joblib
import time
from sklearn.pipeline import Pipeline

# Load the trained model
model: Pipeline = joblib.load('arabic_speech_svm_background_noise.joblib')

# Define the list of commands in the order matching the training labels
commands = ['zero', 'yes', 'left', 'ok', 'open', 'start', 'stop', 'down']

# Audio parameters (must match training parameters)
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second recordings
N_MFCC = 13
FRAME_LENGTH = 256
HOP_LENGTH = 128


def noise_cancellation(audio, sample_rate, n_fft=512, hop_length=HOP_LENGTH, noise_factor=1.5):
    """
    Perform a basic noise cancellation using spectral gating.
    The noise profile is estimated from the first 0.2 seconds of the audio.
    Frequencies that fall below a threshold (noise_factor * noise_estimate)
    are suppressed.
    """
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Estimate noise from the first 0.2 seconds of the recording
    noise_frames = int(0.2 * sample_rate / hop_length)
    noise_est = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Create a mask where frequency bins are considered signal if above the noise threshold
    mask = magnitude > noise_factor * noise_est
    cleaned_stft = stft * mask

    # Reconstruct the denoised audio from the masked STFT
    audio_clean = librosa.istft(cleaned_stft, hop_length=hop_length)
    return audio_clean


def extract_features(audio):
    """Preprocess audio and extract features matching training pipeline."""
    if len(audio) < SAMPLE_RATE * DURATION:
        padding = int(SAMPLE_RATE * DURATION - len(audio))
        audio = np.pad(audio, (0, padding), 'constant')
    else:
        audio = audio[:int(SAMPLE_RATE * DURATION)]

    # Apply noise cancellation before feature extraction
    audio = noise_cancellation(audio, SAMPLE_RATE)

    n_mels = 20  # Must match training
    n_fft = 512  # Must match training
    mfccs = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_mels=n_mels, n_fft=n_fft, hop_length=HOP_LENGTH
    )
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    return mfccs.flatten()


def record_and_predict(device_id):
    print("Recording...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   device=device_id,
                   blocking=True)
    audio = np.squeeze(audio)

    features = extract_features(audio)
    proba = model.predict_proba([features])[0]
    predicted_idx = np.argmax(proba)
    confidence = proba[predicted_idx]

    # Map the predicted index to the command word
    command_word = commands[predicted_idx]

    print(f"Predicted: {command_word} (Confidence: {confidence:.2%})")
    return command_word


if __name__ == "__main__":
    print("Available input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']}")

    device_id = int(input("Enter the device ID for your microphone: "))

    while True:
        command = record_and_predict(device_id)
        print(f"Command recorded: {command}")
        if input("Record another command? (y/n): ").lower() != 'y':
            break


