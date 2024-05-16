import numpy as np
import librosa

def extract_gender_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCCs
    mfccs_mean = np.mean(mfccs, axis=1)  # Mean of MFCCs
    
    # Extract fundamental frequency (pitch)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))[0]
    f0_mean = np.mean(f0)
    
    # Extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    # Combine all features into a single feature vector
    gender_features = np.concatenate([mfccs_mean, [f0_mean, spectral_centroid_mean]])
    
    return gender_features

# Example usage:
audio_file = "C:\\Age and Gender Detection\\Intrenship task\\Internship task\\Task 1\\sample.wav"
gender_features = extract_gender_features(audio_file)
print("Gender features shape:", gender_features.shape)