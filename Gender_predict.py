import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib


# Step 1: Define a function to extract features from audio files
def extract_features(audio_file):
    # Extract relevant features from the audio file (e.g., MFCCs)
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Step 2: Load the dataset and corresponding labels
def load_dataset(data_dir):
    audio_files = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
                labels.append(os.path.basename(root))  # Assuming each folder represents a class
    return audio_files, labels

# Step 3: Check if there are at least two classes in the dataset
def check_classes(labels):
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        raise ValueError("Dataset must contain at least two classes")
    else:
        print("Classes found in the dataset:", unique_classes)

# Step 4: Load the dataset
data_dir = "C:\\Age and Gender Detection\\Intrenship task\\Internship task\\Task 1\\audio_dataset"
audio_files, labels = load_dataset(data_dir)

# Step 5: Check if there are at least two classes in the dataset
check_classes(labels)

# Step 6: Extract features from the audio files
X = []
for audio_file in audio_files:
    features = extract_features(audio_file)
    X.append(features)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(labels)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Training
# Train a Support Vector Machine (SVM) classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 9: Model Evaluation
# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "train.pkl")

