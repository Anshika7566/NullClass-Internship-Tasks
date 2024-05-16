import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import librosa

def record_audio():
    # Simulated function to record audio
    return [1, 2, 3]  # Example audio data

def extract_features(audio_data):
    # Simulated function to extract features
    return [4, 5, 6]  # Example features

def preprocess_features(features):
    # Simulated function to preprocess features
    return [7, 8, 9]  # Example preprocessed features

def predict_gender(features_preprocessed):
    # Simulated function to predict gender
    return "Male"  # Example prediction

def upload_audio_and_gender(audio_data, gender):
    # Simulated function to upload audio and gender
    print("Uploading audio and gender:", audio_data, gender)

def record_and_process():
    audio_data = record_audio()
    
    if np.all(np.array(audio_data) == 0):
        messagebox.showwarning("Warning", "Audio is blank. Please speak.")
        return

    audio_length = len(audio_data)  # Simulated audio length check
    if audio_length < 30:
        messagebox.showwarning("Warning", "Audio length is less than 30 seconds. Please record again.")
        return

    features = extract_features(audio_data)
    features_preprocessed = preprocess_features(features)
    gender = predict_gender(features_preprocessed)

    if "hi" in audio_data.lower():
        messagebox.showwarning("Warning", "Recorded audio contains the word 'HI'. Please record again.")
        return

    upload_audio_and_gender(audio_data, gender)
    messagebox.showinfo("Success", "Audio and gender uploaded successfully!")

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        audio_length = librosa.get_duration(filename="C:\\Age and Gender Detection\\Intrenship task\\Internship task\\Task 1\\sample.wav")
        messagebox.showinfo("File Info", f"Audio length: {audio_length} seconds")

# GUI
root = tk.Tk()
root.title("Audio Processing GUI")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

record_button = tk.Button(frame, text="Record and Process", command=record_and_process)
record_button.pack(side=tk.LEFT, padx=10)

browse_button = tk.Button(frame, text="Browse File", command=browse_file)
browse_button.pack(side=tk.LEFT, padx=10)

root.mainloop()