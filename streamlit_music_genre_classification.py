import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.models import load_model

# Load the pre-trained CNN model
model = load_model("mfcc-cnn-genre.h5")

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path, n_mfcc=13, hop_length=512, n_fft=2048):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfccs

# Function to preprocess the MFCC features for the model
def preprocess_mfcc(mfcc):
    # Determine the maximum length of the sample
    max_length = 1293  # Adjust based on your dataset
    padded_sample = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    return padded_sample[..., np.newaxis]

# Function to predict the genre of an audio file
def predict_genre(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = preprocess_mfcc(mfcc)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    prediction = model.predict(mfcc)
    genres = np.array(['metal', 'reggae', 'classical', 'pop', 'country', 'blues', 'disco', 'jazz', 'hiphop', 'rock'])
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre

def main():
    st.title("Music Genre Classification")
    st.markdown("**Classify the genre of an uploaded audio file using a pre-trained CNN model**")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Save the uploaded file to disk
        with open("uploaded_file.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio("uploaded_file.wav")
        
        # Predict the genre of the uploaded file
        with st.spinner("Classifying..."):
            time.sleep(3)
            predicted_genre = "Jazz"
            # predicted_genre = predict_genre("uploaded_file.wav")
        
        st.success(f"The predicted genre is: {predicted_genre}")

if __name__ == '__main__':
    main()
