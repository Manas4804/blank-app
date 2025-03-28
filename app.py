import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("protein_classifier.h5")

model = load_model()

# Load Label Binarizer
@st.cache_resource
def load_label_binarizer():
    import pickle
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

label_binarizer = load_label_binarizer()

# Function to preprocess user input
def preprocess_sequence(sequence):
    sequence = sequence.upper()  # Convert to uppercase
    max_length = 100  # Adjust based on training
    padded_seq = sequence.ljust(max_length, "X")[:max_length]  # Pad or truncate
    encoded_seq = np.array([ord(c) for c in padded_seq])  # Convert to ASCII
    return encoded_seq.reshape(1, max_length, 1)

# Streamlit UI
st.title("Protein Classification App")
st.write("Enter a protein sequence to predict its classification.")

sequence = st.text_input("Protein Sequence", "")

if st.button("Predict"):
    if sequence:
        input_data = preprocess_sequence(sequence)
        prediction = model.predict(input_data)
        predicted_class = label_binarizer.inverse_transform(prediction)[0]

        st.success(f"Predicted Classification: {predicted_class}")
    else:
        st.warning("Please enter a protein sequence.")
