import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load models with error handling
try:
    pneumonia_model = tf.keras.models.load_model('model5_converted.keras')
except Exception as e:
    pneumonia_model = None
    st.error("⚠️ Pneumonia model failed to load.")

try:
    tb_model = tf.keras.models.load_model('tb_cnn_model.h5')
except Exception as e:
    tb_model = None
    st.error("⚠️ TB model failed to load.")

# Custom CSS for styling
st.set_page_config(page_title="X-ray Disease Detector", layout="centered", page_icon="🩻")
st.markdown("""
    <style>
    body {
        background-color: #eaf6f6;
        color: #1f2937;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #eaf6f6;
        padding: 20px;
    }
    .stButton button {
        background: linear-gradient(to right, #36d1dc, #5b86e5);
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 16px;
    }
    .positive {
        background-color: #e74c3c;
        color: white;
        padding: 14px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        margin-top: 15px;
    }
    .note {
        color: #d35400;
        font-size: 16px;
        font-weight: 600;
        margin-top: 10px;
    }
    .negative {
        background-color: #2ecc71;
        color: white;
        padding: 14px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# App Title
st.markdown("<h1 style='text-align: center;'>🩺 PULMOSCAN </h1>", unsafe_allow_html=True)

# Preprocess image
def preprocess_image(img, model_name):
    try:
        if model_name == "Pneumonia":
            img = img.resize((150, 150))
        else:
            img = img.resize((148, 148))
        img = img.convert('L')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        return img_array
    except:
        return None

# Prediction logic
def predict_disease(model, image):
    try:
        prediction = model.predict(image)
        return prediction[0][0]
    except:
        return None

# Upload and Prediction UI
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded X-ray", use_container_width=True)

        model_choice = st.radio("🧬 Choose Disease Model", ["Pneumonia", "Tuberculosis"])

        if st.button("🔍 Predict"):
            processed = preprocess_image(image, model_choice)

            if processed is None:
                st.error("😢 Please upload again.")
            else:
                model = pneumonia_model if model_choice == "Pneumonia" else tb_model
                if model is None:
                    st.error("⚠️ Model failed to load. Try restarting the app.")
                else:
                    confidence = predict_disease(model, processed)
                    if confidence is None:
                        st.error("😢 Prediction failed. Please try another image.")
                    else:
                        if confidence > 0.01:
                            st.markdown(f"<div class='positive'>{model_choice} Result: Positive</div>", unsafe_allow_html=True)
                            st.markdown("<div class='note'>⚠️ Please consult a doctor immediately for a first-degree medical opinion.</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='negative'>{model_choice} Result: Negative</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("😢 An unexpected error occurred. Please upload again.")
