import streamlit as st
import numpy as np
import cv2
import pickle
import tensorflow as tf
from PIL import Image
import pandas as pd
import os

st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('traffic_sign_model.h5')
        return model
    except:
        st.error("Model file not found. Please run the training notebook first.")
        return None

@st.cache_data
def load_class_names():
    class_names = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    
   
    return class_names

def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def predict_traffic_sign(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_id, confidence

st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        animation: float 3s ease-in-out infinite;
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .sub-title {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .elevvo-text {
        color: #FFD700;
        font-size: 1.5rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .floating-credit {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: bounce 2s infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes glow {
        from { text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 10px rgba(255,255,255,0.3); }
        to { text-shadow: 2px 2px 4px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.6); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-banner">
    <div class="main-title">🚦 Traffic Sign Recognition System</div>
    <div class="sub-title">Advanced AI-powered traffic sign classification using deep learning</div>
    <div class="elevvo-text">Elevvo Pathways</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="floating-credit">
    Created by Aatiqa Sadiq
</div>
""", unsafe_allow_html=True)

model = load_model()
class_names = load_class_names()

if model is not None:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("Upload Traffic Sign Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a traffic sign image for classification"
    )
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Check if it's a valid image (should have 3 dimensions for RGB)
        if len(image_array.shape) >= 2:
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Traffic Sign", use_container_width=True)
            
            with col2:
                st.subheader("Prediction")
                
                with st.spinner("Analyzing traffic sign..."):
                    class_id, confidence = predict_traffic_sign(model, image_array)
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: white; margin-bottom: 10px;">🚦 Traffic Sign Detected</h2>
                    <h3 style="color: #FFD700; margin-bottom: 15px;">{class_name}</h3>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <p style="margin: 5px 0;"><strong>Confidence:</strong> {confidence:.2%}</p>
                            <p style="margin: 5px 0;"><strong>Class ID:</strong> {class_id}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                st.markdown(f"""
                <div style="background-color: {confidence_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                    Confidence Level: {'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Please upload a valid image file (supported formats: PNG, JPG, JPEG)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("About the System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Model Architecture**
        - Convolutional Neural Network (CNN)
        - Multiple Conv2D layers with BatchNormalization
        - Dropout layers for regularization
        - Dense layers for classification
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Features**
        - 43 traffic sign classes
        - Image preprocessing and normalization
        - Data augmentation for better generalization
        - Real-time prediction with confidence scores
        """)
    
    with col3:
        st.markdown("""
        **🎯 Performance**
        - Trained on GTSRB dataset
        - High accuracy classification
        - Robust to various lighting conditions
        - Fast inference time
        """)
    
    if st.checkbox("Show Class Information"):
        st.subheader("Traffic Sign Classes")
        
        if os.path.exists('Data/Meta.csv'):
            meta_df = pd.read_csv('Data/Meta.csv')
            st.dataframe(meta_df)
        else:
            class_df = pd.DataFrame(list(class_names.items()), columns=['Class ID', 'Class Name'])
            st.dataframe(class_df)

else:
    st.error("Unable to load the trained model. Please ensure the model files are available.")
    st.info("Run the Jupyter notebook to train the model first.")

