# Traffic Sign Recognition System

An advanced AI-powered traffic sign classification system using deep learning and computer vision techniques.

## Features

- **Deep Learning Model**: CNN architecture with batch normalization and dropout
- **43 Traffic Sign Classes**: Comprehensive classification using GTSRB dataset
- **Interactive UI**: Beautiful Streamlit web interface with animations
- **Real-time Prediction**: Fast inference with confidence scores
- **Data Augmentation**: Enhanced model generalization

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
Open and run the Jupyter notebook:
```bash
jupyter notebook traffic_sign_recognition.ipynb
```

Run all cells to:
- Load and preprocess the GTSRB dataset
- Train the CNN model
- Evaluate performance
- Save the model in H5 and pickle formats

### 2. Run the Web Application
Launch the Streamlit app:
```bash
streamlit run app.py
```

## Dataset Structure

The system expects the files to be organized as follows:
```

├── app.py
├── traffic_sign_recognition.ipynb
├── requirements
├── traffic_sign_model.pkl
└── class_names.pkl
```

## Model Architecture

- **Input**: 32x32 RGB images
- **Architecture**: Multi-layer CNN with batch normalization
- **Output**: 43 traffic sign classes
- **Optimization**: Adam optimizer with learning rate scheduling

## Performance

- High accuracy on validation set
- Robust to various lighting conditions
- Fast inference time for real-time applications

## Created by Aatiqa Sadiq - Elevvo Pathways





