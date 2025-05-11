"""
# Speech Emotion Recognition Using Hybrid CNN-LSTM + HuBERT


A robust speech emotion recognition (SER) system that identifies emotions from audio samples using a hybrid deep learning approach combining CNN-LSTM architecture with HuBERT pre-trained speech representations.
Overview

This project implements an advanced speech emotion recognition system capable of identifying seven emotional categories (angry, disgust, fear, happy, neutral, sad, and surprise) from audio samples with 94% accuracy. The system leverages a novel hybrid architecture that combines traditional acoustic feature extraction with state-of-the-art self-supervised learning representations, enhanced by ensemble techniques.
Key Features

Hybrid CNN-LSTM + HuBERT Architecture: Combines the power of convolutional and recurrent neural networks with pre-trained speech representations
Comprehensive Feature Extraction: Utilizes MFCCs, energy features, spectral contrast, zero-crossing rate, and chroma features
Ensemble Learning: Implements model ensembling to improve robustness and accuracy
High Performance: Achieves 94% accuracy across seven emotion categories
Data Augmentation: Enhances training with pitch shifting, time stretching, and noise addition

# Architecture
Our hybrid model consists of two parallel pathways:
CNN-LSTM Pathway

Three convolutional blocks with increasing filter sizes (32, 64, 128)
Batch normalization, ReLU activation, and dropout for regularization
Bidirectional LSTM with 2 layers and 128 hidden units
Mean pooling across time steps for fixed-length representation

# HuBERT Pathway
Pre-trained HuBERT model (Hidden-Unit BERT for speech representation)
Fine-tuning of higher encoder layers
Custom output processor to extract relevant representations

# Fusion Mechanism

Concatenation of feature vectors from both pathways
Multi-layer fusion network with attention mechanism
Final classification layer with 7 output neurons (one per emotion)
