# PoseBalanceBot: Self-Balancing Robot with AI Workout Pose Analysis

**Author:** Shinnachot Jeerakan ([nathanjeerk](https://github.com/nathanjeerk))  
**University:** University of Nottingham – Final Year Project 2025

---

## Overview

**PoseBalanceBot** is an intelligent self-balancing robot designed to analyze human workout poses in real-time using AI. Built with a Raspberry Pi 4, the robot leverages MediaPipe for pose estimation, Python-based machine learning models for classification, and a PID controller for physical balancing.

This project bridges **embedded control systems**, **computer vision**, and **AI-based fitness feedback**, showcasing real-time integration of robotics and machine learning.

---

## Features

- **Real-time Pose Detection** using MediaPipe
- **Workout Pose Classification** (Good vs. Bad squat form)
- **PID-based Self-Balancing** system
- **Web Interface** for remote control, PID tuning, and live plotting
- **Live Angle Monitoring** and feedback visualization
- **Offline Training Pipeline** using scikit-learn
- Optimized for Raspberry Pi 4 Model B (4GB)

---

## Project Structure
final_year_project_2025/
├── ai/ # Machine learning models & pose analysis code
├── robot/ # Balancing logic and motor control scripts
├── as/ # (TBD or specify purpose)
├── requirements.txt # Python dependencies
├── README.md # This file
└── index.html # Web-based UI for robot control

---

## Getting Started

### Prerequisites

- Raspberry Pi 4 Model B (4GB)
- Python 3.9+
- OpenCV, MediaPipe, scikit-learn
- Motor driver (e.g., L298N)
- MPU6050 sensor

### Installation

git clone https://github.com/nathanjeerk/final_year_project_2025.git

cd final_year_project_2025

pip install -r requirements.txt

### Model Training

Record pose data using ai/pose_data_collector.py

Train the classifier: python ai/train_model.py

Evaluate results and export model for inference

### Web UI
The robot hosts a Flask-based dashboard featuring:

Joystick-based movement

Live webcam feed

PID graph plotting

Calibration and pose feedback

Access via browser on the same network

### Technologies Used
Python, Flask

MediaPipe, OpenCV

scikit-learn, NumPy

VTK (optional visualization)

HTML/CSS/JavaScript (for web UI)

### Results

Pose classification accuracy: XX%

Balancing repeatability: ±X degrees

Real-time inference latency: X ms

### Future Improvements
Use TensorFlow Lite for faster inference

Add multi-pose support (e.g., lunges, push-ups)

Remote monitoring via cloud dashboard

Add voice or gesture command control
