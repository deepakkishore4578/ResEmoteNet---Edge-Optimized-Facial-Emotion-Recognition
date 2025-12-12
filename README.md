# ResEmoteNet: Edge-Optimized Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![ONNX](https://img.shields.io/badge/ONNX-Runtime-grey) ![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)

## 📌 Project Overview
**ResEmoteNet** is a lightweight, high-performance Convolutional Neural Network (CNN) designed for **real-time facial emotion analysis** on edge robotics hardware. 

This project implements a custom architecture from scratch, integrating **Squeeze-and-Excitation (SE) attention blocks** into a ResNet backbone to dynamically recalibrate feature responses. To address severe class imbalance in the **FER-2013** dataset, a **Stable Diffusion** pipeline was engineered to generate synthetic training samples for underrepresented emotions.

## 🚀 Key Features
* **Custom Architecture:** Implemented `ResEmoteNet` from scratch (PyTorch), integrating channel-wise attention mechanisms for enhanced feature selectivity.
* **Generative Augmentation:** Solved data imbalance by generating **2,000+ synthetic images** (Fear, Disgust) using a Stable Diffusion text-to-image pipeline.
* **Edge Optimization:** Optimized the model for deployment using **ONNX Runtime** and **INT8 quantization**, achieving **~4ms inference latency** on standard CPUs.
* **Real-Time Performance:** Capable of processing **~250 FPS**, making it suitable for human-robot interaction (HRI) systems like Miko robots.

## 📊 Performance Metrics
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Validation Accuracy** | **~80%** | SOTA performance on FER-2013 |
| **Inference Latency** | **3.66 ms** | Benchmarked on CPU via ONNX INT8 |
| **Throughput** | **~270 FPS** | Real-time processing capability |
| **Model Size** | **<15 MB** | Lightweight for embedded storage |

📜 References
Squeeze-and-Excitation Networks (CVPR 2018)

FER-2013 Dataset (Kaggle)

Stable Diffusion v1.5 (RunwayML)
