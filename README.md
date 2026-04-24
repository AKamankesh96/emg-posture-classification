# EMG-Based Classification of Standing Postures

This repository contains code for classifying four standing postures using EMG signals recorded from the **soleus (SOL)** and **flexor digitorum brevis (FDB)**.

The primary objective is to compare how effectively each muscle’s activity can discriminate between standing postures.

---

## Overview

A DeepConvNet-style convolutional neural network is used to classify posture from segmented EMG windows. The analysis includes:

- Model training and evaluation  
- Posture-wise accuracy analysis  
- Performance comparison between SOL and FDB  

---

## Scientific Motivation

Standing balance involves coordinated activation of intrinsic (FDB) and extrinsic (SOL) muscles. This project evaluates whether one muscle group provides more posture-specific EMG information.

---

## Data

The original EMG data are not included due to IRB and study protocol restrictions.

Expected data format:
- Input: n × 64 × 512 EMG windows  
- Labels: integers (1–4)  

---

## Repository Structure
```text
emg-posture-classification/
├── notebooks/
│   └── EMG_Posture_Classification_GitHub.ipynb
├── src/
│   └── emg_posture_classification.py
├── requirements.txt
└── README.md

---

## Notes

- Preprocessing and segmentation are assumed to be completed prior to loading  
- Consider subject-wise splits to avoid data leakage  
- Multiple runs are recommended for stable performance estimates  

---

## Citation

This repository accompanies the following publication:

Kamankesh, A., Rahimi, N., Amiridis, I. G., Sahinis, C., Hatzitaki, V., & Enoka, R. M. (2025).  
Distinguishing the activity of flexor digitorum brevis and soleus across standing postures with deep learning models.  
Gait & Posture, 117, 58–64. https://doi.org/10.1016/j.gaitpost.2024.12.014
