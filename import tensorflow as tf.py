import os
import pickle
import cv2
import numpy as np

import mediapipe as mp

mp_hands = mp.solutions.hands

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks here

               print(f"Image Shape: {img.shape}, Image Size: {img.size}")
