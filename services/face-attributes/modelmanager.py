# frs/services/face-attributes/modelmanager.py

import os
import sys
import requests
import numpy as np
import onnxruntime
import cv2
from typing import Dict, Any

from config import Config as config
from logger import logger


class ModelManager:
    """
    Singleton class that manages loading and running the FairFace ONNX model
    for face attribute prediction (race, gender, age).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing Face-Attributes ModelManager...")
        self._ensure_model_is_downloaded()

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = config.ONNX_INTRA_OP_THREADS
        session_options.inter_op_num_threads = config.ONNX_INTER_OP_THREADS

        logger.info(f"Loading FairFace ONNX model with providers: {config.ONNX_PROVIDERS}")
        self.session = onnxruntime.InferenceSession(
            config.MODEL_PATH,
            providers=config.ONNX_PROVIDERS,
            sess_options=session_options
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [batch, 3, 224, 224]

        self._initialized = True
        logger.info(f"Model loaded successfully from {config.MODEL_PATH}")
        logger.info(f"   - Providers: {self.session.get_providers()}")
        logger.info(f"   - Input: {self.input_name}, Shape: {self.input_shape}")
        logger.info(f"   - Output: {self.output_name}")

    def _ensure_model_is_downloaded(self):
        """Download model if not already present."""
        model_dir = os.path.dirname(config.MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(config.MODEL_PATH):
            logger.info(f"Model not found at {config.MODEL_PATH}. Downloading from {config.MODEL_URL}...")
            try:
                resp = requests.get(config.MODEL_URL, stream=True)
                resp.raise_for_status()
                with open(config.MODEL_PATH, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("FairFace model downloaded successfully.")
            except Exception as e:
                logger.critical(f"FATAL: Could not download model: {e}")
                sys.exit(1)

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop for FairFace ONNX model.
        Expects BGR input from OpenCV, outputs NCHW float32 normalized.
        """
        # Resize
        img = cv2.resize(face_img, (224, 224))
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)  # (1,3,224,224)
        return img

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for input array."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict_attributes(self, face_crop_np: np.ndarray) -> Dict[str, Any]:
        """
        Predict race, gender, and age for a given face crop with confidence scores.
        """
        preprocessed = self._preprocess(face_crop_np)
        outputs = self.session.run([self.output_name], {self.input_name: preprocessed})[0].squeeze()

        # Define labels for each category
        race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        gender_labels = ['Male', 'Female']
        age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

        # Slice the model outputs
        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:]

        # Compute softmax probabilities for each category
        race_probs = self._softmax(race_outputs)
        gender_probs = self._softmax(gender_outputs)
        age_probs = self._softmax(age_outputs)

        # Create dictionaries mapping labels to probabilities
        race_probs_dict = {label: float(prob) for label, prob in zip(race_labels, race_probs)}
        gender_probs_dict = {label: float(prob) for label, prob in zip(gender_labels, gender_probs)}
        age_probs_dict = {label: float(prob) for label, prob in zip(age_labels, age_probs)}
        
        # Determine the top prediction for each category
        predicted_race = max(race_probs_dict, key=race_probs_dict.get)
        predicted_gender = max(gender_probs_dict, key=gender_probs_dict.get)
        predicted_age_range = max(age_probs_dict, key=age_probs_dict.get)

        return {
            "race": predicted_race,
            "gender": predicted_gender,
            "age_range": predicted_age_range,
            "race_probs": race_probs_dict,
            "gender_probs": gender_probs_dict,
            "age_probs": age_probs_dict
        }

# Global singleton instance
model_manager = ModelManager()
