#frs/services/embeddings/modelmanager.py
import os
import sys 
import numpy as np
import onnxruntime
import cv2
import requests
from typing import List

from config import Config as config 
from logger import logger

class ModelManager:
    """
    Manages the lifecycle of a single ONNX inference model.
    This class is implemented as a singleton to ensure the model is loaded only once.
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
            
        logger.info("Initializing ModelManager...")
        self._ensure_model_is_downloaded()

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = config.ONNX_INTRA_OP_THREADS
        session_options.inter_op_num_threads = config.ONNX_INTER_OP_THREADS
        
        logger.info(f"Attempting to load model with providers: {config.ONNX_PROVIDERS}")
        self.session = onnxruntime.InferenceSession(
            config.MODEL_PATH, 
            providers=config.ONNX_PROVIDERS,
            sess_options=session_options
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        logger.info(f"Model loaded successfully from {config.MODEL_PATH}")
        logger.info(f"   - Execution Providers: {self.session.get_providers()}")
        logger.info(f"   - Input Name: {self.input_name}")
        logger.info(f"   - Input Shape: {self.input_shape}")

        self._initialized = True

    def _ensure_model_is_downloaded(self):
        """Checks if the model file exists and downloads it if missing."""
        model_dir = os.path.dirname(config.MODEL_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        if not os.path.exists(config.MODEL_PATH):
            logger.info(f"Model not found at {config.MODEL_PATH}. Downloading from {config.MODEL_URL}...")
            try:
                response = requests.get(config.MODEL_URL, stream=True)
                response.raise_for_status()
                with open(config.MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("Model downloaded successfully.")
            except Exception as e:
                logger.critical(f"FATAL: Failed to download model: {e}")
                sys.exit(1) 

    def _preprocess(self, image_np: np.ndarray) -> np.ndarray:
       

      
        target_size = (self.input_width, self.input_height)

        image_np = cv2.resize(image_np, target_size)
        logger.info(f"After resizing, image shape: {image_np.shape}")
        
        logger.info("Converting image from BGR to RGB color space.")
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        logger.info(f"After color conversion, image shape: {image_rgb.shape}")
        
        logger.info("Normalizing image pixel values.")
        normalized_image = (image_rgb.astype(np.float32) - 127.5) / 128.0
        logger.info(f"After normalization, image shape: {normalized_image.shape}")
        
        logger.info("Adding batch dimension to the image.")
        preprocessed = normalized_image[np.newaxis, :, :, :]
        logger.info(f"Final preprocessed image shape: {preprocessed.shape}")
        
        logger.info("Image preprocessing completed successfully.")
        return preprocessed

    def get_embedding(self, face_crop_np: np.ndarray) -> List[float]:
        """
        Generates a normalized embedding for a given face crop image.
        """
        preprocessed_image = self._preprocess(face_crop_np)
        embedding = self.session.run([self.output_name], {self.input_name: preprocessed_image})[0]
        
        embedding_norm = np.linalg.norm(embedding)
        normalized_embedding = (embedding / embedding_norm).flatten().tolist()
        
        return normalized_embedding

model_manager = ModelManager()