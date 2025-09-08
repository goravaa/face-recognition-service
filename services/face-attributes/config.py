#frs/services/face-attributes/config.py
import os
import logging

class Config:
    # Model settings
    MODEL_URL = os.getenv("MODEL_URL", "https://huggingface.co/garavv/fairface-onnx/resolve/main/fairface.onnx")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/fairface.onnx")
    
    # gRPC settings
    GRPC_PORT = int(os.getenv("GRPC_PORT", "50052"))
    GRPC_MAX_WORKERS = int(os.getenv("GRPC_MAX_WORKERS", "10"))
    GRPC_MAX_MESSAGE_LENGTH = eval(os.getenv("GRPC_MAX_MESSAGE_LENGTH", "50 * 1024 * 1024"))  # 50MB
    
    # ONNX settings for performance
    ONNX_PROVIDERS = os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider").split(",")
    ONNX_INTRA_OP_THREADS = int(os.getenv("ONNX_INTRA_OP_THREADS", "4"))
    ONNX_INTER_OP_THREADS = int(os.getenv("ONNX_INTER_OP_THREADS", "2"))
    
     # Logging setting
    _level_name = os.getenv("LOGGING_LEVEL", "INFO").upper()
    LOGGING_LEVEL = getattr(logging, _level_name, logging.ERROR)