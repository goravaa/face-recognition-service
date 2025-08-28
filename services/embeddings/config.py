#frs/services/embeddings/config.py
import os
from typing import List

class Config:
    # Model settings
    MODEL_URL = os.getenv("MODEL_URL", "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/arc.onnx")
    MODEL_DIMENSIONS = int(os.getenv("MODEL_DIMENSIONS", "512"))
    
    # Scaling settings
    MAX_CONCURRENT_MODELS = int(os.getenv("MAX_CONCURRENT_MODELS", "1"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "1000"))
    
    # gRPC settings
    GRPC_HOST = os.getenv("GRPC_HOST", "0.0.0.0")
    GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
    GRPC_MAX_WORKERS = int(os.getenv("GRPC_MAX_WORKERS", "10"))
    GRPC_MAX_MESSAGE_LENGTH = eval(os.getenv("GRPC_MAX_MESSAGE_LENGTH", "50 * 1024 * 1024"))  # 50MB
    
    # ONNX settings for performance
    ONNX_PROVIDERS = os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider").split(",")
    ONNX_INTRA_OP_THREADS = int(os.getenv("ONNX_INTRA_OP_THREADS", "4"))
    ONNX_INTER_OP_THREADS = int(os.getenv("ONNX_INTER_OP_THREADS", "2"))
    
    # Caching settings
    ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10000"))
    
    # Health and monitoring
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    METRICS_EXPORT_INTERVAL = int(os.getenv("METRICS_EXPORT_INTERVAL", "60"))
    
    # Resource limits
    MEMORY_LIMIT_GB = float(os.getenv("MEMORY_LIMIT_GB", "2.0"))
    CPU_LIMIT = float(os.getenv("CPU_LIMIT", "2.0"))