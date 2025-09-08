#frs/services/api/config.py
import os
import grpc
from qdrant_client import QdrantClient
import logging
# Import gRPC generated stub
from grpc_services import embeddings_pb2_grpc
from grpc_services import face_attributes_pb2_grpc


class Config:
    def __init__(self):
        # ---- QDRANT ----
        self.qdrant_url = self._require("QDRANT_URL")
        self.qdrant_api_key = self._require("QDRANT_API_KEY")

        # ---- EMBEDDER GRPC ----
        self.embedder_host = self._require("EMBEDDER_HOST")   
        self.embedder_port = self._require("EMBEDDER_PORT")  

        # ---- FACE ATTRIBUTES GRPC ----
        self.face_attributes_host = self._require("ATTRIBUTES_HOST") 
        self.face_attributes_port = self._require("ATTRIBUTES_PORT")   


    def _require(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    def get_qdrant_client(self) -> QdrantClient:
        return QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

    def get_embedder_client(self):
        channel = grpc.insecure_channel(f"{self.embedder_host}:{self.embedder_port}")
        return embeddings_pb2_grpc.EmbeddingServiceStub(channel)

    def get_face_attributes_client(self):
        channel = grpc.insecure_channel(f"{self.face_attributes_host}:{self.face_attributes_port}")
        return face_attributes_pb2_grpc.AttributeServiceStub(channel)

        # Logging setting
    _level_name = os.getenv("LOGGING_LEVEL", "DEBUG").upper()
    LOGGING_LEVEL = getattr(logging, _level_name, logging.ERROR)