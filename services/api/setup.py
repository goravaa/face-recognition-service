# frs/services/api/setup.py
from config import Config
from qdrant_client.http.models import VectorParams, Distance
from logger import logger 

def setup_qdrant(qdrant_client):
    collection_name = "faces_embeddings"
    try:
        existing_collections = [c.name for c in qdrant_client.get_collections().collections]
        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection already exists: {collection_name}")
    except Exception as e:
        if "already exists" in str(e):
            logger.info(f"Collection '{collection_name}' already exists, skipping creation.")
  
        logger.error(f"Error while setting up Qdrant collection '{collection_name}': {e}")
