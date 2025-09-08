import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import grpc
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, Query

from logger import logger
from config import Config
from setup import setup_qdrant
from models import (
    DeleteResponse,
    RegisterResponse,
    RecognizeResponse,
    VerifyResponse,
    IdentityItem,
    GetIdentitiesResponse
)

from grpc_health.v1 import health_pb2
from grpc_services import embeddings_pb2_grpc
from grpc_services import face_attributes_pb2_grpc
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct, UpdateStatus
from helpers import get_attributes_async, get_embedding_async, check_grpc_service_async, parse_estimated_attributes

load_dotenv()
config = Config()
FACE_COLLECTION_NAME = "faces_embeddings"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of FastAPI app, initializing Qdrant and gRPC connections."""
    logger.info("Starting API service...")

    qdrant_client = config.get_qdrant_client()
    max_retries = 10
    wait = 5  # initial wait in seconds
    for i in range(max_retries):
        try:
            setup_qdrant(qdrant_client)
            logger.info("Qdrant ready.")
            break
        except Exception as e:
            logger.warning(f"Qdrant not ready (attempt {i+1}/{max_retries}): {e}")
            await asyncio.sleep(wait)
            wait += 5
    else:
        raise RuntimeError("Qdrant not available after retries. Shutting down API.")


    embedding_target = f"{config.embedder_host}:{config.embedder_port}"
    attributes_target = f"{config.face_attributes_host}:{config.face_attributes_port}"
    embedding_channel = grpc.insecure_channel(embedding_target)
    attribute_channel = grpc.insecure_channel(attributes_target)
    embedding_stub = embeddings_pb2_grpc.EmbeddingServiceStub(embedding_channel)
    attribute_stub = face_attributes_pb2_grpc.AttributeServiceStub(attribute_channel)

    for i in range(max_retries):
        embedding_ok, attributes_ok = await asyncio.gather(
            check_grpc_service_async(embedding_target, "embeddings.EmbeddingService"),
            check_grpc_service_async(attributes_target, "faceattributes.AttributeService"),
        )
        if embedding_ok and attributes_ok:
            app.state.grpc_health_status = {
    "embedding_service": "healthy" if embedding_ok else "unhealthy",
    "attribute_service": "healthy" if attributes_ok else "unhealthy",
}
            break
        logger.warning(f"gRPC services not ready (attempt {i+1}/{max_retries}). Retrying in {wait}s...")
        await asyncio.sleep(wait)
        wait += 5
    else:
        embedding_channel.close()
        attribute_channel.close()
        raise RuntimeError("Critical backend gRPC services unavailable. Shutting down API.")

    app.state.qdrant_client = qdrant_client
    app.state.embedding_channel = embedding_channel
    app.state.attribute_channel = attribute_channel
    app.state.embedding_stub = embedding_stub
    app.state.attribute_stub = attribute_stub

    logger.info("Startup complete. All services healthy.")
    yield

    try: embedding_channel.close()
    except Exception: pass
    try: attribute_channel.close()
    except Exception: pass
    logger.info("Shutting down API service and gRPC channels.")


app = FastAPI(
    title="Face Recognition Service API",
    version="1.0.0",
    lifespan=lifespan,
    description="An API for registering, recognizing, verifying, and managing face identities.",
)


@app.get("/health", tags=["Health"])
async def health_setup(request: Request):
    """Check API health and dependencies (Qdrant collections, gRPC services)."""
    try:
        collections = [c.name for c in request.app.state.qdrant_client.get_collections().collections]
        return {"status": "ok", "dependencies": {"qdrant_collections": collections, "grpc_services": request.app.state.grpc_health_status}}
    except Exception as e:
        logger.error(f"Health setup check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: Qdrant connection failed: {e}")


@app.post("/register", response_model=RegisterResponse, tags=["Face Recognition"])
async def register_identity(
    request: Request,
    face_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    """
    Register a new face identity with an optional UUID and metadata.
    
    Args:
        face_id: Optional UUID for the identity (auto-generated if None).
        file: Image file containing the face.
        metadata: Optional JSON string with additional identity data.
    
    Returns:
        RegisterResponse with face_id, metadata, and estimated attributes.
    """
    try:
        image_data = await file.read()
    except Exception as e:
        logger.error(f"Failed reading uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Invalid file upload.")

    if face_id:
        try:
            face_id = str(uuid.UUID(face_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="face_id must be a valid UUID (with or without hyphens).")
    else:
        face_id = str(uuid.uuid4())

    embed_stub = request.app.state.embedding_stub
    attr_stub = request.app.state.attribute_stub

    embedding_task = get_embedding_async(embed_stub, image_data)
    attributes_task = get_attributes_async(attr_stub, image_data)
    embedding, attributes = await asyncio.gather(embedding_task, attributes_task)
    estimated_attrs = parse_estimated_attributes(attributes)
    vec = np.array(embedding, dtype=float)
    norm = np.linalg.norm(vec)
    vec = (vec / norm).tolist() if norm > 0 else vec.tolist()

    metadata_dict: Dict[str, Any] = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            if not isinstance(metadata_dict, dict):
                logger.warning("Invalid metadata provided: metadata must be a JSON object. Ignoring metadata.")
                metadata_dict = {}
        except Exception as e:
            logger.warning(f"Invalid metadata provided: {e}. Ignoring metadata.")
            metadata_dict = {}

    payload = {
        "face_id": face_id,
        "metadata": metadata_dict,
        "attributes_complete": attributes,
    }

    point_id = face_id
    qdrant_client = request.app.state.qdrant_client
    try:
        qdrant_client.upsert(
            collection_name=FACE_COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vec, payload=payload)],
            wait=True,
        )
    except Exception as e:
        logger.error(f"Qdrant upsert failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store embedding in vector DB.")

    logger.info(f"Registered face_id='{face_id}' qdrant_id='{point_id}' attributes_status={attributes.get('status')}")
    
    return {
        "status": "success",
        "message": "Identity registered successfully.",
        "face_id": face_id,
        "metadata": metadata_dict,
        "estimated_attributes": estimated_attrs
    }


@app.post("/recognize", response_model=RecognizeResponse, tags=["Face Recognition"])
async def recognize_identity(request: Request, file: UploadFile = File(...), threshold: float = Form(0.80)):
    """
    Recognize a face by comparing its embedding to stored identities.

    Args:
        file: Image file containing the face.
        threshold: Minimum similarity score for a match (default: 0.80).

    Returns:
        RecognizeResponse with matched face_id, score, metadata, and attributes if above threshold, else None.
    """
    try:
        image_data = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file upload.")

    embedding = await get_embedding_async(request.app.state.embedding_stub, image_data)
    vec = np.array(embedding, dtype=float)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = (vec / norm).tolist()
    else:
        vec = vec.tolist()

    qdrant_client = request.app.state.qdrant_client
    try:
        search_result = qdrant_client.search(
            collection_name=FACE_COLLECTION_NAME,
            query_vector=vec,
            limit=1,
            with_payload=True
        )
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        raise HTTPException(status_code=503, detail="Vector DB search failed.")

    if not search_result or getattr(search_result[0], "score", 0.0) < threshold:
        logger.info("Recognition below threshold -> unknown")
        return {"status": "success", "result": None}

    best = search_result[0]
    face_id = best.payload.get("face_id", "unknown")
    score = float(best.score)
    
    metadata = best.payload.get("metadata", {})
    attributes = best.payload.get("attributes_complete", {})
    parsed_attributes = parse_estimated_attributes(attributes)
    logger.info(f"Recognized {face_id} score={score:.4f}")
    return {
        "status": "success",
        "result": {
            "face_id": face_id,
            "score": score,
            "metadata": metadata,
            "estimated_attributes": parsed_attributes
        }
    }


@app.post("/verify", response_model=VerifyResponse, tags=["Face Recognition"])
async def verify_identity(
    request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...), threshold: float = Form(0.85)
):
    """
    Verify if two face images belong to the same identity.

    Args:
        file1: First image file containing a face.
        file2: Second image file containing a face.
        threshold: Minimum similarity score for verification (default: 0.85).

    Returns:
        VerifyResponse with verification result and similarity score.
    """
    try:
        img1 = await file1.read()
        img2 = await file2.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file upload(s).")

    embed_stub = request.app.state.embedding_stub
    e1_task = get_embedding_async(embed_stub, img1)
    e2_task = get_embedding_async(embed_stub, img2)
    emb1, emb2 = await asyncio.gather(e1_task, e2_task)

    v1 = np.array(emb1, dtype=float)
    v2 = np.array(emb2, dtype=float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        raise HTTPException(status_code=500, detail="Invalid embeddings (zero norm).")

    similarity = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    verified = similarity >= threshold
    logger.info(f"Verify score={similarity:.4f} threshold={threshold} verified={verified}")
    return {"verified": bool(verified), "score": similarity, "status": "success"}


@app.delete("/identities/{face_id}", response_model=DeleteResponse, tags=["Face Recognition"])
async def delete_identity(request: Request, face_id: str):
    """
    Delete a registered face identity by UUID.

    Args:
        face_id: UUID of the identity to delete.

    Returns:
        DeleteResponse confirming deletion.
    """
    try:
        face_id = str(uuid.UUID(face_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="face_id must be a valid UUID (with or without hyphens).")

    qdrant_client = request.app.state.qdrant_client
    try:
        result = qdrant_client.delete(
            collection_name=FACE_COLLECTION_NAME,
            points_selector=[face_id],
            wait=True,
        )
    except Exception as e:
        logger.error(f"Qdrant delete failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete from vector DB.")

    if getattr(result, "status", None) == UpdateStatus.COMPLETED:
        logger.warning(f"Deleted identity {face_id}")
        return {"status": "success", "message": f"Identity '{face_id}' deleted."}
    else:
        logger.error(f"Delete failed for {face_id} status={result.status}")
        raise HTTPException(status_code=500, detail=f"Failed to delete identity '{face_id}'.")


@app.get("/identities", response_model=GetIdentitiesResponse, tags=["Face Recognition"])
async def get_identities(
    request: Request,
    page: int = Query(1, ge=1, description="The page number to retrieve."),
    per_page: int = Query(20, ge=1, le=100, description="The number of identities to return per page."),
):
    """
    Retrieve a paginated list of registered face identities.

    Args:
        page: Page number to retrieve (default: 1).
        per_page: Number of identities per page (default: 20, max: 100).

    Returns:
        GetIdentitiesResponse with list of identities, pagination info, and total count.
    """
    qdrant_client = request.app.state.qdrant_client
    offset = (page - 1) * per_page
    
    try:
        points, _ = qdrant_client.scroll(
            collection_name=FACE_COLLECTION_NAME,
            limit=per_page,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        logger.error(f"Qdrant scroll operation failed: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching identities.")

    identities = []
    for point in points:
        payload = point.payload or {}
        face_id = payload.get("face_id", "Unknown")
        metadata = payload.get("metadata", {})
        raw_attributes = payload.get("attributes_complete", {})
        estimated_attrs = parse_estimated_attributes(raw_attributes)
        
        identities.append(
            IdentityItem(
                face_id=face_id,
                metadata=metadata,
                estimated_attributes=estimated_attrs
            )
        )

    total_points = 0
    try:
        collection_info = qdrant_client.get_collection(FACE_COLLECTION_NAME)
        total_points = collection_info.points_count
    except Exception as e:
        logger.warning(f"Could not retrieve total collection count, pagination data may be incomplete: {e}")
        total_points = len(identities)

    return {
        "status": "success",
        "total": total_points,
        "page": page,
        "per_page": per_page,
        "identities": identities
    }