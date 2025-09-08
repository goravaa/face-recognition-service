# helpers.py

import asyncio
import logging
from typing import Any, Dict, List
from logger import logger
import grpc
from fastapi import HTTPException
from grpc_services import (
    embeddings_pb2,
    face_attributes_pb2,
)
from models import EstimatedAttributesModel, ConfidenceModel
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc


def _call_get_embedding_sync(stub, image_data: bytes, timeout: int = 15) -> List[float]:
    """Synchronous gRPC call to get image embedding."""
    req = embeddings_pb2.EmbeddingRequest(image_data=image_data)
    resp = stub.GetEmbedding(req, timeout=timeout)
    if getattr(resp, "status", None) not in ("success", "ok") or not resp.embedding:
        raise RuntimeError("Embedding service returned error or empty embedding")
    return list(resp.embedding)


def _call_get_attributes_sync(stub, image_data: bytes, timeout: int = 15) -> Dict[str, Any]:
    """Synchronous gRPC call to get face attributes."""
    req = face_attributes_pb2.AttributeRequest(image_data=image_data)
    resp = stub.GetAttributes(req, timeout=timeout)
    # normalize response into a dict (complete)
    res = {
        "status": getattr(resp, "status", None),
        "race": getattr(resp, "race", None),
        "gender": getattr(resp, "gender", None),
        "age": getattr(resp, "age", None),
        "race_probs": dict(getattr(resp, "race_probs", {})),
        "gender_probs": dict(getattr(resp, "gender_probs", {})),
        "age_probs": dict(getattr(resp, "age_probs", {})),
        "error_message": getattr(resp, "error_message", None),
    }
    return res


async def get_embedding_async(stub, image_data: bytes) -> List[float]:
    """Asynchronously calls the embedding gRPC service."""
    try:
        return await asyncio.to_thread(_call_get_embedding_sync, stub, image_data)
    except grpc.RpcError as e:
        logger.error(f"Embedding RPC failed: {e}")
        raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {e}")
    except RuntimeError as e:
        logger.error(f"Embedding service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_attributes_async(stub, image_data: bytes) -> Dict[str, Any]:
    """Asynchronously calls the attributes gRPC service."""
    try:
        return await asyncio.to_thread(_call_get_attributes_sync, stub, image_data)
    except grpc.RpcError as e:
        logger.error(f"Attributes RPC failed: {e}")
        raise HTTPException(status_code=503, detail=f"Attribute service unavailable: {e}")
    except Exception as e:
        logger.error(f"Attribute service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def check_grpc_service_async(channel_target: str, service_name: str) -> bool:
    """Run a quick synchronous health check in threadpool to avoid blocking."""
    def _check(target: str, svc: str) -> bool:
        chan = grpc.insecure_channel(target)
        try:
            stub = health_pb2_grpc.HealthStub(chan)
            resp = stub.Check(health_pb2.HealthCheckRequest(service=svc), timeout=5)
            return resp.status == health_pb2.HealthCheckResponse.SERVING
        except Exception as e:
            logger.warning(f"Health check failed for {svc}@{target}: {e}")
            return False
        finally:
            try:
                chan.close()
            except Exception:
                pass

    return await asyncio.to_thread(_check, channel_target, service_name)

def parse_estimated_attributes(attributes: dict) -> EstimatedAttributesModel:
    # Race
    race_probs = attributes.get("race_probs", {})
    race_best = max(race_probs, key=race_probs.get, default="Unknown")
    race_conf = race_probs.get(race_best, 0.0)

    # Gender
    gender_probs = attributes.get("gender_probs", {})
    gender_best = max(gender_probs, key=gender_probs.get, default="Unknown")
    gender_conf = gender_probs.get(gender_best, 0.0)

    # Age
    age_probs = attributes.get("age_probs", {})
    best_bin = max(age_probs, key=age_probs.get, default="0-0")
    age_conf = age_probs.get(best_bin, 0.0)
    if "+" in best_bin:
        low = int(best_bin.replace("+", ""))
        high = low
    else:
        low, high = map(int, best_bin.split("-"))
    age_estimate = (low + high) // 2

    return EstimatedAttributesModel(
        race=race_best,
        gender=gender_best,
        age=age_estimate,
        confidence=ConfidenceModel(
            race=race_conf,
            gender=gender_conf,
            age=age_conf
        )
    )
