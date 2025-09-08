
#frs/services/api/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ConfidenceModel(BaseModel):
    race: float
    gender: float
    age: float

class EstimatedAttributesModel(BaseModel):
    race: str
    gender: str
    age: int
    confidence: ConfidenceModel

class RegisterResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Identity registered successfully.")
    face_id: str = Field(..., example="john_doe_123")
    metadata: Dict[str, Any]
    estimated_attributes: EstimatedAttributesModel
  

class RecognizeResult(BaseModel):
    face_id: str = Field(..., example="jane_doe_456")
    score: float = Field(..., example=0.92)
    metadata: Dict[str, Any]
    estimated_attributes: EstimatedAttributesModel

class RecognizeResponse(BaseModel):
    status: str = Field(..., example="success")
    result: Optional[RecognizeResult] = None
  
class VerifyResponse(BaseModel):
    verified: bool = Field(..., example=True)
    score: float = Field(..., example=0.88)
    status: str = Field(..., example="success")

class DeleteResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Identity 'john_doe_123' deleted.")

class IdentityItem(BaseModel):
    face_id: str
    metadata: Dict[str, Any]
    estimated_attributes: EstimatedAttributesModel

class GetIdentitiesResponse(BaseModel):
    status: str
    total: int
    page: int
    per_page: int
    identities: list[IdentityItem]
