### Face Recognition Service

Production-ready microservice stack for face recognition and face attribute estimation.

- Uses Qdrant as a vector database to store 512‑D ArcFace embeddings
- Runs ArcFace (ONNX Runtime) to generate embeddings
- Runs FairFace (ONNX Runtime) to estimate attributes (age, ethnicity, gender)
- Exposes a FastAPI HTTP API with Swagger UI at `http://localhost:8000`

Works with Docker Compose v2 (tested with Docker Compose version v2.30.3-desktop).

## Quickstart

1) Clone the repository
```bash
git clone https://github.com/your-org/face-recognition-service.git
cd face-recognition-service
```

2) Copy environment template and adjust if needed
```bash
cp .env.example .env
```

3) Start the stack
```bash
docker compose up -d --build
```

4) Open the API docs
```text
http://localhost:8000
```
Swagger UI is available at `/docs` and ReDoc at `/redoc`.

## Services

- API (FastAPI, port 8000): orchestrates embedding + attribute inference and stores/retrieves vectors from Qdrant
- Embeddings (gRPC): ArcFace ONNX model to produce 512‑D normalized vectors
- Attributes (gRPC): FairFace ONNX model to estimate age, ethnicity, and gender
- Qdrant (port 6333): vector database for similarity search

## Environment Variables

Create `.env` from `.env.example`. Defaults are sane for local usage. Key variables:

- API
  - `API_WORKERS`: Gunicorn workers (e.g., 2)
  - `API_TIMEOUT`: Gunicorn timeout seconds (e.g., 120)
  - `EMBEDDINGS_HOST`, `EMBEDDINGS_PORT`: gRPC embeddings target for API (e.g., `embeddings`, `50051`)
  - `ATTRIBUTES_HOST`, `ATTRIBUTES_PORT`: gRPC attributes target for API (e.g., `attributes`, `50052`)
  - `QDRANT_URL`: Qdrant URL (e.g., `http://qdrant:6333`)
  - `QDRANT_API_KEY`: API key used for Qdrant (also set on Qdrant container)

- Embeddings service
  - `EMBEDDINGS_MODEL_URL`: Remote URL to `arc.onnx` (defaults to a public model)
  - `EMBEDDINGS_MODEL_PATH`: Local path inside container (`./models/arc.onnx`)
  - `EMBEDDINGS_GRPC_PORT`: Default `50051`
  - `GRPC_MAX_WORKERS`, `GRPC_MAX_MESSAGE_LENGTH`
  - `ONNX_PROVIDERS`: e.g., `CPUExecutionProvider`
  - `ONNX_INTRA_OP_THREADS`, `ONNX_INTER_OP_THREADS`

- Attributes service
  - `ATTRIBUTES_MODEL_URL`: Remote URL to `fairface.onnx`
  - `ATTRIBUTES_MODEL_PATH`: Local path inside container (`./models/fairface.onnx`)
  - `ATTRIBUTES_GRPC_PORT`: Default `50052`
  - `GRPC_MAX_WORKERS`, `GRPC_MAX_MESSAGE_LENGTH`
  - `ONNX_PROVIDERS`, `ONNX_INTRA_OP_THREADS`, `ONNX_INTER_OP_THREADS`

- Qdrant
  - `QDRANT_API_KEY`: propagated into container as `QDRANT__SERVICE__API_KEY`

## API Overview

Base URL: `http://localhost:8000`

- Health
  - `GET /health` → status of API, gRPC backends, and Qdrant collections

- Face Recognition
  - `POST /register` → register an identity into Qdrant
  - `POST /recognize` → recognize an identity against stored vectors
  - `POST /verify` → verify if two images are the same identity
  - `DELETE /identities/{face_id}` → delete an identity by UUID
  - `GET /identities?page=&per_page=` → list identities, paginated

Swagger examples are embedded in the schema. Minimal curl examples are below.

### Curl examples

Register (with optional metadata JSON):
```bash
curl -X POST "http://localhost:8000/register" \
  -F "file=@/path/to/image.jpg" \
  -F 'metadata={"name":"Alice"};type=application/json'
```

Recognize (with threshold):
```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "file=@/path/to/image.jpg" \
  -F "threshold=0.80"
```

Verify two images:
```bash
curl -X POST "http://localhost:8000/verify" \
  -F "file1=@/path/to/image1.jpg" \
  -F "file2=@/path/to/image2.jpg" \
  -F "threshold=0.85"
```

Delete identity:
```bash
curl -X DELETE "http://localhost:8000/identities/<face_uuid>"
```

List identities:
```bash
curl "http://localhost:8000/identities?page=1&per_page=20"
```

## Architecture

The API service calls two internal gRPC services:

- EmbeddingService (`embeddings.EmbeddingService`) → returns `repeated float embedding` (512‑D) + status
- AttributeService (`faceattributes.AttributeService`) → returns age, gender, race, and probability maps

Faces are stored in Qdrant collection `faces_embeddings` with payload:

```json
{
  "face_id": "<uuid>",
  "metadata": { /* optional user data */ },
  "attributes_complete": { /* raw attribute response */ }
}
```

Vector config: 512 dimensions, COSINE distance.

## Notes

- First startup downloads ONNX models if not present (see service `MODEL_URL` defaults)
- Health probes are enabled for all services; API waits for gRPC backends and Qdrant to be ready
- Tested with Docker Compose v2.30.3-desktop

## Development

To run locally without Docker, ensure Qdrant is running, set environment variables from `.env.example`, and start each service. Docker is recommended for a consistent environment.

## License

See `LICENSE`.