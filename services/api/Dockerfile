# -------- Builder stage --------
FROM python:3.11-slim AS builder
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
    
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git && \
        rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools && \
        pip wheel --wheel-dir=/wheels -r requirements.txt
    
COPY . .
    
# -------- Final stage --------
FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
    
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 ca-certificates curl && \
        rm -rf /var/lib/apt/lists/*
    
COPY --from=builder /wheels /wheels
COPY --from=builder /app /app
    
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels