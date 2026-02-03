# NLLB Translation API for AMD GPUs (ROCm)

A production-grade, self-hosted multilingual translation API using Meta's **NLLB-200** model, optimized for **AMD GPUs** via ROCm. Features FastAPI with Swagger UI, token-based authentication, translation caching, rate limiting, circuit breaker protection, and comprehensive monitoring.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/georgekhananaev/nllb-rocm/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![ROCm](https://img.shields.io/badge/ROCm-6.2-red.svg)](https://rocm.docs.amd.com/)
[![GitHub stars](https://img.shields.io/github/stars/georgekhananaev/nllb-rocm?style=social)](https://github.com/georgekhananaev/nllb-rocm)

## Features

### Core Features
- **AMD GPU Acceleration** - Runs on AMD GPUs using ROCm (not just NVIDIA!)
- **200+ Languages** - Full NLLB-200 model support with all 202 languages
- **FastAPI + Swagger UI** - Interactive API documentation at `/docs`
- **Token Authentication** - Secure API access with Bearer tokens
- **SQLite Token Management** - Create, list, and revoke API tokens

### Production Features (v3.0)
- **Translation Caching** - Dragonfly/Redis cache for 150x faster repeated translations
- **Rate Limiting** - Per-token request limits with automatic cleanup
- **Circuit Breaker** - GPU failure protection with automatic recovery
- **Prometheus Metrics** - Full observability at `/metrics`
- **Quality Levels** - Choose between fast, balanced, or best quality
- **Request Tracing** - X-Request-ID and X-Response-Time headers
- **Health Probes** - Kubernetes-ready liveness and readiness endpoints

### Performance Optimizations
- **Adaptive Batching** - Dynamic batch processing for optimal throughput
- **uvloop** - Faster async event loop (when available)
- **orjson** - Faster JSON serialization (when available)
- **Structured Logging** - JSON logs with structlog (when available)
- **Thread-safe** - Handles concurrent requests properly
- **Memory Efficient** - No memory leaks, periodic cleanup of rate limit data

## Supported Hardware

### Tested GPU

| GPU | Architecture | VRAM | Status |
|-----|--------------|------|--------|
| AMD Radeon RX 6600 | RDNA2 (gfx1032) | 8GB | Tested & Working |

### Should Work (Untested)

| GPU Family | Architecture | Notes |
|------------|--------------|-------|
| RX 6600 XT | RDNA2 (gfx1032) | Same arch as RX 6600 |
| RX 6700 XT | RDNA2 (gfx1031) | May need `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| RX 6800/6900 | RDNA2 (gfx1030) | Native gfx1030 support |
| RX 7000 series | RDNA3 | May require different patches |

### Requirements

- **ROCm 6.x** compatible AMD GPU
- **Docker** with GPU passthrough
- **8GB+ VRAM** recommended for 3.3B model
- **~2.5GB RAM** for service (model + runtime)
- **~60GB disk** for Docker image

## Quick Start

### 1. Clone and Download Model

```bash
git clone https://github.com/georgekhananaev/nllb-rocm.git
cd nllb-rocm

# Copy and configure environment
cp .env.example .env
# Edit .env and set a secure ADMIN_TOKEN

# Create directories
mkdir -p models data

# Download model from HuggingFace (~3.2GB)
# Option A: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download OpenNMT/nllb-200-3.3B-ct2-int8 --local-dir models/nllb-200-3.3B-ct2-int8

# Option B: Using git lfs
git lfs install
git clone https://huggingface.co/OpenNMT/nllb-200-3.3B-ct2-int8 models/nllb-200-3.3B-ct2-int8
```

### 2. Build Docker Image (First Time Only)

The image includes CTranslate2 built from source with ROCm support:

```bash
# Start build container
docker run -d --privileged --name ct2-builder \
    -v $(pwd):/workspace \
    rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0 \
    sleep infinity

# Install dependencies and build (takes ~10-15 minutes)
docker exec ct2-builder bash -c "
    apt-get update && apt-get install -y git cmake ninja-build &&
    pip install transformers sentencepiece flask tokenizers pybind11 fastapi uvicorn \
        uvloop orjson structlog prometheus-client redis pysbd &&
    cd /build &&
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git &&
    cd CTranslate2 && git checkout v3.23.0 &&
    cp /workspace/ct2_rocm.patch . && git apply ct2_rocm.patch &&
    mkdir build && cd build &&
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DGPU_RUNTIME=HIP \
        -DWITH_CUDNN=ON -DCMAKE_HIP_ARCHITECTURES='gfx1030' \
        -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP -GNinja &&
    ninja -j\$(nproc) && ninja install &&
    export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH &&
    export LIBRARY_PATH=/usr/local/lib:\$LIBRARY_PATH &&
    cd /build/CTranslate2/python && pip install --no-build-isolation .
"

# Save the image
docker commit ct2-builder nllb-rocm-gpu:latest
docker rm -f ct2-builder
```

### 3. Run the Service

**Using Docker Compose (recommended)**

```bash
# Edit .env with your ADMIN_TOKEN first
docker-compose up -d
```

This starts both the translation service and Dragonfly cache.

**Using Docker directly (without cache)**

```bash
docker run -d \
    --privileged \
    --name nllb-translator \
    -v $(pwd)/models:/models:ro \
    -v $(pwd)/scripts:/scripts:ro \
    -v $(pwd)/data:/data \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -e MODEL_PATH=/models/nllb-200-3.3B-ct2-int8 \
    -e DB_PATH=/data/tokens.db \
    -e ADMIN_TOKEN=your-secure-admin-token \
    -e CACHE_ENABLED=false \
    -e LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib \
    --device /dev/kfd:/dev/kfd \
    --device /dev/dri:/dev/dri \
    -p 5000:5000 \
    nllb-rocm-gpu:latest \
    python /scripts/translate.py
```

### 4. Verify GPU Usage

```bash
# Check logs
docker logs nllb-translator

# Expected output:
# GPU detected via PyTorch: AMD Radeon RX 6600
# Model loaded successfully on cuda!
# Connected to cache: redis://dragonfly:6379/0
```

## API Usage

### Documentation

- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc
- **Metrics**: http://localhost:5000/metrics

### Authentication

1. **Create an API token** (using admin token):

```bash
curl -X POST http://localhost:5000/admin/tokens \
  -H "Authorization: Bearer your-secure-admin-token" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "description": "My application"}'
```

Response:
```json
{
  "id": 1,
  "name": "my-app",
  "token": "generated-api-token-here",
  "is_active": true
}
```

2. **Translate text** (using API token):

```bash
curl -X POST http://localhost:5000/translate \
  -H "Authorization: Bearer generated-api-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "source_lang": "eng_Latn",
    "target_lang": "ukr_Cyrl",
    "quality": "fast"
  }'
```

Response:
```json
{
  "translation": "Привіт, як ти сьогодні?",
  "source_lang": "eng_Latn",
  "target_lang": "ukr_Cyrl",
  "device": "AMD Radeon RX 6600",
  "cached": false,
  "quality": "fast"
}
```

### Quality Levels

| Level | Beam Size | Latency | Use Case |
|-------|-----------|---------|----------|
| `fast` | 1 | ~0.5-1s | Real-time translation, chatbots |
| `balanced` | 4 | ~1-1.5s | Good quality/speed tradeoff |
| `best` | 8 | ~1.5-2s | Publishing, official documents |

### API Endpoints

#### Translation

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/translate` | POST | Token | Translate text |
| `/languages` | GET | None | List all 202 language codes |

#### Health & Monitoring

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | None | Legacy health check |
| `/health/live` | GET | None | Kubernetes liveness probe |
| `/health/ready` | GET | None | Kubernetes readiness probe |
| `/health/info` | GET | None | Feature status (uvloop, cache, etc.) |
| `/metrics` | GET | None | Prometheus metrics |
| `/docs` | GET | None | Swagger UI |
| `/redoc` | GET | None | ReDoc documentation |

#### Token Management (Admin)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/admin/tokens` | POST | Admin | Create new API token |
| `/admin/tokens` | GET | Admin | List all tokens |
| `/admin/tokens/{id}` | DELETE | Admin | Deactivate token |
| `/admin/tokens/{id}/activate` | POST | Admin | Reactivate token |

#### Cache Management (Admin)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/admin/cache` | DELETE | Admin | Clear all cached translations |
| `/admin/cache/stats` | GET | Admin | Cache statistics (keys, memory) |

#### Circuit Breaker (Admin)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/admin/circuit-breaker/status` | GET | Admin | Current state and failure count |
| `/admin/circuit-breaker/test` | POST | Admin | Simulate failures (testing) |
| `/admin/circuit-breaker/reset` | POST | Admin | Reset to closed state |

## Configuration

### Environment Variables

#### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_TOKEN` | `admin-secret-change-me` | Admin authentication token (**change this!**) |

#### Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/nllb-200-3.3B-ct2-int8` | Path to model |
| `DB_PATH` | `/data/tokens.db` | SQLite database path |

#### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_TIMEOUT_MS` | `50` | Max wait time (ms) for batch to fill |
| `MAX_BATCH_SIZE` | `8` | Maximum requests per batch |
| `BEAM_SIZE` | `1` | Default beam size (1=fast, 4=balanced, 8=best) |
| `INTER_THREADS` | `2` | Number of HIP/CUDA streams |
| `MIN_BATCH_WAIT_MS` | `5` | Min wait before checking queue |
| `MAX_QUEUE_SIZE` | `1000` | Max queue size before rejecting (503) |

#### Cache (Dragonfly/Redis)

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_ENABLED` | `true` | Enable/disable caching |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL (1 hour default) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis/Dragonfly URL |

#### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable/disable rate limiting |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |

#### Circuit Breaker

| Variable | Default | Description |
|----------|---------|-------------|
| `CIRCUIT_BREAKER_ENABLED` | `true` | Enable/disable circuit breaker |
| `CIRCUIT_BREAKER_THRESHOLD` | `5` | Consecutive failures to open |
| `CIRCUIT_BREAKER_TIMEOUT` | `30` | Seconds before half-open |

#### GPU Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HSA_OVERRIDE_GFX_VERSION` | - | GPU architecture override |
| `ROCR_VISIBLE_DEVICES` | `0` | GPU device index |

### Disabling Features

```bash
# Disable caching (useful for testing)
CACHE_ENABLED=false

# Disable rate limiting
RATE_LIMIT_ENABLED=false

# Disable circuit breaker
CIRCUIT_BREAKER_ENABLED=false
```

### Optional Dependencies

Install these for additional features:

```bash
pip install uvloop orjson structlog prometheus-client redis pysbd
```

| Package | Feature |
|---------|---------|
| `uvloop` | Faster async event loop (~10-20% speedup) |
| `orjson` | Faster JSON serialization |
| `structlog` | Structured JSON logging |
| `prometheus-client` | Prometheus metrics at `/metrics` |
| `redis` | Redis/Dragonfly cache support |
| `pysbd` | Better sentence segmentation |

## How It Works

### Circuit Breaker Pattern

The circuit breaker protects against cascading failures when the GPU encounters errors:

```
       ┌─────────────────────────────────────────────────────────┐
       │                                                         │
       ▼                                                         │
   ┌────────┐  5 failures   ┌────────┐  30s timeout  ┌──────────┐
   │ CLOSED │──────────────▶│  OPEN  │──────────────▶│HALF-OPEN │
   │        │               │        │               │          │
   └────────┘               └────────┘               └──────────┘
       ▲                                                  │
       │                    success                       │
       └──────────────────────────────────────────────────┘
                            failure → back to OPEN
```

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation, requests pass through |
| **OPEN** | All requests immediately return 503 |
| **HALF-OPEN** | One test request allowed; success closes, failure reopens |

### Caching Flow

```
Request → Check Cache → HIT? → Return cached (5ms)
                     ↓
                    MISS
                     ↓
              GPU Translation (~1s)
                     ↓
              Store in Cache
                     ↓
              Return response
```

Cache provides **150x speedup** for repeated translations.

### Rate Limiting

- Per-token sliding window rate limiting
- Automatic cleanup of inactive tokens (every 5 minutes)
- Returns 429 Too Many Requests when exceeded

## Monitoring

### Prometheus Metrics

Available at `/metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `translation_requests_total` | Counter | status, source_lang, target_lang, cached | Total requests |
| `translation_latency_seconds` | Histogram | source_lang, target_lang, quality | Request latency |
| `batch_size` | Histogram | - | Batch sizes processed |
| `translation_queue_depth` | Gauge | - | Current queue depth |
| `cache_hits_total` | Counter | - | Cache hits |
| `cache_misses_total` | Counter | - | Cache misses |
| `rate_limit_exceeded_total` | Counter | token_name | Rate limit exceeded events |
| `circuit_breaker_state` | Gauge | - | 0=closed, 1=open, 2=half-open |

### Health Endpoints

```bash
# Liveness probe (is the process running?)
curl http://localhost:5000/health/live
# {"status": "alive"}

# Readiness probe (can it accept traffic?)
curl http://localhost:5000/health/ready
# {"status": "ready", "model_loaded": true, "cache_connected": true,
#  "queue_size": 0, "circuit_breaker": "closed"}

# Feature info (what's enabled?)
curl http://localhost:5000/health/info
# {"uvloop_enabled": true, "orjson_enabled": true, "cache_enabled": true,
#  "cache_connected": true, "rate_limit_enabled": true, ...}
```

## Architecture

```
                    ┌─────────────────┐
                    │   Client App    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   FastAPI       │
                    │   (uvloop)      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Rate Limiter  │   │ Circuit       │   │ Cache Check   │
│ (per token)   │   │ Breaker       │   │ (Dragonfly)   │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ Batch Queue     │
                    │ (async)         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Batch Processor │
                    │ (adaptive)      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Tokenizer     │   │ CTranslate2   │   │ Decoder       │
│ (ThreadPool)  │   │ (GPU/ROCm)    │   │ (ThreadPool)  │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| Model Loading | ~10-15 seconds |
| Translation (uncached) | ~0.5-1 second |
| Translation (cached) | ~5ms |
| Cache Speedup | **150x faster** |
| GPU Memory | ~3.5GB VRAM |
| System Memory | ~2.2GB RAM |
| Memory Leaks | None (tested with 150+ requests) |

## Language Codes

NLLB uses BCP-47-like codes. Common examples:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `eng_Latn` | English | `zho_Hans` | Chinese (Simplified) |
| `ukr_Cyrl` | Ukrainian | `jpn_Jpan` | Japanese |
| `rus_Cyrl` | Russian | `kor_Hang` | Korean |
| `deu_Latn` | German | `arb_Arab` | Arabic |
| `fra_Latn` | French | `hin_Deva` | Hindi |
| `spa_Latn` | Spanish | `por_Latn` | Portuguese |

Full list: Call `/languages` endpoint or see [FLORES-200 Languages](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)

## Project Structure

```
nllb-rocm/
├── models/                     # Model files (not in git)
│   └── nllb-200-3.3B-ct2-int8/
├── scripts/
│   ├── translate.py            # FastAPI application (main)
│   ├── benchmark.py            # Performance benchmarking
│   └── test_translate.py       # Basic translation test
├── data/                       # Token database (not in git)
├── ct2_rocm.patch              # ROCm patch for CTranslate2
├── docker-compose.yml          # Docker Compose (translator + cache)
├── Dockerfile.rocm-build       # Build instructions
├── .env.example                # Environment variables template
└── README.md
```

## Troubleshooting

### GPU not detected
- Ensure `/dev/kfd` and `/dev/dri` are passed to container
- Check `HSA_OVERRIDE_GFX_VERSION` matches your GPU
- Verify ROCm works on host: `rocm-smi`

### "CUDA driver version insufficient"
- This means CTranslate2 pip package (NVIDIA only) is installed
- Rebuild with the ROCm patch as shown above

### Cache not connecting
- Ensure Dragonfly container is running: `docker ps`
- Check network: containers must be on same Docker network
- Verify `CACHE_ENABLED=true` in environment
- Check `/health/info` for `cache_connected` status

### Rate limit errors (429)
- Increase `RATE_LIMIT_REQUESTS` or `RATE_LIMIT_WINDOW_SECONDS`
- Or disable: `RATE_LIMIT_ENABLED=false`

### Circuit breaker open (503)
- Check `/admin/circuit-breaker/status` for failure count
- Reset with `/admin/circuit-breaker/reset`
- Investigate GPU errors in logs

### High memory usage
- Normal: ~2.2GB for service, ~3.5GB VRAM
- If growing: Check for many unique translations (cache grows)
- Clear cache: `DELETE /admin/cache`

## License

- **This project**: MIT License
- **NLLB-200 model**: CC-BY-NC 4.0 (Meta AI)
- **CTranslate2**: MIT License (OpenNMT)

## Contributing

Contributions welcome! Especially:
- Testing on other AMD GPUs (RX 7000 series, etc.)
- Performance optimizations
- Additional language support

## Acknowledgments

- [Meta AI](https://ai.meta.com/) for NLLB-200 model
- [OpenNMT](https://opennmt.net/) for CTranslate2
- [ROCm](https://rocm.docs.amd.com/) for AMD GPU compute platform
- [DragonflyDB](https://www.dragonflydb.io/) for high-performance caching
