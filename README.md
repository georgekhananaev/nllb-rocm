# NLLB Translation API for AMD GPUs (ROCm)

A self-hosted multilingual translation API using Meta's **NLLB-200** model, optimized for **AMD GPUs** via ROCm. Includes FastAPI with Swagger UI and token-based authentication.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![ROCm](https://img.shields.io/badge/ROCm-6.2-red.svg)

## Features

- **AMD GPU Acceleration** - Runs on AMD GPUs using ROCm (not just NVIDIA!)
- **200+ Languages** - Full NLLB-200 model support
- **FastAPI + Swagger UI** - Interactive API documentation at `/docs`
- **Token Authentication** - Secure API access with Bearer tokens
- **SQLite Token Management** - Create, list, and revoke API tokens
- **Thread-safe** - Handles concurrent requests properly
- **Docker Ready** - Complete containerized setup

## Supported Hardware

### Tested GPU

| GPU | Architecture | VRAM | Status |
|-----|--------------|------|--------|
| AMD Radeon RX 6600 | RDNA2 (gfx1032) | 8GB | ✅ Tested & Working |

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
- **~4GB RAM** for model loading
- **~60GB disk** for Docker image

## Quick Start

### 1. Clone and Download Model

```bash
git clone https://github.com/yourusername/nllb-rocm.git
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
    pip install transformers sentencepiece flask tokenizers pybind11 fastapi uvicorn &&
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

**Option A: Using Docker Compose (recommended)**

```bash
# Edit .env with your ADMIN_TOKEN first
docker-compose up -d
```

**Option B: Using Docker directly**

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
```

## API Usage

### Documentation

- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

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
    "target_lang": "ukr_Cyrl"
  }'
```

Response:
```json
{
  "translation": "Привіт, як ти сьогодні?",
  "source_lang": "eng_Latn",
  "target_lang": "ukr_Cyrl",
  "device": "cuda"
}
```

### Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/translate` | POST | Token | Translate text |
| `/health` | GET | None | Health check |
| `/languages` | GET | None | List language codes |
| `/docs` | GET | None | Swagger UI |
| `/admin/tokens` | POST | Admin | Create token |
| `/admin/tokens` | GET | Admin | List tokens |
| `/admin/tokens/{id}` | DELETE | Admin | Deactivate token |

## Language Codes

NLLB uses BCP-47-like codes. Common examples:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `eng_Latn` | English | `zho_Hans` | Chinese (Simplified) |
| `ukr_Cyrl` | Ukrainian | `jpn_Jpan` | Japanese |
| `rus_Cyrl` | Russian | `kor_Hang` | Korean |
| `deu_Latn` | German | `ara_Arab` | Arabic |
| `fra_Latn` | French | `hin_Deva` | Hindi |
| `spa_Latn` | Spanish | `por_Latn` | Portuguese |

Full list: [FLORES-200 Languages](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/nllb-200-3.3B-ct2-int8` | Path to model |
| `DB_PATH` | `/data/tokens.db` | SQLite database path |
| `ADMIN_TOKEN` | `admin-secret-change-me` | Admin authentication token (**change this!**) |
| `HSA_OVERRIDE_GFX_VERSION` | - | GPU architecture override |
| `BATCH_TIMEOUT_MS` | `50` | Max wait time (ms) for batch to fill |
| `MAX_BATCH_SIZE` | `8` | Maximum requests per batch |
| `BEAM_SIZE` | `1` | Beam size (1=fast, 4=quality) |
| `INTER_THREADS` | `2` | Number of HIP/CUDA streams |

See `.env.example` for a complete list with descriptions.

### GPU Architecture Override

For RX 6600 (gfx1032), set `HSA_OVERRIDE_GFX_VERSION=10.3.0` to emulate gfx1030.

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
├── docker-compose.yml          # Docker Compose config
├── Dockerfile.rocm-build       # Build instructions
├── .env.example                # Environment variables template
└── README.md
```

## Why This Project?

The standard `pip install ctranslate2` only supports **NVIDIA CUDA**. This project provides:

1. **ROCm Patch** - Enables CTranslate2 on AMD GPUs
2. **Complete Setup** - Docker image with everything pre-configured
3. **Production Ready** - Auth, docs, and proper error handling

## Performance

| Metric | Value |
|--------|-------|
| Model Loading | ~10-15 seconds |
| Translation Latency | ~0.5-2 seconds |
| GPU Memory Usage | ~3.5GB |
| Concurrent Requests | Queued (thread-safe) |

## Troubleshooting

### GPU not detected
- Ensure `/dev/kfd` and `/dev/dri` are passed to container
- Check `HSA_OVERRIDE_GFX_VERSION` matches your GPU
- Verify ROCm works on host: `rocm-smi`

### "CUDA driver version insufficient"
- This means CTranslate2 pip package (NVIDIA only) is installed
- Rebuild with the ROCm patch as shown above

### AppArmor permission denied
- Use `--privileged` flag or configure AppArmor

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
