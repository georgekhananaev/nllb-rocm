# NLLB Translation Service with AMD GPU (ROCm)

A self-hosted multilingual translation service using Meta's NLLB-200 model running on AMD GPUs via ROCm.

## Overview

This project runs the **NLLB-200-3.3B-ct2-int8** model (No Language Left Behind) on an AMD RX 6600 GPU using CTranslate2 compiled with ROCm support. The model supports translation between 200+ languages.

### Model Details

- **Model**: `nllb-200-3.3B-ct2-int8` from [OpenNMT/nllb-200-3.3B-ct2-int8](https://huggingface.co/OpenNMT/nllb-200-3.3B-ct2-int8)
- **Parameters**: 3.3 billion (quantized to int8)
- **Size**: ~3.2GB on disk
- **Quantization**: int8 (CPU) / int8_float16 (GPU)
- **Framework**: CTranslate2 with ROCm/HIP backend

### Why Custom Build?

The standard `pip install ctranslate2` package only supports NVIDIA CUDA. For AMD GPUs (ROCm), CTranslate2 must be built from source with HIP support. This project includes:

- A ROCm patch (`ct2_rocm.patch`) for CTranslate2 v3.23.0
- Pre-built Docker image with ROCm-enabled CTranslate2

## Requirements

- AMD GPU with ROCm support (tested on RX 6600)
- Docker with GPU passthrough
- ~4GB GPU memory for inference
- ~60GB disk space for Docker image

## Quick Start

### Start the Translation Service

```bash
docker run -d \
    --privileged \
    --name nllb-translator \
    -v /root/nllb-translation/models:/models:ro \
    -v /root/nllb-translation/scripts:/scripts:ro \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -e MODEL_PATH=/models/nllb-200-3.3B-ct2-int8 \
    -e LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib \
    --device /dev/kfd:/dev/kfd \
    --device /dev/dri:/dev/dri \
    -p 5000:5000 \
    nllb-rocm-gpu:latest \
    python /scripts/translate.py
```

### Verify GPU Usage

```bash
docker logs nllb-translator
```

Expected output:
```
GPU detected via PyTorch: AMD Radeon RX 6600
Loading model from /models/nllb-200-3.3B-ct2-int8
Attempting device: cuda, Compute type: int8_float16
Model loaded successfully on cuda!
Starting translation server on port 5000...
```

## API Usage

### Translate Text

```bash
curl -X POST http://localhost:5000/translate \
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

### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "ok",
  "device": "cuda"
}
```

### List Supported Languages

```bash
curl http://localhost:5000/languages
```

## Language Codes

NLLB uses BCP-47-like language codes. Common examples:

| Code | Language |
|------|----------|
| `eng_Latn` | English |
| `ukr_Cyrl` | Ukrainian |
| `rus_Cyrl` | Russian |
| `deu_Latn` | German |
| `fra_Latn` | French |
| `spa_Latn` | Spanish |
| `ita_Latn` | Italian |
| `pol_Latn` | Polish |
| `por_Latn` | Portuguese |
| `nld_Latn` | Dutch |
| `zho_Hans` | Chinese (Simplified) |
| `jpn_Jpan` | Japanese |
| `kor_Hang` | Korean |
| `ara_Arab` | Arabic |
| `tur_Latn` | Turkish |

Full list: [NLLB Language Codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)

## CLI Mode

You can also use the script directly for one-off translations:

```bash
docker exec nllb-translator bash -c \
  "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH && \
   SOURCE_LANG=eng_Latn TARGET_LANG=deu_Latn \
   python /scripts/translate.py 'Hello world'"
```

## Project Structure

```
nllb-translation/
├── models/
│   └── nllb-200-3.3B-ct2-int8/    # Model files (3.2GB)
│       ├── model.bin              # Quantized weights
│       ├── config.json
│       ├── tokenizer.json
│       ├── shared_vocabulary.json
│       └── ...
├── scripts/
│   └── translate.py               # Translation service
├── ct2_rocm.patch                 # ROCm patch for CTranslate2
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.rocm-build          # Build from source
└── README.md
```

## Building from Source

If you need to rebuild CTranslate2 with ROCm support:

```bash
# Start build container
docker run -d --privileged --name ct2-builder \
    -v /root/nllb-translation:/workspace \
    rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0 \
    sleep infinity

# Install dependencies
docker exec ct2-builder apt-get update
docker exec ct2-builder apt-get install -y git cmake ninja-build

# Clone and patch CTranslate2
docker exec ct2-builder bash -c "
    cd /build && \
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    git checkout v3.23.0 && \
    cp /workspace/ct2_rocm.patch . && \
    git apply ct2_rocm.patch
"

# Build with ROCm (gfx1030 for RX 6600)
docker exec ct2-builder bash -c "
    cd /build/CTranslate2 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DGPU_RUNTIME=HIP \
        -DWITH_CUDNN=ON \
        -DCMAKE_HIP_ARCHITECTURES='gfx1030' \
        -DWITH_MKL=OFF \
        -DOPENMP_RUNTIME=COMP \
        -GNinja && \
    ninja -j\$(nproc) && \
    ninja install
"

# Build Python bindings
docker exec ct2-builder bash -c "
    export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH && \
    export LIBRARY_PATH=/usr/local/lib:\$LIBRARY_PATH && \
    cd /build/CTranslate2/python && \
    pip install --no-build-isolation .
"

# Commit the image
docker commit ct2-builder nllb-rocm-gpu:latest
```

## GPU Architecture Notes

For AMD RX 6600 (gfx1032/RDNA2), use:
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` to emulate gfx1030
- Build CTranslate2 with `-DCMAKE_HIP_ARCHITECTURES="gfx1030"`

## Performance

- Model loading: ~10-15 seconds
- Translation latency: ~0.5-2 seconds (depends on text length)
- Memory usage: ~3.5GB GPU memory

## Troubleshooting

### GPU not detected
- Ensure `/dev/kfd` and `/dev/dri` are passed to the container
- Check `HSA_OVERRIDE_GFX_VERSION` is set correctly for your GPU
- Verify ROCm is working: `rocm-smi` on host

### Falls back to CPU
- Check `LD_LIBRARY_PATH` includes `/usr/local/lib`
- Verify CTranslate2 was built with ROCm support

### AppArmor permission denied
- Use `--privileged` flag or configure AppArmor profiles

## License

- NLLB-200 model: CC-BY-NC 4.0 (Meta AI)
- CTranslate2: MIT License
- This project: MIT License
