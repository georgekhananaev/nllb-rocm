# NLLB Translation with ROCm for AMD RX 6600
FROM rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Install Python dependencies
RUN pip install --no-cache-dir \
    ctranslate2 \
    transformers \
    sentencepiece \
    flask \
    tokenizers \
    fastapi \
    uvicorn \
    pysbd

# Set environment for RX 6600 (gfx1032 -> gfx1030)
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV ROCR_VISIBLE_DEVICES=0

WORKDIR /app
CMD ["/bin/bash"]
