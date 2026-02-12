#!/usr/bin/env python3
"""NLLB Translation service using CTranslate2 with FastAPI - Production Grade.

Phase 1 & 2 optimizations:
- uvloop for faster async event loop
- orjson for faster JSON serialization
- Prometheus metrics
- Structured logging with structlog
- Redis/Dragonfly translation caching
- Rate limiting per token
- Circuit breaker for GPU failures
- Better sentence segmentation with pysbd
- Request ID tracing
- Configurable quality levels
"""

# Install uvloop before importing asyncio
try:
    import uvloop
    uvloop.install()
    UVLOOP_ENABLED = True
except ImportError:
    UVLOOP_ENABLED = False

import ctranslate2
import os
import sys
import asyncio
import threading
import secrets
import sqlite3
import hashlib
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from transformers import NllbTokenizer
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Try importing optional dependencies
try:
    import orjson
    from fastapi.responses import ORJSONResponse
    ORJSON_ENABLED = True
except ImportError:
    from fastapi.responses import JSONResponse as ORJSONResponse
    ORJSON_ENABLED = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import pysbd
    PYSBD_AVAILABLE = True
except ImportError:
    PYSBD_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ============== Configuration ==============

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/nllb-200-3.3B-ct2-int8")
DB_PATH = os.environ.get("DB_PATH", "/data/tokens.db")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin-secret-change-me")

# Batching configuration
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", "50"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "1"))
INTER_THREADS = int(os.environ.get("INTER_THREADS", "2"))
MIN_BATCH_WAIT_MS = int(os.environ.get("MIN_BATCH_WAIT_MS", "5"))

# Cache configuration
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Circuit breaker configuration
CIRCUIT_BREAKER_ENABLED = os.environ.get("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
CIRCUIT_BREAKER_THRESHOLD = int(os.environ.get("CIRCUIT_BREAKER_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT = int(os.environ.get("CIRCUIT_BREAKER_TIMEOUT", "30"))

# Max queue size before rejecting requests
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "1000"))

# ============== Logging Setup ==============

if STRUCTLOG_AVAILABLE:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger()
else:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    _base_logger = logging.getLogger(__name__)

    # Wrapper to make standard logging compatible with structlog-style calls
    class LoggerWrapper:
        def __init__(self, base_logger):
            self._logger = base_logger

        def _format_msg(self, msg, **kwargs):
            if kwargs:
                extra = ' '.join(f'{k}={v}' for k, v in kwargs.items())
                return f"{msg} | {extra}"
            return msg

        def info(self, msg, **kwargs):
            self._logger.info(self._format_msg(msg, **kwargs))

        def warning(self, msg, **kwargs):
            self._logger.warning(self._format_msg(msg, **kwargs))

        def error(self, msg, **kwargs):
            self._logger.error(self._format_msg(msg, **kwargs))

        def debug(self, msg, **kwargs):
            self._logger.debug(self._format_msg(msg, **kwargs))

    logger = LoggerWrapper(_base_logger)

# ============== Prometheus Metrics ==============

if PROMETHEUS_AVAILABLE:
    TRANSLATION_REQUESTS = Counter(
        'translation_requests_total',
        'Total translation requests',
        ['status', 'source_lang', 'target_lang', 'cached']
    )
    TRANSLATION_LATENCY = Histogram(
        'translation_latency_seconds',
        'Translation request latency',
        ['source_lang', 'target_lang', 'quality'],
        buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    )
    BATCH_SIZE_HISTOGRAM = Histogram(
        'batch_size',
        'Batch sizes processed',
        buckets=[1, 2, 4, 8, 16, 32]
    )
    QUEUE_DEPTH = Gauge(
        'translation_queue_depth',
        'Current translation queue depth'
    )
    CACHE_HITS = Counter(
        'cache_hits_total',
        'Total cache hits'
    )
    CACHE_MISSES = Counter(
        'cache_misses_total',
        'Total cache misses'
    )
    GPU_MEMORY_USED = Gauge(
        'gpu_memory_used_bytes',
        'GPU memory used in bytes'
    )
    RATE_LIMIT_EXCEEDED = Counter(
        'rate_limit_exceeded_total',
        'Total rate limit exceeded events',
        ['token_name']
    )
    CIRCUIT_BREAKER_STATE = Gauge(
        'circuit_breaker_state',
        'Circuit breaker state (0=closed, 1=open, 2=half-open)'
    )

# ============== FastAPI App ==============

app = FastAPI(
    title="NLLB Translation API",
    description="Production-grade multilingual translation service using Meta's NLLB-200 model on AMD GPU",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=ORJSONResponse,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ============== Global State ==============

translator = None
device_used = None
device_name = None
_thread_local = threading.local()
tokenizer_pool = ThreadPoolExecutor(max_workers=4)
batch_queue: asyncio.Queue = None
batch_processor_task = None
redis_client = None
model_loaded = False

# Circuit breaker state
class CircuitBreaker:
    def __init__(self, threshold: int, timeout: int):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.threshold:
                self.state = "open"
                if PROMETHEUS_AVAILABLE:
                    CIRCUIT_BREAKER_STATE.set(1)
                logger.warning("Circuit breaker opened", failures=self.failures)

    def record_success(self):
        with self._lock:
            self.failures = 0
            self.state = "closed"
            if PROMETHEUS_AVAILABLE:
                CIRCUIT_BREAKER_STATE.set(0)

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                    if PROMETHEUS_AVAILABLE:
                        CIRCUIT_BREAKER_STATE.set(2)
                    logger.info("Circuit breaker half-open, allowing test request")
                    return True
                return False
            return True  # half-open allows one request

circuit_breaker = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_TIMEOUT)

# Rate limiter state (in-memory, per token)
rate_limit_store: Dict[str, List[float]] = {}
rate_limit_lock = threading.Lock()
rate_limit_last_cleanup = time.time()

# ============== Pydantic Models ==============

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"
    target_lang: str = "ukr_Cyrl"
    quality: str = Field(default="fast", description="Quality level: 'fast' (beam=1), 'balanced' (beam=4), 'best' (beam=8)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you today?",
                "source_lang": "eng_Latn",
                "target_lang": "ukr_Cyrl",
                "quality": "fast"
            }
        }


class TranslateResponse(BaseModel):
    translation: str
    source_lang: str
    target_lang: str
    device: str
    cached: bool = False
    quality: str = "fast"


class HealthResponse(BaseModel):
    status: str
    device: str
    batch_queue_size: int
    config: dict


class LivenessResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    cache_connected: bool
    queue_size: int
    circuit_breaker: str


class MetricsInfo(BaseModel):
    uvloop_enabled: bool
    orjson_enabled: bool
    cache_enabled: bool
    cache_connected: bool
    rate_limit_enabled: bool
    circuit_breaker_enabled: bool
    circuit_breaker_state: str
    structlog_enabled: bool
    pysbd_enabled: bool
    prometheus_enabled: bool


class TokenCreate(BaseModel):
    name: str
    description: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "my-app",
                "description": "Token for my application"
            }
        }


class TokenResponse(BaseModel):
    id: int
    name: str
    token: str
    description: Optional[str]
    created_at: str
    is_active: bool


class TokenInfo(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: str
    is_active: bool
    usage_count: int


class ErrorResponse(BaseModel):
    error: str
    request_id: Optional[str] = None


# ============== Batch Request Dataclass ==============

@dataclass
class BatchRequest:
    """A single translation request waiting in the batch queue."""
    text: str
    source_lang: str
    target_lang: str
    future: asyncio.Future
    beam_size: int = 1
    source_tokens: List[str] = None


# ============== Request ID Middleware ==============

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"

    return response


# ============== Database ==============

def init_db():
    """Initialize SQLite database for token management."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            token TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            usage_count INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized", path=DB_PATH)


@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def verify_token(token: str) -> Optional[dict]:
    """Verify API token and return token info."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM tokens WHERE token = ? AND is_active = 1",
            (token,)
        )
        row = cursor.fetchone()

        if row:
            cursor.execute(
                "UPDATE tokens SET usage_count = usage_count + 1 WHERE id = ?",
                (row["id"],)
            )
            conn.commit()
            return dict(row)

    return None


# ============== Rate Limiting ==============

def check_rate_limit(token_name: str) -> bool:
    """Check if token has exceeded rate limit. Returns True if allowed."""
    global rate_limit_last_cleanup

    if not RATE_LIMIT_ENABLED:
        return True

    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW_SECONDS

    with rate_limit_lock:
        # Periodic cleanup of inactive tokens (every 5 minutes)
        if current_time - rate_limit_last_cleanup > 300:
            tokens_to_remove = [
                token for token, timestamps in rate_limit_store.items()
                if not timestamps or max(timestamps) < window_start
            ]
            for token in tokens_to_remove:
                del rate_limit_store[token]
            rate_limit_last_cleanup = current_time
            if tokens_to_remove:
                logger.info("Rate limit cleanup", removed_tokens=len(tokens_to_remove))

        if token_name not in rate_limit_store:
            rate_limit_store[token_name] = []

        # Remove old entries for this token
        rate_limit_store[token_name] = [
            t for t in rate_limit_store[token_name] if t > window_start
        ]

        if len(rate_limit_store[token_name]) >= RATE_LIMIT_REQUESTS:
            if PROMETHEUS_AVAILABLE:
                RATE_LIMIT_EXCEEDED.labels(token_name=token_name).inc()
            return False

        rate_limit_store[token_name].append(current_time)
        return True


# ============== Authentication ==============

async def get_api_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Validate Bearer token with rate limiting."""
    token = credentials.credentials

    token_info = verify_token(token)
    if not token_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not check_rate_limit(token_info["name"]):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds.",
        )

    return token_info


async def get_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Validate admin token."""
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# ============== Cache Functions ==============

def get_cache_key(text: str, source_lang: str, target_lang: str, beam_size: int) -> str:
    """Generate cache key from translation parameters."""
    content = f"{text}:{source_lang}:{target_lang}:{beam_size}"
    return f"nllb:{hashlib.sha256(content.encode()).hexdigest()}"


async def get_from_cache(key: str) -> Optional[str]:
    """Get translation from cache."""
    if not CACHE_ENABLED or redis_client is None:
        return None

    try:
        result = await redis_client.get(key)
        if result:
            if PROMETHEUS_AVAILABLE:
                CACHE_HITS.inc()
            return result.decode('utf-8') if isinstance(result, bytes) else result
    except Exception as e:
        logger.warning("Cache get failed", error=str(e))

    if PROMETHEUS_AVAILABLE:
        CACHE_MISSES.inc()
    return None


async def set_in_cache(key: str, value: str):
    """Store translation in cache."""
    if not CACHE_ENABLED or redis_client is None:
        return

    try:
        await redis_client.setex(key, CACHE_TTL_SECONDS, value)
    except Exception as e:
        logger.warning("Cache set failed", error=str(e))


# ============== Model Functions ==============

def get_device():
    """Try to use GPU, fallback to CPU. Returns (device_type, device_name)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("GPU detected", name=gpu_name)
            return "cuda", gpu_name
    except Exception as e:
        logger.warning("PyTorch CUDA check failed", error=str(e))
    return "cpu", "CPU"


def get_tokenizer():
    """Get or create a thread-local tokenizer instance."""
    if not hasattr(_thread_local, 'tokenizer'):
        _thread_local.tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH)
    return _thread_local.tokenizer


def init_model():
    """Initialize the translation model with optimized settings."""
    global translator, device_used, device_name, model_loaded

    device, gpu_name = get_device()
    compute_type = os.environ.get("COMPUTE_TYPE", "int8" if device == "cpu" else "int8_float16")

    logger.info("Loading model", path=MODEL_PATH, device=device, compute_type=compute_type)

    try:
        translator = ctranslate2.Translator(
            MODEL_PATH,
            device=device,
            compute_type=compute_type,
            inter_threads=INTER_THREADS,
            max_queued_batches=-1,
        )
        device_used = device
        device_name = gpu_name
        model_loaded = True
    except Exception as e:
        logger.error("Failed to load model on GPU", error=str(e))
        if device != "cpu":
            logger.info("Falling back to CPU")
            translator = ctranslate2.Translator(
                MODEL_PATH,
                device="cpu",
                compute_type="int8",
                inter_threads=4,
            )
            device_used = "cpu"
            device_name = "CPU"
            model_loaded = True
        else:
            raise

    _ = get_tokenizer()
    logger.info("Model loaded successfully", device=device_name)
    return device_used


MAX_SAFE_TOKENS = int(os.environ.get("MAX_SAFE_TOKENS", "80"))
MAX_DECODE_LENGTH = int(os.environ.get("MAX_DECODE_LENGTH", "200"))


def count_tokens(text: str, source_lang: str) -> int:
    """Count the number of tokens for a text without truncation."""
    tokenizer = get_tokenizer()
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors=None, truncation=False)
    return len(encoded["input_ids"])


def tokenize_text(text: str, source_lang: str) -> List[str]:
    """Tokenize text using a thread-local tokenizer."""
    tokenizer = get_tokenizer()
    tokenizer.src_lang = source_lang

    encoded = tokenizer(text, return_tensors=None, truncation=True, max_length=512)
    source_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    return source_tokens


def decode_tokens(output_tokens: List[str], target_lang: str) -> str:
    """Decode output tokens to text."""
    tokenizer = get_tokenizer()

    if output_tokens and output_tokens[0] == target_lang:
        output_tokens = output_tokens[1:]

    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output_text


def translate_batch_sync(requests: List[BatchRequest]) -> List[str]:
    """Translate a batch of pre-tokenized requests synchronously."""
    if not requests:
        return []

    source_batch = [req.source_tokens for req in requests]
    target_prefixes = [[req.target_lang] for req in requests]

    # Use the beam size from first request (batches are grouped by beam size)
    beam_size = requests[0].beam_size

    max_input_len = max(len(tokens) for tokens in source_batch)
    adaptive_max_length = min(MAX_DECODE_LENGTH, max(64, int(max_input_len * 1.5)))

    logger.info(
        "Translating batch",
        batch_size=len(requests),
        max_input_tokens=max_input_len,
        max_decode_length=adaptive_max_length,
        beam_size=beam_size,
    )

    results = translator.translate_batch(
        source_batch,
        target_prefix=target_prefixes,
        beam_size=beam_size,
        max_decoding_length=adaptive_max_length,
        return_scores=False,
        max_batch_size=0,
        disable_unk=True,
    )

    translations = []
    for i, result in enumerate(results):
        output_tokens = result.hypotheses[0]
        translation = decode_tokens(output_tokens, requests[i].target_lang)
        translations.append(translation)

    return translations


async def process_batch(requests: List[BatchRequest]):
    """Process a batch of requests asynchronously."""
    if not requests:
        return

    loop = asyncio.get_event_loop()

    try:
        # Record batch size
        if PROMETHEUS_AVAILABLE:
            BATCH_SIZE_HISTOGRAM.observe(len(requests))

        # Tokenize in parallel
        tokenize_futures = []
        for req in requests:
            future = loop.run_in_executor(
                tokenizer_pool,
                tokenize_text,
                req.text,
                req.source_lang
            )
            tokenize_futures.append(future)

        tokenized_results = await asyncio.gather(*tokenize_futures)

        for req, tokens in zip(requests, tokenized_results):
            req.source_tokens = tokens

        # GPU translation
        translations = await loop.run_in_executor(
            None,
            translate_batch_sync,
            requests
        )

        # Record success
        circuit_breaker.record_success()

        for req, translation in zip(requests, translations):
            if not req.future.done():
                req.future.set_result(translation)

    except Exception as e:
        circuit_breaker.record_failure()
        logger.error("Batch processing error", error=str(e))
        for req in requests:
            if not req.future.done():
                req.future.set_exception(e)


async def batch_processor():
    """Background task that collects requests and processes them in batches."""
    global batch_queue

    while True:
        batch: List[BatchRequest] = []

        try:
            first_request = await batch_queue.get()
            batch.append(first_request)

            while not batch_queue.empty() and len(batch) < MAX_BATCH_SIZE:
                try:
                    batch.append(batch_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(batch) < MAX_BATCH_SIZE:
                timeout_ms = MIN_BATCH_WAIT_MS if len(batch) > 1 else BATCH_TIMEOUT_MS
                deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000.0)

                while len(batch) < MAX_BATCH_SIZE:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break

                    try:
                        request = await asyncio.wait_for(
                            batch_queue.get(),
                            timeout=remaining
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

            # Update queue depth metric
            if PROMETHEUS_AVAILABLE:
                QUEUE_DEPTH.set(batch_queue.qsize())

            await process_batch(batch)

        except asyncio.CancelledError:
            while not batch_queue.empty():
                try:
                    batch.append(batch_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if batch:
                await process_batch(batch)
            raise

        except Exception as e:
            logger.error("Batch processor error", error=str(e))
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


def get_beam_size_for_quality(quality: str) -> int:
    """Map quality level to beam size."""
    quality_map = {
        "fast": 1,
        "balanced": 4,
        "best": 8,
    }
    return quality_map.get(quality.lower(), BEAM_SIZE)


async def translate_text_async(text: str, source_lang: str, target_lang: str, beam_size: int = 1) -> str:
    """Queue a translation request and wait for result."""
    # Check circuit breaker
    if CIRCUIT_BREAKER_ENABLED and not circuit_breaker.can_execute():
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable (circuit breaker open)"
        )

    # Check queue size
    if batch_queue.qsize() >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail="Service overloaded, please retry later"
        )

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    request = BatchRequest(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        future=future,
        beam_size=beam_size,
    )

    await batch_queue.put(request)
    return await future


def translate_text(text: str, source_lang: str = "eng_Latn", target_lang: str = "ukr_Cyrl") -> str:
    """Translate text using NLLB model (sync version for CLI mode)."""
    source_tokens = tokenize_text(text, source_lang)

    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[target_lang]],
        beam_size=BEAM_SIZE,
        max_decoding_length=256,
        return_scores=False,
    )

    output_tokens = results[0].hypotheses[0]
    return decode_tokens(output_tokens, target_lang)


# ============== Sentence Segmentation ==============

def segment_sentences(text: str, language: str = "en") -> List[str]:
    """Segment text into sentences using pysbd if available, regex fallback otherwise."""
    if not PYSBD_AVAILABLE:
        # Regex fallback: split on sentence-ending punctuation followed by space
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    try:
        # Map NLLB language codes to pysbd languages
        lang_map = {
            "eng_Latn": "en",
            "fra_Latn": "fr",
            "deu_Latn": "de",
            "spa_Latn": "es",
            "ita_Latn": "it",
            "por_Latn": "pt",
            "rus_Cyrl": "ru",
            "zho_Hans": "zh",
            "jpn_Jpan": "ja",
        }
        pysbd_lang = lang_map.get(language, "en")
        segmenter = pysbd.Segmenter(language=pysbd_lang, clean=False)
        return segmenter.segment(text)
    except Exception:
        return [text]


# ============== API Endpoints ==============

@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate_endpoint(
    request: TranslateRequest,
    token_info: dict = Depends(get_api_token)
):
    """
    Translate text between languages.

    Quality levels:
    - **fast**: Beam size 1, lowest latency
    - **balanced**: Beam size 4, good quality/speed tradeoff
    - **best**: Beam size 8, highest quality
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    beam_size = get_beam_size_for_quality(request.quality)
    cache_key = get_cache_key(request.text, request.source_lang, request.target_lang, beam_size)

    start_time = time.time()
    cached = False

    try:
        # Check cache first
        cached_result = await get_from_cache(cache_key)
        if cached_result:
            cached = True
            result = cached_result
        else:
            # Check if text is too long and needs sentence-level translation
            token_count = await asyncio.get_event_loop().run_in_executor(
                tokenizer_pool, count_tokens, request.text, request.source_lang
            )

            if token_count > MAX_SAFE_TOKENS:
                # Segment into sentences and translate each separately
                sentences = segment_sentences(request.text, request.source_lang)
                logger.info(
                    "Segmenting long text",
                    token_count=token_count,
                    sentence_count=len(sentences),
                )
                translated_parts = []
                for sentence in sentences:
                    part = await translate_text_async(
                        sentence,
                        request.source_lang,
                        request.target_lang,
                        beam_size,
                    )
                    translated_parts.append(part)
                result = " ".join(translated_parts)
            else:
                result = await translate_text_async(
                    request.text,
                    request.source_lang,
                    request.target_lang,
                    beam_size,
                )
            # Store in cache
            await set_in_cache(cache_key, result)

        duration = time.time() - start_time

        if PROMETHEUS_AVAILABLE:
            TRANSLATION_REQUESTS.labels(
                status="success",
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                cached=str(cached)
            ).inc()
            TRANSLATION_LATENCY.labels(
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                quality=request.quality
            ).observe(duration)

        return TranslateResponse(
            translation=result,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            device=device_name,
            cached=cached,
            quality=request.quality
        )
    except HTTPException:
        raise
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            TRANSLATION_REQUESTS.labels(
                status="error",
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                cached="false"
            ).inc()
        logger.error("Translation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============== Health Endpoints ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Legacy health check endpoint (no authentication required)."""
    queue_size = batch_queue.qsize() if batch_queue else 0
    return HealthResponse(
        status="ok",
        device=device_name or "not initialized",
        batch_queue_size=queue_size,
        config={
            "beam_size": BEAM_SIZE,
            "batch_timeout_ms": BATCH_TIMEOUT_MS,
            "max_batch_size": MAX_BATCH_SIZE,
            "inter_threads": INTER_THREADS,
            "cache_enabled": CACHE_ENABLED,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "rate_limit_enabled": RATE_LIMIT_ENABLED,
        }
    )


@app.get("/health/live", response_model=LivenessResponse, tags=["Health"])
async def liveness():
    """Kubernetes liveness probe - checks if process is running."""
    return LivenessResponse(status="alive")


@app.get("/health/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness():
    """Kubernetes readiness probe - checks if service can accept traffic."""
    cache_connected = False
    if CACHE_ENABLED and redis_client:
        try:
            await redis_client.ping()
            cache_connected = True
        except Exception:
            pass

    queue_size = batch_queue.qsize() if batch_queue else 0

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if queue_size >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail="Queue overloaded")

    if CIRCUIT_BREAKER_ENABLED and circuit_breaker.state == "open":
        raise HTTPException(status_code=503, detail="Circuit breaker open")

    return ReadinessResponse(
        status="ready",
        model_loaded=model_loaded,
        cache_connected=cache_connected,
        queue_size=queue_size,
        circuit_breaker=circuit_breaker.state
    )


@app.get("/health/info", response_model=MetricsInfo, tags=["Health"])
async def metrics_info():
    """Get information about enabled features and optimizations."""
    cache_connected = False
    if CACHE_ENABLED and redis_client:
        try:
            await redis_client.ping()
            cache_connected = True
        except Exception:
            pass

    return MetricsInfo(
        uvloop_enabled=UVLOOP_ENABLED,
        orjson_enabled=ORJSON_ENABLED,
        cache_enabled=CACHE_ENABLED,
        cache_connected=cache_connected,
        rate_limit_enabled=RATE_LIMIT_ENABLED,
        circuit_breaker_enabled=CIRCUIT_BREAKER_ENABLED,
        circuit_breaker_state=circuit_breaker.state,
        structlog_enabled=STRUCTLOG_AVAILABLE,
        pysbd_enabled=PYSBD_AVAILABLE,
        prometheus_enabled=PROMETHEUS_AVAILABLE,
    )


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Prometheus not available")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/languages", tags=["Translation"])
async def languages():
    """
    Get list of all 202 supported language codes.

    NLLB-200 supports 202 languages. Language codes follow the format: xxx_Yyyy
    where xxx is the ISO 639-3 language code and Yyyy is the script.
    """
    return {
        # A
        "ace_Arab": "Acehnese (Arabic script)",
        "ace_Latn": "Acehnese (Latin script)",
        "acm_Arab": "Mesopotamian Arabic",
        "acq_Arab": "Ta'izzi-Adeni Arabic",
        "aeb_Arab": "Tunisian Arabic",
        "afr_Latn": "Afrikaans",
        "ajp_Arab": "South Levantine Arabic",
        "aka_Latn": "Akan",
        "als_Latn": "Albanian (Tosk)",
        "amh_Ethi": "Amharic",
        "apc_Arab": "North Levantine Arabic",
        "arb_Arab": "Modern Standard Arabic",
        "ars_Arab": "Najdi Arabic",
        "ary_Arab": "Moroccan Arabic",
        "arz_Arab": "Egyptian Arabic",
        "asm_Beng": "Assamese",
        "ast_Latn": "Asturian",
        "awa_Deva": "Awadhi",
        "ayr_Latn": "Central Aymara",
        "azb_Arab": "South Azerbaijani",
        "azj_Latn": "North Azerbaijani",
        # B
        "bak_Cyrl": "Bashkir",
        "bam_Latn": "Bambara",
        "ban_Latn": "Balinese",
        "bel_Cyrl": "Belarusian",
        "bem_Latn": "Bemba",
        "ben_Beng": "Bengali",
        "bho_Deva": "Bhojpuri",
        "bjn_Arab": "Banjar (Arabic script)",
        "bjn_Latn": "Banjar (Latin script)",
        "bod_Tibt": "Tibetan",
        "bos_Latn": "Bosnian",
        "bug_Latn": "Buginese",
        "bul_Cyrl": "Bulgarian",
        # C
        "cat_Latn": "Catalan",
        "ceb_Latn": "Cebuano",
        "ces_Latn": "Czech",
        "cjk_Latn": "Chokwe",
        "ckb_Arab": "Central Kurdish",
        "crh_Latn": "Crimean Tatar",
        "cym_Latn": "Welsh",
        # D
        "dan_Latn": "Danish",
        "deu_Latn": "German",
        "dik_Latn": "Southwestern Dinka",
        "dyu_Latn": "Dyula",
        "dzo_Tibt": "Dzongkha",
        # E
        "ell_Grek": "Greek",
        "eng_Latn": "English",
        "epo_Latn": "Esperanto",
        "est_Latn": "Estonian",
        "eus_Latn": "Basque",
        "ewe_Latn": "Ewe",
        # F
        "fao_Latn": "Faroese",
        "fij_Latn": "Fijian",
        "fin_Latn": "Finnish",
        "fon_Latn": "Fon",
        "fra_Latn": "French",
        "fur_Latn": "Friulian",
        "fuv_Latn": "Nigerian Fulfulde",
        # G
        "gaz_Latn": "West Central Oromo",
        "gla_Latn": "Scottish Gaelic",
        "gle_Latn": "Irish",
        "glg_Latn": "Galician",
        "grn_Latn": "Guarani",
        "guj_Gujr": "Gujarati",
        # H
        "hat_Latn": "Haitian Creole",
        "hau_Latn": "Hausa",
        "heb_Hebr": "Hebrew",
        "hin_Deva": "Hindi",
        "hne_Deva": "Chhattisgarhi",
        "hrv_Latn": "Croatian",
        "hun_Latn": "Hungarian",
        "hye_Armn": "Armenian",
        # I
        "ibo_Latn": "Igbo",
        "ilo_Latn": "Ilocano",
        "ind_Latn": "Indonesian",
        "isl_Latn": "Icelandic",
        "ita_Latn": "Italian",
        # J
        "jav_Latn": "Javanese",
        "jpn_Jpan": "Japanese",
        # K
        "kab_Latn": "Kabyle",
        "kac_Latn": "Jingpho",
        "kam_Latn": "Kamba",
        "kan_Knda": "Kannada",
        "kas_Arab": "Kashmiri (Arabic script)",
        "kas_Deva": "Kashmiri (Devanagari script)",
        "kat_Geor": "Georgian",
        "kaz_Cyrl": "Kazakh",
        "kbp_Latn": "Kabiye",
        "kea_Latn": "Kabuverdianu",
        "khk_Cyrl": "Halh Mongolian",
        "khm_Khmr": "Khmer",
        "kik_Latn": "Kikuyu",
        "kin_Latn": "Kinyarwanda",
        "kir_Cyrl": "Kyrgyz",
        "kmb_Latn": "Kimbundu",
        "kmr_Latn": "Northern Kurdish",
        "knc_Arab": "Central Kanuri (Arabic script)",
        "knc_Latn": "Central Kanuri (Latin script)",
        "kon_Latn": "Kikongo",
        "kor_Hang": "Korean",
        # L
        "lao_Laoo": "Lao",
        "lij_Latn": "Ligurian",
        "lim_Latn": "Limburgish",
        "lin_Latn": "Lingala",
        "lit_Latn": "Lithuanian",
        "lmo_Latn": "Lombard",
        "ltg_Latn": "Latgalian",
        "ltz_Latn": "Luxembourgish",
        "lua_Latn": "Luba-Kasai",
        "lug_Latn": "Ganda",
        "luo_Latn": "Luo",
        "lus_Latn": "Mizo",
        "lvs_Latn": "Standard Latvian",
        # M
        "mag_Deva": "Magahi",
        "mai_Deva": "Maithili",
        "mal_Mlym": "Malayalam",
        "mar_Deva": "Marathi",
        "min_Latn": "Minangkabau",
        "mkd_Cyrl": "Macedonian",
        "mlt_Latn": "Maltese",
        "mni_Beng": "Meitei",
        "mos_Latn": "Mossi",
        "mri_Latn": "Maori",
        "mya_Mymr": "Burmese",
        # N
        "nld_Latn": "Dutch",
        "nno_Latn": "Norwegian Nynorsk",
        "nob_Latn": "Norwegian Bokmal",
        "npi_Deva": "Nepali",
        "nso_Latn": "Northern Sotho",
        "nus_Latn": "Nuer",
        "nya_Latn": "Chichewa",
        # O
        "oci_Latn": "Occitan",
        "ory_Orya": "Odia",
        # P
        "pag_Latn": "Pangasinan",
        "pan_Guru": "Punjabi",
        "pap_Latn": "Papiamento",
        "pbt_Arab": "Southern Pashto",
        "pes_Arab": "Western Persian",
        "plt_Latn": "Plateau Malagasy",
        "pol_Latn": "Polish",
        "por_Latn": "Portuguese",
        "prs_Arab": "Dari",
        # Q
        "quy_Latn": "Ayacucho Quechua",
        # R
        "ron_Latn": "Romanian",
        "run_Latn": "Rundi",
        "rus_Cyrl": "Russian",
        # S
        "sag_Latn": "Sango",
        "san_Deva": "Sanskrit",
        "sat_Beng": "Santali",
        "scn_Latn": "Sicilian",
        "shn_Mymr": "Shan",
        "sin_Sinh": "Sinhala",
        "slk_Latn": "Slovak",
        "slv_Latn": "Slovenian",
        "smo_Latn": "Samoan",
        "sna_Latn": "Shona",
        "snd_Arab": "Sindhi",
        "som_Latn": "Somali",
        "sot_Latn": "Southern Sotho",
        "spa_Latn": "Spanish",
        "srd_Latn": "Sardinian",
        "srp_Cyrl": "Serbian",
        "ssw_Latn": "Swati",
        "sun_Latn": "Sundanese",
        "swe_Latn": "Swedish",
        "swh_Latn": "Swahili",
        "szl_Latn": "Silesian",
        # T
        "tam_Taml": "Tamil",
        "taq_Latn": "Tamasheq (Latin script)",
        "taq_Tfng": "Tamasheq (Tifinagh script)",
        "tat_Cyrl": "Tatar",
        "tel_Telu": "Telugu",
        "tgk_Cyrl": "Tajik",
        "tgl_Latn": "Tagalog",
        "tha_Thai": "Thai",
        "tir_Ethi": "Tigrinya",
        "tpi_Latn": "Tok Pisin",
        "tsn_Latn": "Tswana",
        "tso_Latn": "Tsonga",
        "tuk_Latn": "Turkmen",
        "tum_Latn": "Tumbuka",
        "tur_Latn": "Turkish",
        "twi_Latn": "Twi",
        "tzm_Tfng": "Central Atlas Tamazight",
        # U
        "uig_Arab": "Uyghur",
        "ukr_Cyrl": "Ukrainian",
        "umb_Latn": "Umbundu",
        "urd_Arab": "Urdu",
        "uzn_Latn": "Northern Uzbek",
        # V
        "vec_Latn": "Venetian",
        "vie_Latn": "Vietnamese",
        # W
        "war_Latn": "Waray",
        "wol_Latn": "Wolof",
        # X
        "xho_Latn": "Xhosa",
        # Y
        "ydd_Hebr": "Eastern Yiddish",
        "yor_Latn": "Yoruba",
        "yue_Hant": "Cantonese",
        # Z
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "zsm_Latn": "Standard Malay",
        "zul_Latn": "Zulu",
    }


# ============== Admin Endpoints ==============

@app.post("/admin/tokens", response_model=TokenResponse, tags=["Admin"])
async def create_token(
    request: TokenCreate,
    _: bool = Depends(get_admin_token)
):
    """Create a new API token (admin only)."""
    token = secrets.token_urlsafe(32)

    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO tokens (name, token, description) VALUES (?, ?, ?)",
                (request.name, token, request.description)
            )
            conn.commit()
            token_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Token name already exists")

        cursor.execute("SELECT * FROM tokens WHERE id = ?", (token_id,))
        row = cursor.fetchone()

    logger.info("Token created", name=request.name, token_id=token_id)

    return TokenResponse(
        id=row["id"],
        name=row["name"],
        token=row["token"],
        description=row["description"],
        created_at=row["created_at"],
        is_active=bool(row["is_active"])
    )


@app.get("/admin/tokens", response_model=List[TokenInfo], tags=["Admin"])
async def list_tokens(_: bool = Depends(get_admin_token)):
    """List all API tokens (admin only)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, created_at, is_active, usage_count FROM tokens")
        rows = cursor.fetchall()

    return [
        TokenInfo(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
            usage_count=row["usage_count"]
        )
        for row in rows
    ]


@app.delete("/admin/tokens/{token_id}", tags=["Admin"])
async def delete_token(token_id: int, _: bool = Depends(get_admin_token)):
    """Deactivate an API token (admin only)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tokens SET is_active = 0 WHERE id = ?",
            (token_id,)
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Token not found")

    logger.info("Token deactivated", token_id=token_id)
    return {"message": "Token deactivated"}


@app.post("/admin/tokens/{token_id}/activate", tags=["Admin"])
async def activate_token(token_id: int, _: bool = Depends(get_admin_token)):
    """Reactivate a deactivated token (admin only)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tokens SET is_active = 1 WHERE id = ?",
            (token_id,)
        )
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Token not found")

    logger.info("Token activated", token_id=token_id)
    return {"message": "Token activated"}


# ============== Circuit Breaker Admin Endpoints ==============

@app.post("/admin/circuit-breaker/test", tags=["Admin"])
async def test_circuit_breaker(_: bool = Depends(get_admin_token)):
    """
    Test circuit breaker by simulating failures (admin only).
    This will open the circuit breaker after threshold failures.
    """
    # Simulate failures to trigger circuit breaker
    for i in range(CIRCUIT_BREAKER_THRESHOLD):
        circuit_breaker.record_failure()

    return {
        "message": f"Simulated {CIRCUIT_BREAKER_THRESHOLD} failures",
        "state": circuit_breaker.state,
        "failures": circuit_breaker.failures
    }


@app.post("/admin/circuit-breaker/reset", tags=["Admin"])
async def reset_circuit_breaker(_: bool = Depends(get_admin_token)):
    """Reset circuit breaker to closed state (admin only)."""
    circuit_breaker.record_success()
    return {
        "message": "Circuit breaker reset",
        "state": circuit_breaker.state,
        "failures": circuit_breaker.failures
    }


@app.get("/admin/circuit-breaker/status", tags=["Admin"])
async def circuit_breaker_status(_: bool = Depends(get_admin_token)):
    """Get circuit breaker status (admin only)."""
    return {
        "state": circuit_breaker.state,
        "failures": circuit_breaker.failures,
        "threshold": CIRCUIT_BREAKER_THRESHOLD,
        "timeout_seconds": CIRCUIT_BREAKER_TIMEOUT,
        "enabled": CIRCUIT_BREAKER_ENABLED
    }


# ============== Cache Admin Endpoints ==============

@app.delete("/admin/cache", tags=["Admin"])
async def clear_cache(_: bool = Depends(get_admin_token)):
    """Clear translation cache (admin only)."""
    if not CACHE_ENABLED or redis_client is None:
        raise HTTPException(status_code=400, detail="Cache not enabled")

    try:
        # Delete all keys with nllb: prefix
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match="nllb:*", count=100)
            if keys:
                deleted += await redis_client.delete(*keys)
            if cursor == 0:
                break

        logger.info("Cache cleared", deleted_keys=deleted)
        return {"message": f"Cleared {deleted} cached translations"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/cache/stats", tags=["Admin"])
async def cache_stats(_: bool = Depends(get_admin_token)):
    """Get cache statistics (admin only)."""
    if not CACHE_ENABLED or redis_client is None:
        raise HTTPException(status_code=400, detail="Cache not enabled")

    try:
        info = await redis_client.info("memory")
        dbsize = await redis_client.dbsize()

        return {
            "keys": dbsize,
            "used_memory": info.get("used_memory_human", "unknown"),
            "used_memory_peak": info.get("used_memory_peak_human", "unknown"),
            "ttl_seconds": CACHE_TTL_SECONDS,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Startup/Shutdown ==============

async def init_redis():
    """Initialize Redis/Dragonfly connection."""
    global redis_client

    if not CACHE_ENABLED or not REDIS_AVAILABLE:
        logger.info("Cache disabled or redis library not available")
        return

    try:
        redis_client = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=False,
        )
        await redis_client.ping()
        logger.info("Connected to cache", url=REDIS_URL)
    except Exception as e:
        logger.warning("Failed to connect to cache", error=str(e))
        redis_client = None


def warm_up_tokenizer_pool():
    """Pre-initialize tokenizers in thread pool."""
    import concurrent.futures

    def init_worker_tokenizer():
        get_tokenizer()
        return True

    futures = [tokenizer_pool.submit(init_worker_tokenizer) for _ in range(4)]
    concurrent.futures.wait(futures)
    logger.info("Tokenizer pool warmed up")


def warm_up_gpu():
    """Run a warm-up translation to initialize GPU kernels."""
    try:
        result = translate_text("Hello", "eng_Latn", "fra_Latn")
        logger.info("GPU warm-up complete", test_result=result)
    except Exception as e:
        logger.error("GPU warm-up failed", error=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize model, database, cache, and batch processor on startup."""
    global batch_queue, batch_processor_task

    logger.info("Starting NLLB Translation API",
                version="3.0.0",
                uvloop=UVLOOP_ENABLED,
                orjson=ORJSON_ENABLED,
                cache=CACHE_ENABLED)

    init_db()
    init_model()
    warm_up_tokenizer_pool()
    warm_up_gpu()

    await init_redis()

    batch_queue = asyncio.Queue()
    batch_processor_task = asyncio.create_task(batch_processor())
    logger.info("Batch processor started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global batch_processor_task, redis_client

    if batch_processor_task:
        batch_processor_task.cancel()
        try:
            await batch_processor_task
        except asyncio.CancelledError:
            pass

    if redis_client:
        await redis_client.close()

    # Shutdown thread pool
    tokenizer_pool.shutdown(wait=False)

    logger.info("Service shutdown complete")


# ============== Main ==============

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # CLI mode
        init_db()
        init_model()

        text = " ".join(sys.argv[1:])
        source_lang = os.environ.get("SOURCE_LANG", "eng_Latn")
        target_lang = os.environ.get("TARGET_LANG", "ukr_Cyrl")

        print(f"\nTranslating: {text}")
        print(f"From: {source_lang} -> To: {target_lang}")

        result = translate_text(text, source_lang, target_lang)
        print(f"\nResult: {result}")
        print(f"Device: {device_name}")
    else:
        # Server mode
        logger.info("Starting NLLB Translation API server",
                   host="0.0.0.0",
                   port=5000,
                   uvloop=UVLOOP_ENABLED,
                   orjson=ORJSON_ENABLED)
        print(f"Swagger UI: http://0.0.0.0:5000/docs")
        print(f"ReDoc: http://0.0.0.0:5000/redoc")
        print(f"Metrics: http://0.0.0.0:5000/metrics")
        uvicorn.run(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
