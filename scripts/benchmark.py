#!/usr/bin/env python3
"""Comprehensive benchmark for NLLB translation service."""

import time
import statistics
import concurrent.futures
import requests
import json
import sys
import os

# Configuration
BASE_URL = os.environ.get("API_URL", "http://localhost:5000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin-secret-change-me")

# Test sentences of varying lengths
TEST_SENTENCES = {
    "short": "Hello, how are you?",
    "medium": "The quick brown fox jumps over the lazy dog. This is a test sentence to measure translation performance.",
    "long": """Artificial intelligence has revolutionized the way we approach complex problems in various fields.
    From healthcare diagnostics to autonomous vehicles, machine learning algorithms are transforming industries
    and creating new possibilities that were once thought impossible. The advancement of natural language processing
    has enabled computers to understand and generate human language with remarkable accuracy.""",
    "very_long": """The history of artificial intelligence dates back to the mid-20th century when pioneers like
    Alan Turing began questioning whether machines could think. This philosophical inquiry led to the development
    of early computing machines and eventually to the field we now know as AI. Over the decades, AI has gone through
    several periods of intense optimism followed by disappointment, known as AI winters. However, the recent
    explosion of data availability and computational power has ushered in a new era of machine learning breakthroughs.
    Deep learning, a subset of machine learning, has proven particularly effective at tasks involving pattern
    recognition, including image classification, speech recognition, and natural language understanding.
    These advances have practical applications in fields ranging from medicine and finance to entertainment and
    transportation. As we continue to push the boundaries of what machines can do, important ethical considerations
    arise regarding privacy, bias, and the impact on employment."""
}

def get_or_create_token():
    """Get existing token or create a new one for benchmarking."""
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

    # Create new benchmark token
    resp = requests.post(
        f"{BASE_URL}/admin/tokens",
        headers=headers,
        json={"name": f"benchmark-{int(time.time())}", "description": "Benchmark token"}
    )
    if resp.status_code == 200:
        return resp.json()["token"]
    else:
        print(f"Failed to create token: {resp.text}")
        sys.exit(1)

def translate(text, token, source_lang="eng_Latn", target_lang="ukr_Cyrl"):
    """Send translation request and return timing info."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }

    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/translate", headers=headers, json=payload, timeout=60)
    elapsed = time.perf_counter() - start

    if resp.status_code != 200:
        return None, elapsed, resp.text

    return resp.json(), elapsed, None

def benchmark_single_requests(token, iterations=10):
    """Benchmark sequential single requests."""
    print("\n" + "="*60)
    print("SEQUENTIAL SINGLE REQUEST BENCHMARK")
    print("="*60)

    results = {}

    for name, text in TEST_SENTENCES.items():
        times = []
        word_count = len(text.split())

        print(f"\nTesting '{name}' ({word_count} words)...")

        for i in range(iterations):
            result, elapsed, error = translate(text, token)
            if error:
                print(f"  Error on iteration {i}: {error}")
                continue
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.3f}s")

        if times:
            results[name] = {
                "word_count": word_count,
                "char_count": len(text),
                "iterations": len(times),
                "min": min(times),
                "max": max(times),
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "words_per_second": word_count / statistics.mean(times)
            }

    print("\n" + "-"*60)
    print("SEQUENTIAL RESULTS SUMMARY:")
    print("-"*60)
    print(f"{'Type':<12} {'Words':<8} {'Mean':<10} {'Min':<10} {'Max':<10} {'WPS':<10}")
    for name, stats in results.items():
        print(f"{name:<12} {stats['word_count']:<8} {stats['mean']:.3f}s    {stats['min']:.3f}s    {stats['max']:.3f}s    {stats['words_per_second']:.1f}")

    return results

def benchmark_concurrent_requests(token, concurrent_users=5, requests_per_user=5):
    """Benchmark concurrent requests to test threading/async behavior."""
    print("\n" + "="*60)
    print(f"CONCURRENT REQUEST BENCHMARK ({concurrent_users} users, {requests_per_user} req each)")
    print("="*60)

    text = TEST_SENTENCES["medium"]
    all_times = []
    errors = 0

    def worker(user_id):
        times = []
        for i in range(requests_per_user):
            result, elapsed, error = translate(text, token)
            if error:
                return None, error
            times.append(elapsed)
        return times, None

    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(worker, i) for i in range(concurrent_users)]

        for future in concurrent.futures.as_completed(futures):
            times, error = future.result()
            if error:
                errors += 1
                print(f"  Worker error: {error}")
            elif times:
                all_times.extend(times)

    total_time = time.perf_counter() - start_time
    total_requests = concurrent_users * requests_per_user

    print(f"\nTotal requests: {total_requests}")
    print(f"Successful: {len(all_times)}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(all_times) / total_time:.2f} requests/second")

    if all_times:
        print(f"\nPer-request latency:")
        print(f"  Min: {min(all_times):.3f}s")
        print(f"  Max: {max(all_times):.3f}s")
        print(f"  Mean: {statistics.mean(all_times):.3f}s")
        print(f"  Median: {statistics.median(all_times):.3f}s")
        p95_idx = min(int(len(all_times)*0.95), len(all_times)-1)
        print(f"  P95: {sorted(all_times)[p95_idx]:.3f}s")

    return {
        "total_requests": total_requests,
        "successful": len(all_times),
        "errors": errors,
        "total_time": total_time,
        "throughput": len(all_times) / total_time if all_times else 0,
        "latencies": all_times
    }

def benchmark_high_concurrency(token, total_requests=50, concurrent_workers=10):
    """Benchmark high concurrency scenario."""
    print("\n" + "="*60)
    print(f"HIGH CONCURRENCY BENCHMARK ({total_requests} requests, {concurrent_workers} workers)")
    print("="*60)

    sentences = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can process complex data patterns.",
    ]

    all_times = []
    errors = 0

    def single_translate(idx):
        text = sentences[idx % len(sentences)]
        result, elapsed, error = translate(text, token)
        return elapsed if not error else None

    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = [executor.submit(single_translate, i) for i in range(total_requests)]

        for future in concurrent.futures.as_completed(futures):
            elapsed = future.result()
            if elapsed:
                all_times.append(elapsed)
            else:
                errors += 1

    total_time = time.perf_counter() - start_time

    print(f"\nTotal requests: {total_requests}")
    print(f"Successful: {len(all_times)}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(all_times) / total_time:.2f} requests/second")

    if all_times:
        print(f"\nPer-request latency:")
        print(f"  Min: {min(all_times):.3f}s")
        print(f"  Max: {max(all_times):.3f}s")
        print(f"  Mean: {statistics.mean(all_times):.3f}s")
        sorted_times = sorted(all_times)
        print(f"  P50: {sorted_times[int(len(all_times)*0.50)]:.3f}s")
        print(f"  P95: {sorted_times[min(int(len(all_times)*0.95), len(all_times)-1)]:.3f}s")
        print(f"  P99: {sorted_times[min(int(len(all_times)*0.99), len(all_times)-1)]:.3f}s")

    return {
        "total_requests": total_requests,
        "successful": len(all_times),
        "throughput": len(all_times) / total_time if all_times else 0,
        "mean_latency": statistics.mean(all_times) if all_times else 0
    }

def benchmark_batch_efficiency(token):
    """Test batch processing efficiency."""
    print("\n" + "="*60)
    print("BATCH EFFICIENCY TEST")
    print("="*60)

    text = TEST_SENTENCES["short"]
    batch_sizes = [1, 2, 4, 8, 16]

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(translate, text, token) for _ in range(batch_size)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.perf_counter() - start

        successful = sum(1 for r in results if r[0] is not None)
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Per-item avg: {elapsed/batch_size:.3f}s")
        print(f"  Effective throughput: {successful/elapsed:.2f} req/s")
        print(f"  Successful: {successful}/{batch_size}")

def check_health():
    """Check API health and device status."""
    print("="*60)
    print("SYSTEM CHECK")
    print("="*60)

    resp = requests.get(f"{BASE_URL}/health")
    if resp.status_code == 200:
        health = resp.json()
        print(f"Status: {health['status']}")
        print(f"Device: {health['device']}")
        print(f"Queue size: {health.get('batch_queue_size', 'N/A')}")
        print(f"Config: {json.dumps(health.get('config', {}), indent=2)}")
        return health['device']
    else:
        print(f"Health check failed: {resp.status_code}")
        return None

def main():
    print("\n" + "#"*60)
    print("# NLLB TRANSLATION SERVICE BENCHMARK v2.0")
    print("#"*60)

    device = check_health()
    if not device:
        print("Service not available!")
        sys.exit(1)

    token = get_or_create_token()
    print(f"\nUsing token: {token[:20]}...")

    # Run benchmarks
    sequential_results = benchmark_single_requests(token, iterations=5)
    concurrent_results = benchmark_concurrent_requests(token, concurrent_users=5, requests_per_user=3)
    high_conc_results = benchmark_high_concurrency(token, total_requests=50, concurrent_workers=10)
    benchmark_batch_efficiency(token)

    # Final summary
    print("\n" + "#"*60)
    print("# FINAL SUMMARY")
    print("#"*60)
    print(f"Device: {device}")
    print(f"\nSequential Performance:")
    for name, stats in sequential_results.items():
        print(f"  {name}: {stats['mean']:.3f}s avg ({stats['words_per_second']:.1f} words/sec)")
    print(f"\nConcurrent Performance (5 users):")
    print(f"  Throughput: {concurrent_results['throughput']:.2f} req/s")
    print(f"\nHigh Concurrency (10 workers):")
    print(f"  Throughput: {high_conc_results['throughput']:.2f} req/s")
    print(f"  Mean latency: {high_conc_results['mean_latency']:.3f}s")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
