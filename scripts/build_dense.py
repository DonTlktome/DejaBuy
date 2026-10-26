"""Build the dense retrieval route: INT8-quantised bge-small ONNX encoder + catalogue embeddings.

    python3 -m scripts.build_dense            # needs data/onnx_model/{model.onnx, tokenizer.json}

Steps
  1. dynamic INT8 quantisation of the official ONNX export (weights to int8, activations
     quantised on the fly) → data/onnx_model/model_int8.onnx
  2. encode every catalogue product (title + first features, no query instruction: bge puts the
     instruction on the query side only) in batches with the INT8 model, CLS pooling, L2-normalised
     → data/embeddings.npz (asins, float16 matrix)
  3. report encode throughput, the fp32↔int8 cosine agreement on a sample, and file sizes

Then `USE_DENSE=1 DENSE_BACKEND=onnx EMBED_ONNX_MODEL=model_int8.onnx` turns the route on.
Runtime dependencies for the route: numpy, onnxruntime, tokenizers (the default path needs none).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def product_text(p: dict) -> str:
    features = " ".join(str(f) for f in (p.get("features") or [])[:3])
    return f"{p.get('title') or ''}. {features}".strip()[:512]


def make_session(model: Path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def encode(session, tokenizer, texts: list[str], max_len: int = 128) -> np.ndarray:
    tokenizer.enable_truncation(max_len)
    tokenizer.enable_padding()
    enc = tokenizer.encode_batch(texts)
    inputs = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
              "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
    if "token_type_ids" in {i.name for i in session.get_inputs()}:
        inputs["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
    cls = session.run(None, inputs)[0][:, 0, :]
    return cls / np.linalg.norm(cls, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(ROOT / "data" / "onnx_model"))
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "embeddings.npz"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="encode only the first N products (smoke test)")
    args = parser.parse_args()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from tokenizers import Tokenizer

    model_dir = Path(args.model_dir)
    fp32, int8 = model_dir / "model.onnx", model_dir / "model_int8.onnx"
    if not int8.exists():
        t0 = time.time()
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
        print(f"quantised in {time.time() - t0:.0f}s: {fp32.stat().st_size / 1e6:.0f} MB → {int8.stat().st_size / 1e6:.0f} MB")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    s_int8 = make_session(int8)

    products = [json.loads(line) for line in Path(args.catalog).open(encoding="utf-8")]
    if args.limit:
        products = products[: args.limit]
    texts = [product_text(p) for p in products]

    # agreement check on a sample, before the long run
    s_fp32 = make_session(fp32)
    sample = texts[:256]
    a = np.vstack([encode(s_fp32, tokenizer, sample[i:i + 64]) for i in range(0, len(sample), 64)])
    b = np.vstack([encode(s_int8, tokenizer, sample[i:i + 64]) for i in range(0, len(sample), 64)])
    agree = float(np.mean(np.sum(a * b, axis=1)))
    top1 = float(np.mean(np.argmax(a @ a.T - np.eye(len(a)) * 2, 1) == np.argmax(b @ a.T - np.eye(len(a)) * 2, 1)))
    print(f"fp32 vs int8 on {len(sample)} products: mean cosine {agree:.4f}, nearest-neighbour agreement {top1:.3f}")
    del s_fp32

    t0 = time.time()
    chunks = []
    for i in range(0, len(texts), args.batch):
        chunks.append(encode(s_int8, tokenizer, texts[i:i + args.batch]).astype(np.float16))
        if (i // args.batch) % 100 == 0:
            done = i + len(chunks[-1])
            print(f"  {done:,}/{len(texts):,} ({done / (time.time() - t0):.0f} products/s)", flush=True)
    matrix = np.vstack(chunks)
    np.savez(args.out, asins=np.array([p["parent_asin"] for p in products]), matrix=matrix)
    print(f"{len(texts):,} products in {time.time() - t0:.0f}s → {args.out} ({Path(args.out).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
