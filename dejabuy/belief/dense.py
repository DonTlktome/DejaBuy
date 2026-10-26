"""Optional dense retrieval route (sentence embeddings over the catalogue).

Disabled unless USE_DENSE=1 and the assets exist; any failure degrades to lexical retrieval."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dejabuy import config

# bge 查询侧指令前缀（spec §1-④）：查询编码时必须加，商品侧（预计算）不加
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _warn(reason: str) -> None:
    """USE_DENSE=1 但资产缺失时显式降级——防"以为开了稠密路其实跑的纯 BM25"（留言板 T-004）。"""
    print(f"[M2] USE_DENSE=1 但稠密路未生效（{reason}），本场为纯 BM25。", file=sys.stderr)


def _torch_encoder():
    """SentenceTransformer/torch 后端（现状，内存峰值 +660MB）。"""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # 断网红线：只读本地缓存，绝不在运行时尝试下载（预计算脚本已拉过权重）
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")

    def encode(text: str):
        return model.encode(
            [QUERY_INSTRUCTION + text], convert_to_numpy=True, normalize_embeddings=True,
        )[0].astype(np.float32)

    return encode


def _onnx_encoder():
    """onnx-runtime 后端（实验 O10）：tokenizers + onnxruntime，不依赖 torch。

    资产：data/onnx_model/{model.onnx, tokenizer.json}（bge-small-en-v1.5 官方 onnx）。
    CLS pooling（与 ST 侧 bge 的 1_Pooling 配置一致），输出 L2 归一化 float32。
    """
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(getattr(config, "EMBED_ONNX_DIR", "data/onnx_model"))
    model_file = getattr(config, "EMBED_ONNX_MODEL", "model.onnx")
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(
        str(model_dir / model_file), providers=["CPUExecutionProvider"]
    )
    input_names = {i.name for i in session.get_inputs()}

    def encode(text: str):
        enc = tok.encode(QUERY_INSTRUCTION + text)
        inputs = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        vec = session.run(None, inputs)[0][0][0].astype(np.float32)  # CLS
        return vec / np.linalg.norm(vec)

    return encode


class DenseIndex:
    def __init__(self, asins, matrix, encode_fn) -> None:
        self.asins = asins          # (N,) str 数组，与 matrix 行一一对应
        self.matrix = matrix        # (N, 384) float32，已归一化
        self._encode = encode_fn    # str -> 归一化 float32 向量
        self._vec_cache: dict[str, object] = {}  # 查询文本 → 向量（约束问干后逐轮不变，省重复编码）

    @classmethod
    def from_env(cls) -> "DenseIndex | None":
        """按 spec §1-⑦ 做全链路降级：任何资产缺失都返回 None，绝不抛给上层。"""
        if not config.USE_DENSE:
            return None  # 用户没开 = 静默；开了却降级才需要警告
        try:
            path = Path(config.EMBEDDINGS_PATH)
            if not path.exists():
                _warn(f"npz 缺失：{path}")
                return None
            import numpy as np

            data = np.load(path, allow_pickle=False)
            # 实验 O10：DENSE_BACKEND=onnx 走轻量运行时；onnx 资产缺失回退 torch，再缺降级
            if getattr(config, "DENSE_BACKEND", "torch") == "onnx":
                try:
                    encode_fn = _onnx_encoder()
                except Exception as exc:
                    _warn(f"onnx 后端不可用（{type(exc).__name__}: {exc}），回退 torch")
                    encode_fn = _torch_encoder()
            else:
                encode_fn = _torch_encoder()
            matrix = data["matrix"]
            if matrix.dtype != np.float32:          # stored as float16 on disk (half the size), searched in float32
                matrix = matrix.astype(np.float32)
            return cls(asins=data["asins"], matrix=matrix, encode_fn=encode_fn)
        except Exception as exc:
            _warn(f"{type(exc).__name__}: {exc}")
            return None

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """返回 [(parent_asin, cosine_sim)]，按相似度降序，最多 top_k 条。"""
        if not query_text.strip():
            return []
        import numpy as np

        vector = self._vec_cache.get(query_text)
        if vector is None:
            vector = self._encode(query_text)
            self._vec_cache[query_text] = vector
        sims = self.matrix @ vector
        top_k = min(top_k, len(self.asins))
        # argpartition 拿 top-k 再局部排序，避免全量 sort
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(str(self.asins[i]), float(sims[i])) for i in idx]
