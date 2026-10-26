"""Learn the semantic-memory prior weights from real purchase sequences.

    python3 -m scripts.train_memory            # → data/memory_weights.json

Model: a conditional logit (multinomial choice) over the target's category pool,

    P(target = i | user) ∝ exp(base_i + w · f(i, profile_user))

where `base_i` is the cold-start purchase prior the agent already uses (review volume + has-price,
same constants as belief/evidence.py) and `f` are the memory affinity features
(memory/semantic.py: FEATURES). Only `w` is learned, on the train split, with L2; negatives are
subsampled uniformly per session with the standard log((N-1)/k) correction. The dev split picks
L2 and the half-life; the script then reports single-shot ranking on dev and test against the
cold-start prior and against a shuffled-memory control (each user scored with another user's
history), overall and per stratum.

The runtime only needs the weights (a dot product, standard library); numpy / scipy are
training-time dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dejabuy import config  # noqa: E402
from dejabuy.memory.items import catalogue_item  # noqa: E402
from dejabuy.memory.semantic import FEATURES  # noqa: E402
from dejabuy.memory.store import MemoryStore  # noqa: E402


def base_prior(item: dict) -> float:
    s = config.POP_WEIGHT * math.log10(1 + item.get("rating_number", 0)) / 5.0
    if item.get("price") is not None:
        s += config.HAS_PRICE_WEIGHT
    return s


def load_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


class Data:
    def __init__(self, bench: Path, catalog_path: Path, half_life: float) -> None:
        self.items: dict[str, dict] = {}
        self.pools: dict[str, list[str]] = defaultdict(list)
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = catalogue_item(json.loads(line))
                self.items[item["parent_asin"]] = item
                self.pools[item["coarse"]].append(item["parent_asin"])
        self.base = {a: base_prior(it) for a, it in self.items.items()}
        self.store = MemoryStore.load(bench, half_life_days=half_life)
        self.splits = {name: load_split(bench / f"{name}.jsonl")
                       for name in ("train", "dev", "test", "test_loyal") if (bench / f"{name}.jsonl").exists()}

    def profile(self, sample: dict, other: dict | None = None):
        src = other or sample
        return self.store.profile(src["user_profile"]["user_id"], int(src["user_profile"]["as_of"]))


def build_matrix(data: Data, samples: list[dict], negatives: int, rng: random.Random):
    """Per session: [target] + sampled negatives → (X, base, offsets, session index)."""
    xs, bs, corr, sess = [], [], [], []
    for k, s in enumerate(samples):
        prof = data.profile(s)
        if prof is None:
            continue
        target = s["ground_truth"]["parent_asin"]
        pool = [a for a in data.pools[data.items[target]["coarse"]] if a != target]
        if not pool:
            continue
        neg = pool if len(pool) <= negatives else rng.sample(pool, negatives)
        c = math.log(len(pool) / len(neg))
        for j, a in enumerate([target] + neg):
            xs.append(prof.features(data.items[a]))
            bs.append(data.base[a])
            corr.append(0.0 if j == 0 else c)
            sess.append(k)
    return (np.asarray(xs, dtype=np.float64), np.asarray(bs), np.asarray(corr), np.asarray(sess))


def fit(X, base, corr, sess, l2: float) -> np.ndarray:
    """Conditional logit by L-BFGS; the first row of each session is the chosen item."""
    starts = np.flatnonzero(np.r_[True, sess[1:] != sess[:-1]])
    n_sessions = len(starts)

    def loss(w):
        z = base + corr + X @ w
        zmax = np.maximum.reduceat(z, starts)
        e = np.exp(z - np.repeat(zmax, np.diff(np.r_[starts, len(z)])))
        denom = np.add.reduceat(e, starts)
        logp = z[starts] - zmax - np.log(denom)
        p = e / np.repeat(denom, np.diff(np.r_[starts, len(z)]))
        grad = -(X[starts].sum(axis=0) - np.add.reduceat(p[:, None] * X, starts).sum(axis=0))
        return -logp.sum() / n_sessions + l2 * w @ w, grad / n_sessions + 2 * l2 * w

    res = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return res.x


class FullPool:
    """Every session's whole category pool as one stacked matrix, for exact vectorised ranking."""

    def __init__(self, data: Data, samples: list[dict], shuffle: bool = False) -> None:
        rng = random.Random(11)
        order = list(range(len(samples)))
        rng.shuffle(order)
        xs, bs, seg, tgt_rows, strata = [], [], [], [], []
        row = 0
        for i, s in enumerate(samples):
            other = samples[order[i]] if shuffle else None
            prof = data.profile(s, other)
            target = s["ground_truth"]["parent_asin"]
            pool = data.pools[data.items[target]["coarse"]]
            tgt_rows.append(row + pool.index(target))
            for a in pool:
                xs.append(prof.features(data.items[a]) if prof is not None else [0.0] * len(FEATURES))
                bs.append(data.base[a])
                seg.append(i)
            row += len(pool)
            strata.append(s.get("stratum", "other"))
        self.X = np.asarray(xs)
        self.base = np.asarray(bs)
        self.seg = np.asarray(seg)
        self.starts = np.flatnonzero(np.r_[True, self.seg[1:] != self.seg[:-1]])
        self.tgt = np.asarray(tgt_rows)
        self.strata = np.asarray(strata)

    def ranks(self, w: np.ndarray) -> np.ndarray:
        z = self.base + self.X @ w
        above = (z > z[self.tgt][self.seg]).astype(np.int64)
        return 1 + np.add.reduceat(above, self.starts)

    def mrr(self, w: np.ndarray) -> float:
        return float(np.mean(1.0 / self.ranks(w)))


def coordinate_ascent(pool: FullPool, w0: np.ndarray, rounds: int = 4) -> np.ndarray:
    """Direct MRR maximisation from the logit solution: per weight, try a grid of values."""
    w = w0.copy()
    best = pool.mrr(w)
    grid = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
    for _ in range(rounds):
        improved = False
        for k in range(len(w)):
            for scale in grid:
                cand = w.copy()
                cand[k] = scale if w0[k] == 0 else scale * (abs(w0[k]) if abs(w0[k]) > 0.1 else 1.0)
                value = pool.mrr(cand)
                if value > best + 1e-6:
                    best, w, improved = value, cand, True
        if not improved:
            break
    return w


def rank_eval(data: Data, samples: list[dict], weights: dict[str, float] | None, shuffle: bool = False):
    """Rank of the target in its full category pool, single shot (no conversation)."""
    out = defaultdict(list)
    rng = random.Random(11)
    order = list(range(len(samples)))
    rng.shuffle(order)
    for i, s in enumerate(samples):
        other = samples[order[i]] if shuffle else None
        prof = data.profile(s, other) if weights else None
        target = s["ground_truth"]["parent_asin"]
        pool = data.pools[data.items[target]["coarse"]]

        def score(a):
            v = data.base[a]
            if prof is not None:
                v += sum(weights.get(n, 0.0) * f for n, f in zip(FEATURES, prof.features(data.items[a])))
            return v
        ts = score(target)
        rank = 1 + sum(1 for a in pool if a != target and score(a) > ts)
        out["all"].append(rank)
        out[s.get("stratum", "other")].append(rank)
    return {k: {"n": len(v), "mrr": round(sum(1 / r for r in v) / len(v), 4),
                "top10": round(sum(r <= 10 for r in v) / len(v), 4)} for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default=str(ROOT / "data" / "memory_bench"))
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "memory_weights.json"))
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--negatives", type=int, default=200)
    parser.add_argument("--direct", type=int, default=8000, help="train sessions for direct MRR search")
    parser.add_argument("--half-lives", nargs="+", type=float, default=[365])
    parser.add_argument("--l2s", nargs="+", type=float, default=[0.01, 0.02, 0.03])
    args = parser.parse_args()

    feature_sets = {
        "all": FEATURES,
        # store affinity, its loyalty gate, already-owned, department and price band: the features that
        # carried signal in the permutation analysis; material / colour / card values only added noise
        "store_dept_price": ("store_share", "store_seen", "store_bad", "same_item", "loyal_store",
                             "dept_share", "price_fit"),
    }
    configs = []
    for half_life in args.half_lives:
        t0 = time.time()
        data = Data(Path(args.bench), Path(args.catalog), half_life)
        X, base, corr, sess = build_matrix(data, data.splits["train"][: args.train], args.negatives, random.Random(3))
        dev_pool = FullPool(data, data.splits["dev"])
        print(f"half-life {half_life:.0f}d: {X.shape[0]:,} rows ({time.time() - t0:.0f}s); "
              f"cold dev MRR {dev_pool.mrr(np.zeros(len(FEATURES))):.4f}", flush=True)
        for set_name, names in feature_sets.items():
            mask = np.array([f in names for f in FEATURES])
            for l2 in args.l2s:
                w_vec = np.zeros(len(FEATURES))
                w_vec[mask] = fit(X[:, mask], base, corr, sess, l2)
                dev_mrr = dev_pool.mrr(w_vec)
                w = dict(zip(FEATURES, (round(float(v), 4) for v in w_vec)))
                print(f"  {set_name:16s} l2 {l2}: dev MRR {dev_mrr:.4f}  w={w}", flush=True)
                configs.append((dev_mrr, len(names), l2, half_life, set_name, w, data))
    # Selection rule (fixed before looking at test): best dev MRR; anything within 0.001 of it counts
    # as a tie, broken by fewer features, then lighter regularisation, then the 365-day half-life.
    top = max(c[0] for c in configs)
    tied = [c for c in configs if c[0] >= top - 0.001]
    tied.sort(key=lambda c: (c[1], c[2], abs(c[3] - 365)))
    dev_mrr, _, l2, half_life, method, weights, data = tied[0]
    best = (dev_mrr, half_life, l2, weights, data, method)
    _, half_life, l2, weights, data, method = best
    report = {}
    for split in ("dev", "test", "test_loyal"):
        if split in data.splits:
            report[split] = {
                "cold": rank_eval(data, data.splits[split], None),
                "memory": rank_eval(data, data.splits[split], weights),
                "shuffled": rank_eval(data, data.splits[split], weights, shuffle=True),
            }
    Path(args.out).write_text(json.dumps({
        "weights": weights, "half_life_days": half_life, "l2": l2, "method": method, "direct_sessions": args.direct,
        "train_sessions": args.train, "negatives": args.negatives, "single_shot_ranking": report,
    }, indent=1), encoding="utf-8")
    print(f"\nbest: {method}, half-life {half_life:.0f}d, l2 {l2} → {args.out}")
    for split, conds in report.items():
        print(f"\n{split}")
        for cond, groups in conds.items():
            print(f"  {cond:9s} " + "  ".join(f"{g} n={v['n']} MRR {v['mrr']:.3f} top10 {v['top10']:.3f}"
                                             for g, v in sorted(groups.items())))


if __name__ == "__main__":
    main()
