import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z']+", text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def load_embeddings(path: Path) -> tuple[list[str], dict[str, int], list[list[float]]]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"Unexpected embedding header in {path}: {header}")
        vocab_size, dim = int(header[0]), int(header[1])
        vocab: list[str] = []
        vectors: list[list[float]] = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            vec = [float(x) for x in parts[1:]]
            if len(vec) != dim:
                continue
            vocab.append(word)
            vectors.append(vec)
    if len(vocab) != vocab_size:
        print(f"Warning: header vocab size={vocab_size}, parsed={len(vocab)}")
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_idx, vectors


def build_indexed_tokens(tokens: list[str], word_to_idx: dict[str, int]) -> list[int]:
    return [word_to_idx[t] for t in tokens if t in word_to_idx]


def build_context_map(indexed_tokens: list[int], window: int) -> dict[int, set[int]]:
    context_map: dict[int, set[int]] = defaultdict(set)
    n = len(indexed_tokens)
    for i in range(n):
        center = indexed_tokens[i]
        left = max(0, i - window)
        right = min(n, i + window + 1)
        for j in range(left, right):
            if i == j:
                continue
            context_map[center].add(indexed_tokens[j])
    return context_map


def nearest_neighbors(
    word_idx: int,
    vectors: list[list[float]],
    top_k: int,
) -> list[tuple[int, float]]:
    target = vectors[word_idx]
    sims = []
    for idx, vec in enumerate(vectors):
        if idx == word_idx:
            continue
        sims.append((idx, cosine_similarity(target, vec)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def evaluate_pairs(
    indexed_tokens: list[int],
    vectors: list[list[float]],
    window: int,
    sample_pairs: int,
    seed: int,
) -> dict[str, float]:
    random.seed(seed)
    if len(indexed_tokens) < 3:
        return {
            "eval/positive_cosine_mean": 0.0,
            "eval/negative_cosine_mean": 0.0,
            "eval/separation_margin": 0.0,
        }

    positives: list[tuple[int, int]] = []
    for i in range(len(indexed_tokens)):
        left = max(0, i - window)
        right = min(len(indexed_tokens), i + window + 1)
        for j in range(left, right):
            if i != j:
                positives.append((indexed_tokens[i], indexed_tokens[j]))
    if not positives:
        return {
            "eval/positive_cosine_mean": 0.0,
            "eval/negative_cosine_mean": 0.0,
            "eval/separation_margin": 0.0,
        }

    sample = random.sample(positives, min(sample_pairs, len(positives)))
    vocab_size = len(vectors)

    pos_scores = []
    neg_scores = []
    wins = 0
    for center, context in sample:
        pos = cosine_similarity(vectors[center], vectors[context])
        pos_scores.append(pos)
        neg_idx = random.randrange(vocab_size)
        while neg_idx == center or neg_idx == context:
            neg_idx = random.randrange(vocab_size)
        neg = cosine_similarity(vectors[center], vectors[neg_idx])
        neg_scores.append(neg)
        if pos > neg:
            wins += 1

    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)
    return {
        "eval/positive_cosine_mean": pos_mean,
        "eval/negative_cosine_mean": neg_mean,
        "eval/separation_margin": pos_mean - neg_mean,
        "eval/pairwise_accuracy": wins / len(sample),
    }


def evaluate_recall_at_k(
    context_map: dict[int, set[int]],
    vectors: list[list[float]],
    k_values: list[int],
    max_centers: int,
    seed: int,
) -> dict[str, float]:
    random.seed(seed)
    centers = [idx for idx, contexts in context_map.items() if contexts]
    if not centers:
        return {f"eval/recall@{k}": 0.0 for k in k_values}
    chosen = random.sample(centers, min(max_centers, len(centers)))

    recalls = {k: 0.0 for k in k_values}
    for center_idx in chosen:
        neighbors = nearest_neighbors(center_idx, vectors, top_k=max(k_values))
        neighbor_indices = [idx for idx, _ in neighbors]
        true_contexts = context_map[center_idx]
        for k in k_values:
            topk = set(neighbor_indices[:k])
            hit = 1.0 if any(c in topk for c in true_contexts) else 0.0
            recalls[k] += hit

    total = float(len(chosen))
    return {f"eval/recall@{k}": recalls[k] / total for k in k_values}


def build_projection_table(vocab: list[str], vectors: list[list[float]], n: int) -> list[tuple[str, float, float]]:
    entries = []
    for i in range(min(n, len(vocab))):
        vec = vectors[i]
        x = vec[0] if len(vec) > 0 else 0.0
        y = vec[1] if len(vec) > 1 else 0.0
        entries.append((vocab[i], x, y))
    return entries


def load_training_metadata(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def init_wandb(args, extra_config: dict):
    if not args.wandb:
        return None
    if wandb is None:
        raise RuntimeError(
            "W&B logging requested, but package 'wandb' is not installed. "
            "Install with: py -m pip install wandb"
        )
    try:
        settings = wandb.Settings(start_method="thread")
    except Exception:
        settings = None

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "config": extra_config,
        "mode": args.wandb_mode,
        "dir": str(args.wandb_dir) if args.wandb_dir else None,
        "job_type": "evaluation",
        "settings": settings,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"Warning: W&B init failed, continuing without W&B logging. Reason: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Word2Vec embeddings.")
    parser.add_argument("--input", type=Path, default=Path("data/alice.txt"), help="Raw text corpus path.")
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/embeddings.txt"), help="Embeddings file path.")
    parser.add_argument("--window", type=int, default=2, help="Context window used for evaluation.")
    parser.add_argument("--max-tokens", type=int, default=40000, help="Token cap to keep evaluation lightweight.")
    parser.add_argument("--sample-pairs", type=int, default=10000, help="Number of positive pairs to sample.")
    parser.add_argument("--max-centers", type=int, default=500, help="Number of center words used for recall@k.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/train_metadata.json"), help="Optional metadata JSON path.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="word2vec-local", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user).")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"], help="W&B mode.")
    parser.add_argument("--wandb-dir", type=Path, default=Path(".wandb"), help="W&B local directory.")
    args = parser.parse_args()

    vocab, word_to_idx, vectors = load_embeddings(args.embeddings)
    text = read_text(args.input)
    tokens = tokenize(text)
    if args.max_tokens > 0:
        tokens = tokens[: args.max_tokens]
    indexed_tokens = build_indexed_tokens(tokens, word_to_idx)
    context_map = build_context_map(indexed_tokens, args.window)

    pair_metrics = evaluate_pairs(
        indexed_tokens=indexed_tokens,
        vectors=vectors,
        window=args.window,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
    )
    recall_metrics = evaluate_recall_at_k(
        context_map=context_map,
        vectors=vectors,
        k_values=[1, 5, 10],
        max_centers=args.max_centers,
        seed=args.seed,
    )
    all_metrics = {**pair_metrics, **recall_metrics}

    print(f"Evaluated embeddings: {args.embeddings}")
    print(f"Tokens used: {len(tokens)}")
    print(f"In-vocab tokens: {len(indexed_tokens)}")
    for key, value in all_metrics.items():
        print(f"{key}: {value:.4f}")

    for query in ["alice", "rabbit", "queen", "king"]:
        if query not in word_to_idx:
            continue
        q_idx = word_to_idx[query]
        neighbors = nearest_neighbors(q_idx, vectors, top_k=5)
        pretty = ", ".join(f"{vocab[idx]}:{score:.3f}" for idx, score in neighbors)
        print(f"Nearest to '{query}': {pretty}")

    if args.wandb_dir:
        args.wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(args.wandb_dir.resolve()))

    training_metadata = load_training_metadata(args.metadata)
    config = {
        "input": str(args.input),
        "embeddings": str(args.embeddings),
        "window": args.window,
        "max_tokens": args.max_tokens,
        "sample_pairs": args.sample_pairs,
        "max_centers": args.max_centers,
        "seed": args.seed,
        "vocab_size": len(vocab),
    }
    config.update({f"train_{k}": v for k, v in training_metadata.items()})
    wandb_run = init_wandb(args, config)
    if wandb_run is not None:
        wandb_run.log(all_metrics)

        for query in ["alice", "rabbit", "queen", "king"]:
            if query not in word_to_idx:
                continue
            q_idx = word_to_idx[query]
            neighbors = nearest_neighbors(q_idx, vectors, top_k=10)
            table = wandb.Table(columns=["query", "neighbor", "cosine_similarity"])
            for idx, score in neighbors:
                table.add_data(query, vocab[idx], score)
            wandb_run.log({f"eval/neighbors/{query}": table})

        projection = build_projection_table(vocab, vectors, n=300)
        proj_table = wandb.Table(columns=["word", "x", "y"])
        for word, x, y in projection:
            proj_table.add_data(word, x, y)
        wandb_run.log({"eval/embedding_projection_first2d": proj_table})
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"Warning: W&B finish failed: {exc}")


if __name__ == "__main__":
    main()
