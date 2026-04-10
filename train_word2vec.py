import argparse
import json
import math
import os
import random
import re
from collections import Counter
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


def build_vocab(tokens: list[str], min_count: int) -> tuple[list[str], dict[str, int], list[int], Counter]:
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort(key=lambda w: (-counts[w], w))

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    indexed = [word_to_idx[t] for t in tokens if t in word_to_idx]
    return vocab, word_to_idx, indexed, counts


def build_training_pairs(indexed_tokens: list[int], window_size: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n = len(indexed_tokens)
    for i in range(n):
        center = indexed_tokens[i]
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)
        for j in range(left, right):
            if i == j:
                continue
            context = indexed_tokens[j]
            pairs.append((center, context))
    return pairs


def init_matrix(rows: int, cols: int, scale: float = 0.5) -> list[list[float]]:
    return [[(random.random() - 0.5) * scale for _ in range(cols)] for _ in range(rows)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(x: float) -> float:
    if x < -20:
        return 0.0
    if x > 20:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    num = dot(a, b)
    da = math.sqrt(dot(a, a))
    db = math.sqrt(dot(b, b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def sample_negative_indices(vocab_size: int, count: int, forbidden: set[int]) -> list[int]:
    negatives: list[int] = []
    while len(negatives) < count:
        idx = random.randrange(vocab_size)
        if idx in forbidden:
            continue
        negatives.append(idx)
    return negatives


def train_skipgram_negative_sampling(
    pairs: list[tuple[int, int]],
    vocab_size: int,
    dim: int,
    epochs: int,
    learning_rate: float,
    negative_samples: int,
    seed: int,
    wandb_run=None,
) -> tuple[list[list[float]], list[list[float]]]:
    random.seed(seed)
    w_in = init_matrix(vocab_size, dim, scale=0.2)
    w_out = init_matrix(vocab_size, dim, scale=0.2)

    if not pairs:
        return w_in, w_out

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0

        for center_idx, context_idx in pairs:
            center_vec = w_in[center_idx]

            # Positive sample update.
            out_pos = w_out[context_idx]
            score_pos = dot(center_vec, out_pos)
            sig_pos = sigmoid(score_pos)
            grad_pos = sig_pos - 1.0
            total_loss += -math.log(max(sig_pos, 1e-12))

            center_grad = [grad_pos * v for v in out_pos]
            for k in range(dim):
                out_pos[k] -= learning_rate * grad_pos * center_vec[k]

            # Negative sample updates.
            negatives = sample_negative_indices(vocab_size, negative_samples, {center_idx, context_idx})
            for neg_idx in negatives:
                out_neg = w_out[neg_idx]
                score_neg = dot(center_vec, out_neg)
                sig_neg = sigmoid(score_neg)
                grad_neg = sig_neg
                total_loss += -math.log(max(1.0 - sig_neg, 1e-12))

                for k in range(dim):
                    center_grad[k] += grad_neg * out_neg[k]
                    out_neg[k] -= learning_rate * grad_neg * center_vec[k]

            for k in range(dim):
                center_vec[k] -= learning_rate * center_grad[k]

        avg_loss = total_loss / len(pairs)
        print(f"Epoch {epoch}/{epochs} - avg_loss: {avg_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train/avg_loss": avg_loss})

    return w_in, w_out


def save_embeddings(path: Path, vocab: list[str], embeddings: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(vocab)} {len(embeddings[0])}\n")
        for word, vec in zip(vocab, embeddings):
            values = " ".join(f"{x:.6f}" for x in vec)
            f.write(f"{word} {values}\n")


def save_vocab(path: Path, vocab: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(vocab), encoding="utf-8")


def save_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def nearest_neighbors(
    word: str,
    vocab: list[str],
    word_to_idx: dict[str, int],
    embeddings: list[list[float]],
    top_k: int,
) -> list[tuple[str, float]]:
    if word not in word_to_idx:
        return []
    i = word_to_idx[word]
    target = embeddings[i]
    sims = []
    for j, other in enumerate(vocab):
        if j == i:
            continue
        sims.append((other, cosine_similarity(target, embeddings[j])))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


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
        "settings": settings,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"Warning: W&B init failed, continuing without W&B logging. Reason: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Word2Vec (Skip-gram + Negative Sampling) trainer.")
    parser.add_argument("--input", type=Path, default=Path("data/alice.txt"), help="Path to raw text corpus.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/embeddings.txt"), help="Embeddings file output path.")
    parser.add_argument("--vocab-output", type=Path, default=Path("artifacts/vocab.txt"), help="Vocabulary file output path.")
    parser.add_argument("--metadata-output", type=Path, default=Path("artifacts/train_metadata.json"), help="Metadata JSON output path.")
    parser.add_argument("--dim", type=int, default=50, help="Embedding dimension.")
    parser.add_argument("--window", type=int, default=2, help="Context window size.")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum count for vocabulary inclusion.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--neg-samples", type=int, default=5, help="Number of negative samples per pair.")
    parser.add_argument("--max-tokens", type=int, default=50000, help="Limit tokens for quick local training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="word2vec-local", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user).")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"], help="W&B mode.")
    parser.add_argument("--wandb-dir", type=Path, default=Path(".wandb"), help="W&B local directory.")
    args = parser.parse_args()

    random.seed(args.seed)
    text = read_text(args.input)
    tokens = tokenize(text)
    if args.max_tokens > 0:
        tokens = tokens[: args.max_tokens]

    vocab, word_to_idx, indexed_tokens, raw_counts = build_vocab(tokens, args.min_count)
    pairs = build_training_pairs(indexed_tokens, args.window)

    print(f"Input file: {args.input}")
    print(f"Total tokens (after cap): {len(tokens)}")
    print(f"Vocabulary size (min_count={args.min_count}): {len(vocab)}")
    print(f"Training pairs: {len(pairs)}")

    if args.wandb_dir:
        args.wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(args.wandb_dir.resolve()))

    wandb_run = init_wandb(
        args,
        {
            "input": str(args.input),
            "dim": args.dim,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "neg_samples": args.neg_samples,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "vocab_size": len(vocab),
            "pairs": len(pairs),
        },
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "dataset/num_tokens": len(tokens),
                "dataset/vocab_size": len(vocab),
                "dataset/num_pairs": len(pairs),
            }
        )

    w_in, _ = train_skipgram_negative_sampling(
        pairs=pairs,
        vocab_size=len(vocab),
        dim=args.dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        negative_samples=args.neg_samples,
        seed=args.seed,
        wandb_run=wandb_run,
    )

    save_embeddings(args.output, vocab, w_in)
    save_vocab(args.vocab_output, vocab)
    save_metadata(
        args.metadata_output,
        {
            "input": str(args.input),
            "embedding_path": str(args.output),
            "vocab_path": str(args.vocab_output),
            "dim": args.dim,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "neg_samples": args.neg_samples,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "num_tokens": len(tokens),
            "vocab_size": len(vocab),
            "num_pairs": len(pairs),
        },
    )
    print(f"Embeddings written to: {args.output}")
    print(f"Vocabulary written to: {args.vocab_output}")
    print(f"Metadata written to: {args.metadata_output}")

    for query in ["alice", "rabbit", "queen", "king"]:
        neighbors = nearest_neighbors(query, vocab, word_to_idx, w_in, top_k=5)
        if not neighbors:
            continue
        pretty = ", ".join(f"{w}:{s:.3f}" for w, s in neighbors)
        print(f"Nearest to '{query}': {pretty}")
        if wandb_run is not None:
            table = wandb.Table(columns=["query", "neighbor", "cosine_similarity"])
            for w, s in neighbors:
                table.add_data(query, w, s)
            wandb_run.log({f"neighbors/{query}": table})

    print("Top 10 most common vocab terms:")
    for w in vocab[:10]:
        print(f"  {w}: {raw_counts[w]}")
    if wandb_run is not None:
        wandb_run.log({"vocab/top10": "\n".join(f"{w}: {raw_counts[w]}" for w in vocab[:10])})
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"Warning: W&B finish failed: {exc}")


if __name__ == "__main__":
    main()
