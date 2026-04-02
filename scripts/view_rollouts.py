"""View decoded rollouts from a training step, one at a time.

Run from the prime-rl directory:

    uv run python scripts/view_rollouts.py <step>

Examples:
    uv run python scripts/view_rollouts.py 0           # browse rollouts one by one
    uv run python scripts/view_rollouts.py 5 --index 3  # jump to rollout #3
    uv run python scripts/view_rollouts.py 0 --rank 2   # only rank 2
"""

import argparse
import sys
from pathlib import Path

import msgspec
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prime_rl.transport.types import MicroBatch


def extract_rollouts(batch: MicroBatch, tokenizer) -> list[dict]:
    """Extract individual rollouts from a packed micro-batch."""
    ids = batch.input_ids
    mask = batch.loss_mask
    advs = batch.advantages
    logprobs = batch.inference_logprobs

    # Find completion regions (contiguous True in loss_mask)
    regions = []
    in_completion = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_completion:
            start = i
            in_completion = True
        elif not m and in_completion:
            regions.append((start, i))
            in_completion = False
    if in_completion:
        regions.append((start, len(mask)))

    rollouts = []
    for s, e in regions:
        # Scan backwards from completion start to find prompt beginning (skip padding)
        pad_id = tokenizer.pad_token_id or 0
        prompt_start = 0
        for j in range(s - 1, -1, -1):
            if ids[j] == pad_id:
                prompt_start = j + 1
                break

        completion_lps = logprobs[s:e]
        avg_logprob = sum(completion_lps) / max(len(completion_lps), 1)

        rollouts.append({
            "prompt_tokens": s - prompt_start,
            "completion_tokens": e - s,
            "prompt_ids": ids[prompt_start:s],
            "completion_ids": ids[s:e],
            "avg_advantage": sum(advs[s:e]) / max(len(advs[s:e]), 1),
            "avg_logprob": avg_logprob,
        })

    return rollouts


def load_all_rollouts(step_dir: Path, tokenizer, decoder, rank: int | None = None) -> list[dict]:
    """Load and flatten all rollouts from a step directory."""
    if rank is not None:
        rank_files = [step_dir / f"rank_{rank}.bin"]
        if not rank_files[0].exists():
            print(f"Error: {rank_files[0]} does not exist")
            sys.exit(1)
    else:
        rank_files = sorted(step_dir.glob("rank_*.bin"))

    if not rank_files:
        print(f"No rank_*.bin files found in {step_dir}")
        sys.exit(1)

    all_rollouts = []
    for rank_file in rank_files:
        rank_num = rank_file.stem.split("_")[1]
        with open(rank_file, "rb") as f:
            micro_batches = decoder.decode(f.read())

        for bi, batch in enumerate(micro_batches):
            for ri, r in enumerate(extract_rollouts(batch, tokenizer)):
                r["rank"] = rank_num
                r["micro_batch"] = bi
                r["rollout"] = ri
                all_rollouts.append(r)

    return all_rollouts


def display_rollout(idx: int, total: int, r: dict, tokenizer):
    """Display a single rollout."""
    prompt_text = tokenizer.decode(r["prompt_ids"], skip_special_tokens=False)
    completion_text = tokenizer.decode(r["completion_ids"], skip_special_tokens=False)

    print(f"\n{'='*80}")
    print(f"  Rollout {idx}/{total - 1}  (rank={r['rank']} micro_batch={r['micro_batch']} rollout={r['rollout']})")
    print(f"  prompt_tokens={r['prompt_tokens']}  completion_tokens={r['completion_tokens']}")
    print(f"  avg_advantage={r['avg_advantage']:+.4f}  avg_logprob={r['avg_logprob']:.4f}")
    print(f"{'='*80}")
    print(f"\n--- PROMPT ---\n{prompt_text}")
    print(f"\n--- COMPLETION ---\n{completion_text}\n")


def main():
    parser = argparse.ArgumentParser(description="View decoded rollouts from a training step, one at a time.")
    parser.add_argument("step", type=int, help="Step number to inspect")
    parser.add_argument("--rollouts-dir", type=str, default="/mnt/ckpts/rollouts",
                        help="Base rollouts directory (default: /mnt/ckpts/rollouts)")
    parser.add_argument("--rank", type=int, default=None,
                        help="Only show rollouts from this rank (default: all ranks)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B",
                        help="Tokenizer model name")
    parser.add_argument("--index", "-i", type=int, default=0,
                        help="Starting rollout index (default: 0)")
    args = parser.parse_args()

    step_dir = Path(args.rollouts_dir) / f"step_{args.step}"
    if not step_dir.exists():
        available = sorted(p.name for p in Path(args.rollouts_dir).iterdir() if p.is_dir())
        print(f"Error: {step_dir} does not exist. Available steps: {', '.join(available)}")
        sys.exit(1)

    print(f"Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])

    print(f"Loading rollouts from {step_dir}...")
    all_rollouts = load_all_rollouts(step_dir, tokenizer, decoder, rank=args.rank)
    total = len(all_rollouts)
    print(f"Found {total} rollouts.")

    idx = args.index
    if idx >= total:
        print(f"Error: index {idx} out of range (0-{total - 1})")
        sys.exit(1)

    while True:
        display_rollout(idx, total, all_rollouts[idx], tokenizer)

        try:
            cmd = input("[Enter]=next  [p]=prev  [N]=jump to N  [q]=quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd == "" or cmd == "n":
            idx = min(idx + 1, total - 1)
        elif cmd == "p":
            idx = max(idx - 1, 0)
        elif cmd == "q":
            break
        elif cmd.isdigit():
            jump = int(cmd)
            if 0 <= jump < total:
                idx = jump
            else:
                print(f"  Index out of range (0-{total - 1})")

    print("Done.")


if __name__ == "__main__":
    main()
