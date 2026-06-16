"""ECHO on the Agent World Model env — a runnable, CPU-only walkthrough.

What it shows, end to end:

  1. Take a real **AWM** tool-use episode (offline fixture by default; ``--live``
     captures one from a running ``agent_world_model_env`` server).
  2. Turn it into an ECHO **role-masked trajectory** (action / env_output / warning).
  3. Count the "free" signal: how many target tokens are *environment observations*
     that standard agent-RL **masks out and discards** — the tokens ECHO recovers.
  4. Compute the loss three ways on the same forward pass:
       * vanilla GRPO            (λ = 0)
       * ECHO                    (λ = 0.05)
       * verifier-free ECHO      (reward off — env-token CE only)

Default tokenizer/model are tiny + deterministic (no downloads). Use ``--hf MODEL``
for a real Hugging Face tokenizer + model (e.g. ``sshleifer/tiny-gpt2`` or a small
Qwen). Run from this directory:

    python run_demo.py
    python run_demo.py --hf sshleifer/tiny-gpt2
    python run_demo.py --live http://localhost:8899
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from awm import awm_episode_to_trajectory, live_capture
from echo import ACTION, CONTEXT, ENV_OUTPUT, WARNING, echo_loss, tokenize_trajectory

HERE = pathlib.Path(__file__).parent
FIXTURE = HERE / "fixtures" / "awm_ecommerce_episode.json"


# ── tiny, deterministic, zero-download tokenizer + model ─────────────────────
class CharTokenizer:
    """A minimal Hugging Face-shaped tokenizer (callable -> {"input_ids": [...]})."""

    def __init__(self, corpus: str) -> None:
        chars = sorted(set(corpus))
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for pad
        self.vocab_size = len(self.stoi) + 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [self.stoi[c] for c in text]}


class ToyLM(torch.nn.Module):
    """A small seeded next-token model — enough to make the loss numbers real."""

    def __init__(self, vocab_size: int, dim: int = 48) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab_size, dim)
        self.lm = torch.nn.Linear(dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # [B, T] -> [B, T, V]
        return self.lm(self.emb(ids))


def load_episode(args: argparse.Namespace) -> dict:
    episode = json.loads(FIXTURE.read_text())
    if args.live:
        print(f"• live: replaying {len(episode['steps'])} tool calls against {args.live}")
        episode = live_capture(args.live, episode)
        print("• captured real observations from agent_world_model_env\n")
    return episode


def build_logits(tok: dict, args: argparse.Namespace):
    input_ids = tok["input_ids"].unsqueeze(0)  # [1, T]
    if args.hf:
        from transformers import AutoModelForCausalLM  # lazy

        model = AutoModelForCausalLM.from_pretrained(args.hf)
        with torch.no_grad():
            logits = model(input_ids).logits
    else:
        model = ToyLM(vocab_size=int(input_ids.max()) + 2)
        with torch.no_grad():
            logits = model(input_ids)
    return input_ids, logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", metavar="MODEL", help="HF model id for real tokenizer+logits")
    ap.add_argument("--live", metavar="BASE_URL", help="capture from a running AWM server")
    ap.add_argument("--lam", type=float, default=0.05, help="ECHO world-model coeff λ")
    args = ap.parse_args()

    episode = load_episode(args)
    traj = awm_episode_to_trajectory(episode)

    # tokenizer: real HF, or a char tokenizer fit to this trajectory's text
    if args.hf:
        from transformers import AutoTokenizer  # lazy

        tokenizer = AutoTokenizer.from_pretrained(args.hf)
    else:
        tokenizer = CharTokenizer("".join(s.text for s in traj.segments))

    tok = tokenize_trajectory(tokenizer, traj, world_loss_target="env_only")
    input_ids, logits = build_logits(tok, args)
    masks = {k: v.unsqueeze(0) for k, v in tok.items() if k != "input_ids"}

    # ── role accounting (next-token: roles taken from target tokens) ─────────
    n_ctx = int((torch.tensor([s.role == CONTEXT for s in traj.segments])).sum())  # seg-level
    a = tok["action_mask"][1:]
    o = tok["obs_mask"][1:]
    w = tok["warning_mask"][1:]
    ctx = (~a & ~o & ~w)
    n_action, n_obs, n_warn, n_context = int(a.sum()), int(o.sum()), int(w.sum()), int(ctx.sum())
    learnable = n_action + n_obs
    pct_obs = 100.0 * n_obs / max(learnable, 1)
    ratio = n_obs / max(n_action, 1)

    print("=" * 64)
    print(f"AWM scenario : {traj.meta.get('scenario')}  (task_idx {traj.meta.get('task_idx')})")
    print(f"task         : {traj.task_prompt[:70]}...")
    print(f"steps        : {traj.meta.get('num_steps')}   reward: {traj.reward}")
    print("-" * 64)
    print("per-token roles (target tokens):")
    print(f"  context     {n_context:5d}   (given — never a loss target)")
    print(f"  action      {n_action:5d}   (GRPO / policy-gradient target)")
    print(f"  env_output  {n_obs:5d}   (ECHO world-model target — normally discarded)")
    print(f"  warning     {n_warn:5d}   (harness boilerplate — excluded from env loss)")
    print("-" * 64)
    print(f"ECHO 'free signal': {n_obs}/{learnable} learnable tokens "
          f"({pct_obs:.0f}%) are environment observations")
    print(f"                    standard agent-RL trains on {n_action}; "
          f"ECHO adds {n_obs} more ({ratio:.1f}x) at ~zero extra compute")
    print("-" * 64)

    adv = torch.tensor([traj.reward])  # single-sequence stand-in for group-relative adv

    _, m_grpo = echo_loss(logits, input_ids, masks["action_mask"], masks["obs_mask"], adv,
                          world_model_coeff=0.0)
    _, m_echo = echo_loss(logits, input_ids, masks["action_mask"], masks["obs_mask"], adv,
                          world_model_coeff=args.lam)
    _, m_free = echo_loss(logits, input_ids, masks["action_mask"], masks["obs_mask"], adv,
                          world_model_coeff=args.lam, use_rl=False)

    print("loss on the SAME forward pass:")
    print(f"  vanilla GRPO (λ=0)        loss={m_grpo['loss']:+.4f}  "
          f"(l_grpo={m_grpo['l_grpo']:+.4f}, l_env={m_grpo['l_env']:.4f} unused)")
    print(f"  ECHO (λ={args.lam})            loss={m_echo['loss']:+.4f}  "
          f"(l_grpo={m_echo['l_grpo']:+.4f}, l_env={m_echo['l_env']:.4f})")
    print(f"  verifier-free ECHO        loss={m_free['loss']:+.4f}  "
          f"(reward off → l_grpo=0, pure env-token CE)")
    print("=" * 64)
    print("takeaway: the env_output tokens above are computed in the same forward")
    print("pass and normally thrown away. ECHO turns them into dense training signal")
    print("— 'the world is a free loss function.'  See README.md / RFC 010 (PR #16).")


if __name__ == "__main__":
    main()
