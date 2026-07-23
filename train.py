"""Improved trainer.
Changes from baseline:
- AdamW with weight decay only on weight matrices (not bias/norm)
- Cosine LR decay with linear warmup
- Gradient clipping (norm=1.0)
- Gradient accumulation for larger effective batch
- Periodic dev bpb eval

Hard caps: max 2000 optimizer steps, max 2,000,000 unique params.

Usage from starter/:
    python train.py --data ../data/train_corpus.txt --dev ../data/dev_eval.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time
import torch
import torch.nn.functional as F
from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS  = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device="cpu"):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + block + 1] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, warmup, total, lr_max, lr_min=3e-4):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def eval_bpb(model, cfg, tok, text, device="cpu"):
    n_bytes = len(text.encode("utf-8"))
    ids = torch.tensor(tok.encode(text), dtype=torch.long, device=device)
    block, stride = cfg.block_size, max(1, cfg.block_size // 2)
    total_nll, n_scored = 0.0, 0
    scored = 1
    model.eval()
    while scored < len(ids):
        start = max(0, scored - stride)
        end   = min(len(ids), start + block)
        window = ids[start:end]
        logits, _ = model(window[None, :])
        logp = torch.log_softmax(logits[0], dim=-1)
        targets = ids[start + 1:end]
        nll = -logp[torch.arange(len(targets)), targets]
        offset = scored - (start + 1)
        total_nll += nll[offset:].sum().item()
        n_scored  += len(nll) - offset
        scored = end
    model.train()
    return total_nll / math.log(2) / n_bytes


def make_optimizer(model, lr, weight_decay=0.1):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",         required=True)
    ap.add_argument("--dev",          default=None)
    ap.add_argument("--steps",        type=int,   default=2000)
    ap.add_argument("--batch",        type=int,   default=16)
    ap.add_argument("--accum",        type=int,   default=2)
    ap.add_argument("--lr",           type=float, default=3e-3)
    ap.add_argument("--lr_min",       type=float, default=3e-4)
    ap.add_argument("--warmup",       type=int,   default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip",    type=float, default=1.0)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out",          default="ckpt.pt")
    ap.add_argument("--log_every",    type=int,   default=100)
    ap.add_argument("--eval_every",   type=int,   default=500)
    args = ap.parse_args()

    assert args.steps <= MAX_STEPS, "cap: max %d steps" % MAX_STEPS
    torch.manual_seed(args.seed)
    device = "cpu"

    tok = tokenizer_mod.load()
    print("tokenizer: vocab_size=%d" % tok.vocab_size)

    text = open(args.data, encoding="utf-8").read()
    ids  = torch.tensor(tok.encode(text), dtype=torch.long)
    print("corpus: %d bytes -> %d tokens (%.3f tok/byte)" % (
        len(text.encode("utf-8")), len(ids),
        len(ids)/len(text.encode("utf-8"))))

    dev_text = open(args.dev, encoding="utf-8").read() if args.dev else None

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n_unique = model.n_params_unique()
    print("model: %d total params, %d unique (tie_weights=%s)" % (
        model.n_params(), n_unique, cfg.tie_weights))
    assert n_unique <= MAX_PARAMS, "cap exceeded: %d > %d" % (n_unique, MAX_PARAMS)

    opt = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    t0 = time.time()
    losses_log = []
    opt_step = 0

    print("\nsteps=%d  batch=%d  accum=%d  eff_batch=%d  lr=%.1e->%.1e  warmup=%d  block=%d" % (
        args.steps, args.batch, args.accum, args.batch*args.accum,
        args.lr, args.lr_min, args.warmup, cfg.block_size))

    while opt_step < args.steps:
        opt.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for _ in range(args.accum):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            _, loss = model(x, y)
            (loss / args.accum).backward()
            batch_loss += loss.item() / args.accum

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        lr_now = get_lr(opt_step, args.warmup, args.steps, args.lr, args.lr_min)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.step()
        opt_step += 1
        losses_log.append(batch_loss)

        if opt_step % args.log_every == 0 or opt_step == 1:
            avg = sum(losses_log[-args.log_every:]) / len(losses_log[-args.log_every:])
            ms  = (time.time() - t0) / opt_step * 1000
            print("step %5d  loss %.4f  lr %.2e  %.0fms/step" % (
                opt_step, avg, lr_now, ms))

        if dev_text and (opt_step % args.eval_every == 0 or opt_step == args.steps):
            bpb = eval_bpb(model, cfg, tok, dev_text, device)
            print("  >>> step %d  dev bpb=%.4f" % (opt_step, bpb))

    torch.save({
        "model":  model.state_dict(),
        "config": {k: getattr(cfg, k) for k in dir(cfg)
                   if not k.startswith("_") and not callable(getattr(cfg, k))},
        "steps":  opt_step,
        "train_loss_curve": losses_log,
    }, args.out)
    print("\nSaved %s  (%.0fs total, %d steps)" % (args.out, time.time()-t0, opt_step))
    print("Unique params: %d" % model.n_params_unique())


if __name__ == "__main__":
    main()
