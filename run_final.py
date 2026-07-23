import math, time, sys, subprocess, json, os
import torch
import torch.nn as nn

# ── Fast BPE encode (word-level chunking to avoid O(n^2) on full corpus) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_FILE = os.path.join(_HERE, "bpe_vocab.json")

def load_bpe():
    with open(_VOCAB_FILE) as f:
        data = json.load(f)
    merges = [tuple(m) for m in data["merges"]]
    vocab  = data["vocab"]
    merge_rank = {m: i for i, m in enumerate(merges)}
    id_to_bytes = {i: bytes(v) for i, v in enumerate(vocab)}
    return merges, merge_rank, id_to_bytes, len(vocab)

def bpe_encode_fast(text, merge_rank, id_to_bytes):
    """Encode by splitting on spaces first, encode each word, concatenate.
    This is ~100x faster than encoding the full corpus as one sequence
    because each word is short, so the O(k^2) inner loop stays tiny."""
    raw = text.encode("utf-8")
    # Split at byte 32 (space) to get word chunks; encode each separately
    chunks = []
    start = 0
    for i, b in enumerate(raw):
        if b == 32 or i == len(raw)-1:
            end = i if b == 32 else i+1
            if end > start:
                chunks.append(list(raw[start:end]))
            if b == 32:
                chunks.append([32])
            start = i+1 if b == 32 else i+1
    ids = []
    n_merges = len(merge_rank)
    for chunk in chunks:
        c = list(chunk)
        while len(c) >= 2:
            best_rank = n_merges
            best_pos  = -1
            for i in range(len(c)-1):
                r = merge_rank.get((c[i], c[i+1]), n_merges)
                if r < best_rank:
                    best_rank = r
                    best_pos  = i
            if best_pos == -1:
                break
            c = c[:best_pos] + [256 + best_rank] + c[best_pos+2:]
        ids.extend(c)
    return ids

def bpe_decode(ids, id_to_bytes):
    return b"".join(id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")

from model import GPT, Config

MAX_STEPS  = 2000
MAX_PARAMS = 2_000_000

def get_batch(ids, block, batch):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x  = torch.stack([ids[i:i+block]   for i in ix])
    y  = torch.stack([ids[i+1:i+block+1] for i in ix])
    return x, y

def get_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * p))

torch.manual_seed(42)

print("Loading BPE vocab...", flush=True)
merges, merge_rank, id_to_bytes, vocab_size = load_bpe()
print("vocab_size=%d" % vocab_size, flush=True)

print("Encoding corpus (fast chunked BPE)...", flush=True)
t_enc = time.time()
text = open("../data/train_corpus.txt", encoding="utf-8").read()
ids_list = bpe_encode_fast(text, merge_rank, id_to_bytes)
ids = torch.tensor(ids_list, dtype=torch.long)
print("corpus: %d bytes -> %d tokens (%.3f tok/byte)  encode=%.1fs" % (
    len(text.encode("utf-8")), len(ids),
    len(ids)/len(text.encode("utf-8")), time.time()-t_enc), flush=True)

# Verify round-trip on sample
sample = text[:200]
sample_ids = bpe_encode_fast(sample, merge_rank, id_to_bytes)
recovered  = bpe_decode(sample_ids, id_to_bytes)
assert recovered == sample, "ROUND-TRIP FAIL"
print("Round-trip OK", flush=True)

cfg = Config()
cfg.vocab_size  = vocab_size
cfg.block_size  = 128
cfg.n_layer     = 4
cfg.n_head      = 4
cfg.n_embd      = 160
cfg.tie_weights = True
cfg.dropout     = 0.0

model = GPT(cfg)

def gpt2_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)
model.apply(gpt2_init)
for name, p in model.named_parameters():
    if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
        nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

n = sum(p.numel() for p in model.parameters())
print("sum(p.numel() for p in model.parameters()) = %d" % n, flush=True)
assert n <= MAX_PARAMS, "OVER CAP: %d" % n
print("param check PASSED", flush=True)

decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad: continue
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-3, betas=(0.9, 0.95), eps=1e-8)

model.train()
t0 = time.time()
losses = []
print("Training: steps=2000 batch=16 lr=3e-3->3e-4 warmup=100 clip=1.0", flush=True)
for step in range(1, MAX_STEPS+1):
    x, y = get_batch(ids, cfg.block_size, 16)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr_now = get_lr(step, 100, MAX_STEPS, 3e-3, 3e-4)
    for pg in opt.param_groups: pg["lr"] = lr_now
    opt.step()
    losses.append(loss.item())
    if step % 100 == 0 or step == 1:
        avg = sum(losses[-100:]) / len(losses[-100:])
        print("step %5d  loss %.4f  lr %.2e  (%.0f ms/step)" % (
            step, avg, lr_now, (time.time()-t0)/step*1000), flush=True)

print("Training done %.0fs" % (time.time()-t0), flush=True)

torch.save({
    "model":  model.state_dict(),
    "config": {k: getattr(cfg, k) for k in dir(cfg)
               if not k.startswith("_") and not callable(getattr(cfg, k))},
    "steps":  MAX_STEPS,
    "train_loss_curve": losses,
}, "ckpt.pt")
print("Saved ckpt.pt", flush=True)

print("Running evaluate.py...", flush=True)
r = subprocess.run(["python3", "-u", "evaluate.py", "--checkpoint", "ckpt.pt",
                    "--text_file", "../data/dev_eval.txt"],
                   capture_output=True, text=True)
print(r.stdout, flush=True)
if r.stderr: print("STDERR:", r.stderr[:300], flush=True)
print("ALL DONE", flush=True)
