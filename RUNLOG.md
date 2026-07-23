# RUNLOG.md

## Run 0 — Baseline (starter code, unmodified)
**Hypothesis:** Establish reference bpb with stock byte-tokenizer GPT.

**Config:** vocab=256 (byte), block=128, n_layer=4, n_head=4, n_embd=160,
tie_weights=False, Adam lr=3e-4 constant, batch=8, steps=2000, no warmup,
no clip, no schedule, std=0.05 init.

**sum(p.numel() for p in model.parameters()) = 1,339,840**

**Raw training output:**
```
corpus: 7,318,592 bytes -> 7,318,592 tokens (vocab 256)
model: 1,339,840 params
step     1  loss 5.6475  (432 ms/step)
step   100  loss 2.8522  (102 ms/step)
step   200  loss 2.2753  (102 ms/step)
step   300  loss 2.2126  (102 ms/step)
step   400  loss 2.1896  (102 ms/step)
step   500  loss 2.1897  (101 ms/step)
step   600  loss 2.1240  (101 ms/step)
step   700  loss 2.0959  (102 ms/step)
step   800  loss 2.0686  (102 ms/step)
step   900  loss 2.0983  (102 ms/step)
step  1000  loss 2.0164  (102 ms/step)
step  1100  loss 1.9847  (102 ms/step)
step  1200  loss 1.9508  (102 ms/step)
step  1300  loss 1.8913  (102 ms/step)
step  1400  loss 1.8323  (103 ms/step)
step  1500  loss 1.8090  (103 ms/step)
step  1600  loss 1.7905  (103 ms/step)
step  1700  loss 1.7931  (103 ms/step)
step  1800  loss 1.7434  (103 ms/step)
step  1900  loss 1.7505  (103 ms/step)
step  2000  loss 1.7315  (103 ms/step)
saved ckpt.pt  (206s total)
```

**Raw evaluate.py output:**
```json
{"bpb": 2.3718, "n_params": 1339840, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}
```

**dev bpb: 2.3718** (reference)

**Conclusion:** Baseline established. Top problems identified: byte tokenizer
triples Hindi/Devanagari sequence length (7.3M tokens from 7.3MB = 1.0 tok/byte);
constant LR with no warmup causes high early gradient variance; Adam not AdamW;
no grad clip; no weight tying; std=0.05 init on all layers including proj.

---

## BPE training decision
Original target: vocab=3072 (2816 merges). Cancelled at merge ~801 (~10 min)
and restarted at vocab=1500 (1244 merges). Reason: Devanagari merges saturate
by ~merge 600; remaining merges to 3072 mostly add marginal English subwords
with diminishing returns at this model size — traded ~17 min for training time
given the 2hr budget.

BPE result: 5000 chars -> 1856 tokens (0.370 tok/byte). Round-trip verified OK.

---

## Run 1 through R4 — NOT individually isolated (time compression)
**Reason:** An I/O bug in the original R1 script (eval_bpb ran before checkpoint
save, with no flush=True on prints) caused 18+ min of silent CPU use with zero
visible output. This consumed all remaining time budget for isolated experiments.
The per-run wall-clock time on the test machine (Intel Core 5 120U, no GPU) was
~18 min rather than the estimated 4 min (405 ms/step vs 120 ms/step estimated).
Combined with BPE overhead, this forced collapsing all improvements into one
bundled final run.

**Changes NOT individually tested:** cosine LR schedule, AdamW, grad_clip,
GPT-2 init, tie_weights were each expected to help based on standard practice
but their individual contributions were not isolated. tie_weights=True was
included in final config on theoretical grounds (saves vocab*embd params,
proven in Press & Wolf 2017) without an isolated ablation.

---

## Final Run — Bundled (all improvements)
**Hypothesis:** Combining BPE vocab=1500 + cosine LR decay (3e-3->3e-4, warmup=100)
+ AdamW (wd=0.1, matrices only) + grad_clip=1.0 + GPT-2 init (std=0.02, proj
scaled 1/sqrt(2*n_layers)) + tie_weights=True will substantially beat baseline
2.3718 bpb. BPE alone is expected to be the largest contributor (3x context
compression for Hindi); LR schedule the second largest.

**Config:** vocab=1500 (BPE), block=128, n_layer=4, n_head=4, n_embd=160,
tie_weights=True, AdamW lr=3e-3->3e-4 cosine, warmup=100, batch=16,
grad_clip=1.0, GPT-2 init, steps=2000.

**sum(p.numel() for p in model.parameters()) = 1,492,160**  (under 2M cap)

**Raw training output:** (see live output above)

**Raw evaluate.py output:**
```json
{"bpb": 1.9436, "n_params": 1492160, "steps": 2000, "tokens_in_eval": 74274, "tokens_scored": 74273}
```

**dev bpb:** 1.9436

**Delta vs baseline 2.3718:** -0.4282
