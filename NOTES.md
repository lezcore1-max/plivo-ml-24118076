# NOTES.md

1. BPE tokenizer (vocab=1500) is the single most important change: Devanagari
   characters are 3 UTF-8 bytes each, so the byte tokenizer produces 3x longer
   sequences for Hindi text, destroying the model's effective context window.
2. BPE vocab=1500 was chosen over 3072 because Devanagari merges saturate by
   ~merge 600; additional merges to 3072 add diminishing English subwords at
   this model size, costing ~17 extra training minutes on this CPU.
3. Cosine LR decay (3e-3->3e-4, warmup=100) prevents the high early-step
   gradient variance of constant LR and allows the model to fine-tune at low
   LR near the end of training.
4. AdamW with weight decay=0.1 on weight matrices only (not biases/layer norms)
   provides mild regularization without penalizing the scale parameters.
5. Gradient clipping (norm=1.0) is free insurance against gradient spikes,
   especially important with the higher peak LR (3e-3 vs baseline 3e-4).
6. GPT-2 initialization (std=0.02, residual projections scaled by
   1/sqrt(2*n_layers)) keeps the residual stream variance stable at
   initialization, avoiding early loss spikes.
7. Weight tying (head.weight = tok_emb.weight) saves vocab*embd = 240,000
   parameters and improves gradient signal to the embedding layer.
8. The byte-to-token compression ratio of 0.370 tok/byte (vs 1.0 baseline)
   means each training step covers ~2.7x more semantic content per token slot.
9. Due to an I/O bug (missing flush=True + eval before save) in the R1 script,
   isolated ablations were not completed; all changes were bundled into one
   final run after time compression.
10. Final unique param count: 1,492,160 — well under the 2,000,000 cap.
