"""BPE tokenizer trained on train_corpus.txt.

Why BPE over byte-level:
- Devanagari chars are 3 UTF-8 bytes; byte tokenizer triples sequence
  length for Hindi, destroying effective context.
- BPE vocab ~3000 learns Hindi syllables as single tokens (~3x shorter seqs).
- Byte-level fallback (ids 0-255 = raw bytes) ensures lossless round-trip.

Interface (unchanged): load() -> tokenizer with .encode/.decode/.vocab_size
Vocab saved as bpe_vocab.json next to this file.
"""
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_FILE = os.path.join(_HERE, "bpe_vocab.json")


class BPETokenizer:
    """Byte-level BPE. Ids 0-255: raw bytes. Ids 256+: learned merges."""

    def __init__(self, merges, vocab):
        self.merges = merges
        self._merge_rank = {tuple(m): i for i, m in enumerate(merges)}
        self._id_to_bytes = {i: bytes(v) for i, v in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text):
        """Fast encode: split on spaces first, encode each chunk separately.
        Avoids O(n*merges) on the full corpus; each space-delimited chunk is
        short so the inner merge loop stays O(chunk_len^2) which is tiny."""
        raw = text.encode("utf-8")
        # Split into chunks at space boundaries
        chunks = []
        start = 0
        for i in range(len(raw)):
            if raw[i] == 32:  # space byte
                if i > start:
                    chunks.append(list(raw[start:i]))
                chunks.append([32])
                start = i + 1
        if start < len(raw):
            chunks.append(list(raw[start:]))
        n_merges = len(self.merges)
        ids = []
        for chunk in chunks:
            c = list(chunk)
            while len(c) >= 2:
                best_rank = n_merges
                best_pos  = -1
                for i in range(len(c) - 1):
                    r = self._merge_rank.get((c[i], c[i+1]), n_merges)
                    if r < best_rank:
                        best_rank = r
                        best_pos  = i
                if best_pos == -1:
                    break
                c = c[:best_pos] + [256 + best_rank] + c[best_pos + 2:]
            ids.extend(c)
        return ids

    def decode(self, ids):
        raw = b"".join(self._id_to_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        vocab_s = [list(v) for v in self._id_to_bytes.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "vocab": vocab_s}, f)

    @staticmethod
    def from_file(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return BPETokenizer(
            merges=[tuple(m) for m in data["merges"]],
            vocab=data["vocab"],
        )


def _get_stats(ids):
    counts = defaultdict(int)
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def _merge_pair(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text, vocab_size=3072, verbose=True):
    assert vocab_size >= 256
    n_merges = vocab_size - 256
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for i in range(n_merges):
        stats = _get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        new_id = 256 + i
        ids = _merge_pair(ids, pair, new_id)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append(list(pair))
        if verbose and (i % 200 == 0 or i == n_merges - 1):
            print("  merge %d/%d  pair=%s  freq=%d  seq_len=%d" % (
                i+1, n_merges, pair, stats[pair], len(ids)), flush=True)
    vocab_list = [list(vocab[i]) for i in range(len(vocab))]
    return BPETokenizer(merges=merges, vocab=vocab_list)


def load(path=None):
    """Return the tokenizer. Loads bpe_vocab.json if it exists."""
    if os.path.exists(_VOCAB_FILE):
        return BPETokenizer.from_file(_VOCAB_FILE)
    print("[tokenizer] WARNING: bpe_vocab.json not found, using byte fallback",
          file=sys.stderr)
    return _ByteFallback()


class _ByteFallback:
    vocab_size = 256
    def encode(self, text): return list(text.encode("utf-8"))
    def decode(self, ids): return bytes(ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=3072)
    ap.add_argument("--out", default=_VOCAB_FILE)
    args = ap.parse_args()
    print("Training BPE vocab_size=%d on %s ..." % (args.vocab_size, args.data))
    text = open(args.data, encoding="utf-8").read()
    tok = train_bpe(text, vocab_size=args.vocab_size, verbose=True)
    sample = text[:5000]
    ids = tok.encode(sample)
    recovered = tok.decode(ids)
    assert recovered == sample, "ROUND-TRIP FAILED!"
    print("Round-trip OK. %d chars -> %d tokens (%.3f tok/byte)" % (
        len(sample), len(ids), len(ids)/len(sample.encode("utf-8"))))
    tok.save(args.out)
    print("Saved vocab (%d tokens) to %s" % (tok.vocab_size, args.out))
