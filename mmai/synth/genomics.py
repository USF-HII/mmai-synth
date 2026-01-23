import random
from collections import defaultdict, Counter

def _train_markov_k(seq_list, k=5):
    """Train an order-k Markov model over A/C/G/T/N."""
    starts = Counter()
    trans = defaultdict(Counter)
    for s in seq_list:
        s = s.upper()
        if len(s) < k+1: continue
        starts[s[:k]] += 1
        for i in range(len(s)-k):
            trans[s[i:i+k]][s[i+k]] += 1
    return starts, trans

def _sample_markov_k(starts, trans, length, rng):
    k = len(next(iter(starts))) if starts else 1
    prefix = rng.choices(list(starts.keys()), weights=list(starts.values()))[0]
    out = [c for c in prefix]
    while len(out) < length:
        state = "".join(out[-k:])
        nxts = trans.get(state)
        if not nxts:
            # restart
            state = rng.choice(list(trans.keys()))
            nxts = trans[state]
        ch = rng.choices(list(nxts.keys()), weights=list(nxts.values()))[0]
        out.append(ch)
    return "".join(out)

def synthesize_fastq_baseline(records, num_reads=None, seed=42):
    """
    Baseline FASTQ synthesizer:
      - learns read length distribution and mean quality distribution
      - trains an order-5 nucleotide Markov model
      - samples new sequences/qualities accordingly
    """
    rng = random.Random(seed)
    seqs, lens, quals = [], [], []
    for _, seq, qual in records:
        seqs.append(seq)
        lens.append(len(seq))
        quals.append(sum(qual)/len(qual))
    if not seqs:
        return []

    starts, trans = _train_markov_k(seqs, k=5)
    from statistics import mean
    L = lens
    Q = quals
    n = num_reads or len(seqs)
    out = []
    for i in range(n):
        length = rng.choice(L)
        seq = _sample_markov_k(starts, trans, length, rng)
        # Simple quality sampler: Normal around mean with fixed stdev, clipped to [5, 40]
        mu = rng.choice(Q)
        q = [max(5, min(40, int(rng.gauss(mu, 3)))) for _ in range(length)]
        out.append((f"syn_read_{i}", seq, q))
    return out
