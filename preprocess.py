import re
from collections import Counter

def clean_text(s):
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s!?.,'\"-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s):

    return s.split()

def build_vocab(token_lists, min_freq=2, add_special=True):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    word2idx = {}
    idx = 0
    if add_special:
        word2idx["<PAD>"] = idx; idx += 1
        word2idx["<UNK>"] = idx; idx += 1
    for w, freq in counter.items():
        if freq >= min_freq:
            word2idx[w] = idx
            idx += 1
    return word2idx

def texts_to_sequences(texts, word2idx, max_len=50):
    seqs = []
    unk = word2idx.get("<UNK>", 1)
    pad = word2idx.get("<PAD>", 0)
    for t in texts:
        toks = tokenize(clean_text(t))
        idxs = [word2idx.get(tok, unk) for tok in toks][:max_len]
        if len(idxs) < max_len:
            idxs = idxs + [pad] * (max_len - len(idxs))
        seqs.append(idxs)
    return seqs 
