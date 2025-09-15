# train_transformer_v2.py
import os, time, numpy as np
from preprocess import clean_text, tokenize, build_vocab, texts_to_sequences
from utils import load_csv, save_json, compute_precision_recall_f1
from transformer_model_v2 import TransformerEncoder
from optim import Adam  # your numpy Adam

# Config (tune)
TRAIN_PATH = "Corona_NLP_train.csv"
TEST_PATH  = "Corona_NLP_test.csv"
MAX_LEN = 50
MIN_FREQ = 2
EMB_DIM = 100     # input embedding dim
D_MODEL = 256     # model hidden dim (must be divisible by n_heads)
NUM_HEADS = 8
D_FF = 512
BATCH = 64
EPOCHS =10
LR = 1e-3
GRAD_CLIP = 5.0
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(42)

# Load data
train_texts, train_labels = load_csv(TRAIN_PATH)
test_texts, test_labels = load_csv(TEST_PATH)

unique_labels = sorted(set(train_labels))
label2id = {lab:i for i,lab in enumerate(unique_labels)}
id2label = {i:lab for lab,i in label2id.items()}

y_train = np.array([label2id[l] for l in train_labels], dtype=np.int32)
y_test = np.array([label2id.get(l,-1) for l in test_labels], dtype=np.int32)

# Preprocess / vocab
tokenized = [tokenize(clean_text(t)) for t in train_texts]
word2idx = build_vocab(tokenized, min_freq=MIN_FREQ, add_special=True)
vocab_size = len(word2idx)
print("Vocab size:", vocab_size)

X_train = np.array(texts_to_sequences(train_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)
X_test = np.array(texts_to_sequences(test_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)

# Model
model = TransformerEncoder(vocab_size=vocab_size, embed_dim=EMB_DIM, d_model=D_MODEL,
                           n_heads=NUM_HEADS, d_ff=D_FF, n_classes=len(unique_labels),
                           max_len=MAX_LEN, seed=42)

# params dict for optimizer (flatten model attrs)
params = {
    "Emb": model.Emb, "W_proj": model.W_proj,
    "Wq": model.Wq, "Wk": model.Wk, "Wv": model.Wv, "Wo": model.Wo,
    "W1": model.W1, "b1": model.b1, "W2": model.W2, "b2": model.b2,
    "gamma_attn": model.gamma_attn, "beta_attn": model.beta_attn,
    "gamma_ff": model.gamma_ff, "beta_ff": model.beta_ff,
    "W_out": model.W_out, "b_out": model.b_out
}

optimizer = Adam(params, lr=LR)

def accumulate_embedding_grads(grad_Emb_updates):
    # grad_Emb_updates already aggregated in grads["Emb"] returned by backward
    return grad_Emb_updates

# Training loop
N = X_train.shape[0]
indices = np.arange(N)
best_f1 = -1.0

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    np.random.shuffle(indices)
    total_loss = 0.0
    for i in range(0, N, BATCH):
        batch_idx = indices[i:i+BATCH]
        Xb = X_train[batch_idx]    # (B,T)
        yb = y_train[batch_idx]

        logits, cache = model.forward(Xb)
        # cross-entropy and dlogits
        # stable softmax
        logits2 = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits2)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        B = logits.shape[0]
        one_hot = np.zeros_like(probs); one_hot[np.arange(B), yb] = 1
        loss = -np.sum(one_hot * np.log(probs + 1e-12)) / B
        dlogits = (probs - one_hot) / B

        total_loss += loss * Xb.shape[0]

        grads_model, grad_emb_updates = model.backward(dlogits, cache)

        # build grads dict for optimizer
        grads = {}
        grads["Emb"] = grad_emb_updates
        # copy other grads
        for k in ["W_proj","Wq","Wk","Wv","Wo","W1","b1","W2","b2","gamma_attn","beta_attn","gamma_ff","beta_ff","W_out","b_out"]:
            grads[k] = grads_model.get(k, np.zeros_like(params[k]))

        # clip
        for k in grads:
            np.clip(grads[k], -GRAD_CLIP, GRAD_CLIP, out=grads[k])

        optimizer.step(params, grads)

        # sync params back to model object for next forward
        model.Emb = params["Emb"]
        model.W_proj = params["W_proj"]
        model.Wq = params["Wq"]; model.Wk = params["Wk"]; model.Wv = params["Wv"]; model.Wo = params["Wo"]
        model.W1 = params["W1"]; model.b1 = params["b1"]; model.W2 = params["W2"]; model.b2 = params["b2"]
        model.gamma_attn = params["gamma_attn"]; model.beta_attn = params["beta_attn"]
        model.gamma_ff = params["gamma_ff"]; model.beta_ff = params["beta_ff"]
        model.W_out = params["W_out"]; model.b_out = params["b_out"]

    avg_loss = total_loss / N
    t1 = time.time()

    # evaluate
    # batched inference
    preds = []
    for j in range(0, X_test.shape[0], 256):
        xb = X_test[j:j+256]
        logits, _ = model.forward(xb)
        preds.extend(list(np.argmax(logits, axis=1)))
    preds = np.array(preds, dtype=np.int32)
    mask = y_test >= 0
    _, macro_f1 = compute_precision_recall_f1(y_test[mask], preds[mask], num_classes=len(unique_labels))

    print(f"Epoch {epoch}/{EPOCHS} loss={avg_loss:.4f} time={t1-t0:.1f}s macroF1={macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        print("Saving best model.")
        save_dict = {"Emb": params["Emb"]}
        for k in ["W_proj","Wq","Wk","Wv","Wo","W1","b1","W2","b2","gamma_attn","beta_attn","gamma_ff","beta_ff","W_out","b_out"]:
            save_dict[k] = params[k]
        np.savez(os.path.join(SAVE_DIR, "transformer_best_v2.npz"), **save_dict)
        save_json(word2idx, os.path.join(SAVE_DIR, "word2idx.json"))
        save_json(label2id, os.path.join(SAVE_DIR, "label2id.json"))

print("Training finished. Best F1:", best_f1)
