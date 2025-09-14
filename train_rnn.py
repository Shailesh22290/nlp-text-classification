# train_rnn.py
import numpy as np
import os
import time
from utils import load_csv, save_json, compute_precision_recall_f1
from preprocess import clean_text, tokenize, build_vocab, texts_to_sequences
from rnn_model import init_embeddings, SimpleRNN, cross_entropy_loss_and_grad
from optim import Adam

# ---------- Config ----------
TRAIN_PATH = "dataset/Corona_NLP_train.csv"
TEST_PATH  = "dataset/Corona_NLP_test.csv"   # optional for final eval
MAX_LEN = 50
MIN_FREQ = 2
EMB_DIM = 100
HIDDEN = 128
BATCH = 64
EPOCHS =1
LR = 1e-3
GRAD_CLIP = 5.0
SEED = 42
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(SEED)

# 1) load raw data
train_texts, train_labels = load_csv(TRAIN_PATH)
test_texts, test_labels = load_csv(TEST_PATH)

# 2) build label mapping from train set
unique_labels = sorted(list(set(train_labels)))
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}

y = np.array([label2id[l] for l in train_labels], dtype=np.int32)
y_test = np.array([label2id.get(l, -1) for l in test_labels], dtype=np.int32)

# 3) tokenize and vocab
tokenized = [tokenize(clean_text(t)) for t in train_texts]
word2idx = build_vocab(tokenized, min_freq=MIN_FREQ, add_special=True)
vocab_size = len(word2idx)
print("Vocab size:", vocab_size)

# 4) convert to sequences
X = np.array(texts_to_sequences(train_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)  # (N, T)
X_test = np.array(texts_to_sequences(test_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)

# 5) init embeddings and model
embeddings = init_embeddings(vocab_size, EMB_DIM, seed=SEED)  # (V, D)
model = SimpleRNN(input_dim=EMB_DIM, hidden_dim=HIDDEN, output_dim=len(unique_labels), seed=SEED)

# 6) combine params (so optimizer can update embeddings too)
params = {"Emb": embeddings}
for k, v in model.params.items():
    params[k] = v

optimizer = Adam(params, lr=LR)

# 7) training loop
N = X.shape[0]
indices = np.arange(N)
best_val_f1 = -1.0

# small helper: convert gradient wrt X_emb into gradient for full embeddings matrix
def accumulate_embedding_grads(grad_X_emb, X_batch, vocab_size, emb_dim):
    grad_emb = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    B, T, D = grad_X_emb.shape
    for i in range(B):
        for t in range(T):
            idx = int(X_batch[i, t])
            grad_emb[idx] += grad_X_emb[i, t]
    return grad_emb

def predict_on_array(X_arr):
    # returns preds
    B_all = 256
    preds = []
    for i in range(0, X_arr.shape[0], B_all):
        xb = X_arr[i:i+B_all]
        emb = params["Emb"][xb]  # (b, T, D)
        logits, _ = model.forward(emb)
        preds.extend(list(np.argmax(logits, axis=1)))
    return np.array(preds, dtype=np.int32)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    np.random.shuffle(indices)
    total_loss = 0.0
    for i in range(0, N, BATCH):
        batch_idx = indices[i:i+BATCH]
        X_batch = X[batch_idx]        # (B, T)
        y_batch = y[batch_idx]        # (B,)
        B = X_batch.shape[0]

        X_emb = params["Emb"][X_batch]  # (B, T, D)
        logits, cache = model.forward(X_emb)
        loss, dlogits = cross_entropy_loss_and_grad(logits, y_batch)
        total_loss += loss * B

        # backward on model to get grads for model params and dX_emb
        grads_model, dX_emb = model.backward(dlogits, cache)

        # convert dX_emb to gradient for embeddings
        grad_emb = accumulate_embedding_grads(dX_emb, X_batch, vocab_size, EMB_DIM)

        # build grads dict matching params keys
        grads = {"Emb": grad_emb}
        for k in grads_model:
            grads[k] = grads_model[k]

        # gradient clipping
        for k in grads:
            np.clip(grads[k], -GRAD_CLIP, GRAD_CLIP, out=grads[k])

        # optimizer step updates params in-place
        optimizer.step(params, grads)

    avg_loss = total_loss / N
    t1 = time.time()

    # validation (here use test set for quick check)
    y_pred = predict_on_array(X_test)
    # filter test labels that might not be in train mapping (-1)
    mask = (y_test >= 0)
    results, macro_f1 = compute_precision_recall_f1(y_test[mask], y_pred[mask], num_classes=len(unique_labels))
    print(f"Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.4f}  time: {t1-t0:.1f}s  test_macroF1: {macro_f1:.4f}")

    # save best
    if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        print("Saving best model (f1 improved).")
        save_path = os.path.join(SAVE_DIR, "rnn_best.npz")
        save_dict = {"Emb": params["Emb"]}
        for k in model.params:
            save_dict[k] = model.params[k]
        np.savez(save_path, **save_dict)
        # save vocab and label maps
        save_json(word2idx, os.path.join(SAVE_DIR, "word2idx.json"))
        save_json(label2id, os.path.join(SAVE_DIR, "label2id.json"))

print("Training finished. Best test macro F1:", best_val_f1)
