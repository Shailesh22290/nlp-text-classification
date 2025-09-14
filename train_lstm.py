# train_lstm.py
import numpy as np, os, time
from utils import load_csv, save_json, compute_precision_recall_f1
from preprocess import clean_text, tokenize, build_vocab, texts_to_sequences
from rnn_model import init_embeddings, cross_entropy_loss_and_grad
from lstm_model import LSTM
from optim import Adam

# ---------- Config ----------
TRAIN_PATH = "Corona_NLP_train.csv"
TEST_PATH  = "Corona_NLP_test.csv"
MAX_LEN = 50
MIN_FREQ = 2
EMB_DIM = 100
HIDDEN = 128
BATCH = 64
EPOCHS = 25
LR = 1e-3
GRAD_CLIP = 5.0
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1) load data
train_texts, train_labels = load_csv(TRAIN_PATH)
test_texts, test_labels = load_csv(TEST_PATH)

unique_labels = sorted(set(train_labels))
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}

y = np.array([label2id[l] for l in train_labels], dtype=np.int32)
y_test = np.array([label2id.get(l, -1) for l in test_labels], dtype=np.int32)

tokenized = [tokenize(clean_text(t)) for t in train_texts]
word2idx = build_vocab(tokenized, min_freq=MIN_FREQ, add_special=True)
vocab_size = len(word2idx)

X = np.array(texts_to_sequences(train_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)
X_test = np.array(texts_to_sequences(test_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)

# embeddings + model
embeddings = init_embeddings(vocab_size, EMB_DIM)
model = LSTM(input_dim=EMB_DIM, hidden_dim=HIDDEN, output_dim=len(unique_labels))
params = {"Emb": embeddings}
params.update(model.params)
optimizer = Adam(params, lr=LR)

def accumulate_embedding_grads(grad_X_emb, X_batch, vocab_size, emb_dim):
    grad_emb = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    B, T, D = grad_X_emb.shape
    for i in range(B):
        for t in range(T):
            grad_emb[X_batch[i,t]] += grad_X_emb[i,t]
    return grad_emb

def predict_on_array(X_arr):
    preds = []
    for i in range(0, len(X_arr), 128):
        xb = X_arr[i:i+128]
        emb = params["Emb"][xb]
        logits, _ = model.forward(emb)
        preds.extend(list(np.argmax(logits, axis=1)))
    return np.array(preds)

# training
best_f1 = -1
for epoch in range(1, EPOCHS+1):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    total_loss = 0
    for i in range(0, len(X), BATCH):
        b = idxs[i:i+BATCH]
        Xb, yb = X[b], y[b]
        emb = params["Emb"][Xb]
        logits, cache = model.forward(emb)
        loss, dlogits = cross_entropy_loss_and_grad(logits, yb)
        total_loss += loss * len(b)
        grads_model, dX_emb = model.backward(dlogits, cache)
        grad_emb = accumulate_embedding_grads(dX_emb, Xb, vocab_size, EMB_DIM)
        grads = {"Emb": grad_emb}
        grads.update(grads_model)
        for k in grads:
            np.clip(grads[k], -GRAD_CLIP, GRAD_CLIP, out=grads[k])
        optimizer.step(params, grads)
    avg_loss = total_loss / len(X)
    y_pred = predict_on_array(X_test)
    mask = y_test >= 0
    _, macro_f1 = compute_precision_recall_f1(y_test[mask], y_pred[mask], len(unique_labels))
    print(f"Epoch {epoch} loss={avg_loss:.4f}, test_macroF1={macro_f1:.4f}")
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        np.savez(os.path.join(SAVE_DIR, "lstm_best.npz"), **params)
        save_json(word2idx, os.path.join(SAVE_DIR, "word2idx.json"))
        save_json(label2id, os.path.join(SAVE_DIR, "label2id.json"))
        print("Saved new best model.")

print("Training done. Best F1:", best_f1)
