# train_transformer.py
import numpy as np, os, time
from utils import load_csv, save_json, compute_precision_recall_f1
from preprocess import clean_text, tokenize, build_vocab, texts_to_sequences
from rnn_model import init_embeddings, cross_entropy_loss_and_grad
from transformer_model import Transformer
from optim import Adam

# Config (tune as needed)
TRAIN_PATH = "dataset/Corona_NLP_train.csv"
TEST_PATH  = "dataset/Corona_NLP_test.csv"
MAX_LEN = 50
MIN_FREQ = 2
EMB_DIM = 100   # must equal transformer's d_model
D_MODEL = 100
NUM_HEADS = 5
D_FF = 256
BATCH = 256
EPOCHS = 25
LR = 1e-3
GRAD_CLIP = 5.0
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(42)

# Load data
train_texts, train_labels = load_csv(TRAIN_PATH)
test_texts, test_labels = load_csv(TEST_PATH)

unique_labels = sorted(set(train_labels))
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}

y = np.array([label2id[l] for l in train_labels], dtype=np.int32)
y_test = np.array([label2id.get(l, -1) for l in test_labels], dtype=np.int32)

# Preprocess/vocab
tokenized = [tokenize(clean_text(t)) for t in train_texts]
word2idx = build_vocab(tokenized, min_freq=MIN_FREQ, add_special=True)
vocab_size = len(word2idx)
print("Vocab size:", vocab_size)

X = np.array(texts_to_sequences(train_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)
X_test = np.array(texts_to_sequences(test_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)

# Embeddings and model
embeddings = init_embeddings(vocab_size, EMB_DIM)
model = Transformer(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, output_dim=len(unique_labels), max_len=MAX_LEN)

# combine params (embedding + transformer params)
params = {"Emb": embeddings}
params.update(model.params)

optimizer = Adam(params, lr=LR)

# helper: accumulate embedding grads
def accumulate_embedding_grads(grad_X_emb, X_batch, vocab_size, emb_dim):
    grad_emb = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    B, T, D = grad_X_emb.shape
    for i in range(B):
        for t in range(T):
            idx = int(X_batch[i, t])
            grad_emb[idx] += grad_X_emb[i, t]
    return grad_emb

def predict_on_array(X_arr):
    preds = []
    for i in range(0, X_arr.shape[0], 256):
        xb = X_arr[i:i+256]
        emb = params["Emb"][xb]
        logits, _ = model.forward(emb)
        preds.extend(list(np.argmax(logits, axis=1)))
    return np.array(preds, dtype=np.int32)

# Training loop
N = X.shape[0]
indices = np.arange(N)
best_f1 = -1.0
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    np.random.shuffle(indices)
    total_loss = 0.0
    for i in range(0, N, BATCH):
        batch_idx = indices[i:i+BATCH]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        B = X_batch.shape[0]

        X_emb = params["Emb"][X_batch]  # (B, T, D)
        logits, cache = model.forward(X_emb)
        loss, dlogits = cross_entropy_loss_and_grad(logits, y_batch)
        total_loss += loss * B

        grads_model, dX_emb = model.backward(dlogits, cache)
        grad_emb = accumulate_embedding_grads(dX_emb, X_batch, vocab_size, EMB_DIM)

        # build grads dict matching params
        grads = {"Emb": grad_emb}
        # copy transformer grads to grads dict (names must match params keys)
        for k in grads_model:
            grads[k] = grads_model[k]
        # gradient clipping
        for k in grads:
            np.clip(grads[k], -GRAD_CLIP, GRAD_CLIP, out=grads[k])

        optimizer.step(params, grads)

    avg_loss = total_loss / N
    t1 = time.time()
    # evaluate on test set (fast batched inference)
    y_pred = predict_on_array(X_test)
    mask = y_test >= 0
    _, macro_f1 = compute_precision_recall_f1(y_test[mask], y_pred[mask], num_classes=len(unique_labels))
    print(f"Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.4f}  time: {t1-t0:.1f}s  test_macroF1: {macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        print("Saving best model.")
        save_dict = {"Emb": params["Emb"]}
        for k in model.params:
            # note: model.params includes encoder and W_out,b_out
            save_dict[k] = params[k]
        np.savez(os.path.join(SAVE_DIR, "transformer_best.npz"), **save_dict)
        save_json(word2idx, os.path.join(SAVE_DIR, "word2idx.json"))
        save_json(label2id, os.path.join(SAVE_DIR, "label2id.json"))
print("Training finished. Best test macro F1:", best_f1)
