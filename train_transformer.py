# train_transformer.py
import os, time, numpy as np
from utils import load_csv, save_json, compute_precision_recall_f1
from preprocess import clean_text, tokenize, build_vocab, texts_to_sequences
from rnn_model import init_embeddings, cross_entropy_loss_and_grad
from transformer_model import Transformer
from optim import Adam

# Config
TRAIN_PATH = "Corona_NLP_train.csv"
TEST_PATH  = "Corona_NLP_test.csv"
MAX_LEN = 50
MIN_FREQ = 2
EMB_DIM = 100
D_MODEL = 100
NUM_HEADS = 5
D_FF = 256
BATCH = 64
EPOCHS = 3
LR = 1e-3
GRAD_CLIP = 5.0
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(42)

def accumulate_embedding_grads(grad_X_emb,X_batch,vocab_size,emb_dim):
    grad_emb = np.zeros((vocab_size,emb_dim),dtype=np.float32)
    B,T,D = grad_X_emb.shape
    for i in range(B):
        for t in range(T):
            grad_emb[X_batch[i,t]] += grad_X_emb[i,t]
    return grad_emb

def predict_on_array(params,model,X_arr):
    preds=[]
    for i in range(0,X_arr.shape[0],128):
        xb=X_arr[i:i+128]
        emb=params["Emb"][xb]
        logits,_=model.forward(emb)
        preds.extend(np.argmax(logits,axis=1))
    return np.array(preds)

# Load data
train_texts, train_labels = load_csv(TRAIN_PATH)
test_texts, test_labels = load_csv(TEST_PATH)
unique_labels = sorted(set(train_labels))
label2id = {lab:i for i,lab in enumerate(unique_labels)}
id2label = {i:lab for lab,i in label2id.items()}
y_train = np.array([label2id[l] for l in train_labels],dtype=np.int32)
y_test  = np.array([label2id.get(l,-1) for l in test_labels],dtype=np.int32)

# Vocab + sequences
tokenized=[tokenize(clean_text(t)) for t in train_texts]
word2idx=build_vocab(tokenized,min_freq=MIN_FREQ,add_special=True)
vocab_size=len(word2idx)
print("Vocab size:",vocab_size)
X_train=np.array(texts_to_sequences(train_texts,word2idx,max_len=MAX_LEN),dtype=np.int32)
X_test=np.array(texts_to_sequences(test_texts,word2idx,max_len=MAX_LEN),dtype=np.int32)

# Model
embeddings=init_embeddings(vocab_size,EMB_DIM)
model=Transformer(d_model=D_MODEL,num_heads=NUM_HEADS,d_ff=D_FF,output_dim=len(unique_labels),max_len=MAX_LEN)
params={"Emb":embeddings}
params.update(model.params)
optimizer=Adam(params,lr=LR)

# Training
N=X_train.shape[0]
indices=np.arange(N)
best_f1=-1.0
for epoch in range(1,EPOCHS+1):
    t0=time.time(); np.random.shuffle(indices); total_loss=0.0
    for i in range(0,N,BATCH):
        b=indices[i:i+BATCH]
        Xb,yb=X_train[b],y_train[b]
        emb=params["Emb"][Xb]
        logits,cache=model.forward(emb)
        loss,dlogits=cross_entropy_loss_and_grad(logits,yb)
        total_loss+=loss*len(b)
        grads_model,dX_emb=model.backward(dlogits,cache)
        grad_emb=accumulate_embedding_grads(dX_emb,Xb,vocab_size,EMB_DIM)
        grads={"Emb":grad_emb}; grads.update(grads_model)
        for k in grads: np.clip(grads[k],-GRAD_CLIP,GRAD_CLIP,out=grads[k])
        optimizer.step(params,grads)
    avg_loss=total_loss/N; t1=time.time()
    y_pred=predict_on_array(params,model,X_test); mask=y_test>=0
    _,macro_f1=compute_precision_recall_f1(y_test[mask],y_pred[mask],len(unique_labels))
    print(f"Epoch {epoch}/{EPOCHS} loss={avg_loss:.4f} time={t1-t0:.1f}s macroF1={macro_f1:.4f}")
    if macro_f1>best_f1:
        best_f1=macro_f1; print("Saving best model.")
        save_dict={"Emb":params["Emb"]}
        for k in model.params: save_dict[k]=params[k]
        np.savez(os.path.join(SAVE_DIR,"transformer_best.npz"),**save_dict)
        save_json(word2idx,os.path.join(SAVE_DIR,"word2idx.json"))
        save_json(label2id,os.path.join(SAVE_DIR,"label2id.json"))
print("Training finished. Best F1:",best_f1)
