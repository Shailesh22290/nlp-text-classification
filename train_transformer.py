# train_transformer_improved.py
import numpy as np, os, time
from utils import load_csv, save_json
from preprocess import clean_text, tokenize, build_vocab, texts_to_sequences
from rnn_model import init_embeddings, cross_entropy_loss_and_grad
from transformer_model import Transformer
from optim import Adam

# Improved Config
TRAIN_PATH = "Corona_NLP_train.csv"
TEST_PATH  = "Corona_NLP_test.csv"
MAX_LEN = 64  # Increased for better context
MIN_FREQ = 3  # Slightly higher to reduce noise
EMB_DIM = 128  # Increased embedding dimension
D_MODEL = 128  # Must match EMB_DIM
NUM_HEADS = 8  # More heads for better attention
D_FF = 512     # Larger FFN for more capacity
BATCH = 128    # Reduced batch size for better gradient estimates
EPOCHS = 50    # More epochs with early stopping
LR = 5e-4      # Lower learning rate for stability
WARMUP_STEPS = 500  # Learning rate warmup
GRAD_CLIP = 1.0     # Tighter gradient clipping
DROPOUT = 0.1       # Add dropout for regularization
PATIENCE = 8        # Early stopping patience
WEIGHT_DECAY = 1e-4 # L2 regularization
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(42)

# Custom evaluation function with proper error handling
def compute_metrics(y_true, y_pred, num_classes, average='macro'):
    """
    Compute precision, recall, f1 with proper error handling
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0, 0.0, 0.0
    
    # Initialize metrics
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    
    # Calculate per-class metrics
    for class_id in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # Calculate precision, recall, f1
        if tp + fp > 0:
            precision_per_class[class_id] = tp / (tp + fp)
        else:
            precision_per_class[class_id] = 0.0
            
        if tp + fn > 0:
            recall_per_class[class_id] = tp / (tp + fn)
        else:
            recall_per_class[class_id] = 0.0
            
        if precision_per_class[class_id] + recall_per_class[class_id] > 0:
            f1_per_class[class_id] = 2 * precision_per_class[class_id] * recall_per_class[class_id] / (precision_per_class[class_id] + recall_per_class[class_id])
        else:
            f1_per_class[class_id] = 0.0
    
    # Calculate average metrics
    if average == 'macro':
        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)
        f1 = np.mean(f1_per_class)
    elif average == 'micro':
        # Calculate micro-averaged metrics
        all_tp = sum(np.sum((y_true == i) & (y_pred == i)) for i in range(num_classes))
        all_fp = sum(np.sum((y_true != i) & (y_pred == i)) for i in range(num_classes))
        all_fn = sum(np.sum((y_true == i) & (y_pred != i)) for i in range(num_classes))
        
        precision = all_tp / (all_tp + all_fp) if all_tp + all_fp > 0 else 0.0
        recall = all_tp / (all_tp + all_fn) if all_tp + all_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    else:  # weighted average
        support = np.array([np.sum(y_true == i) for i in range(num_classes)])
        total_support = np.sum(support)
        
        if total_support > 0:
            precision = np.sum(precision_per_class * support) / total_support
            recall = np.sum(recall_per_class * support) / total_support
            f1 = np.sum(f1_per_class * support) / total_support
        else:
            precision = recall = f1 = 0.0
    
    return precision, recall, f1

# Calculate accuracy as well
def compute_accuracy(y_true, y_pred):
    """Simple accuracy calculation"""
    if len(y_true) == 0:
        return 0.0
    return np.mean(y_true == y_pred)

# Learning rate scheduler with warmup and decay
class LRScheduler:
    def __init__(self, d_model, warmup_steps=500):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def get_lr(self, base_lr):
        self.step_count += 1
        arg1 = self.step_count ** (-0.5)
        arg2 = self.step_count * (self.warmup_steps ** (-1.5))
        return base_lr * (self.d_model ** (-0.5)) * min(arg1, arg2)

# Improved Adam optimizer with weight decay
class AdamWeightDecay:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
    
    def step(self, params, grads, lr=None):
        if lr is None:
            lr = self.lr
        self.t += 1
        
        for k in params:
            if k in grads:
                # Add weight decay to gradients (except biases and layer norm params)
                g = grads[k].copy()
                if self.weight_decay > 0 and 'b' not in k and 'beta' not in k and 'gamma' not in k:
                    g += self.weight_decay * params[k]
                
                # Adam update
                self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
                
                # Bias correction
                m_hat = self.m[k] / (1 - self.beta1 ** self.t)
                v_hat = self.v[k] / (1 - self.beta2 ** self.t)
                
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Data augmentation - simple word dropout
def word_dropout(sequences, word2idx, dropout_rate=0.1):
    """Randomly replace some words with <UNK> token for regularization"""
    augmented = sequences.copy()
    unk_idx = word2idx.get('<UNK>', 1)
    
    mask = np.random.random(sequences.shape) < dropout_rate
    # Don't dropout padding tokens (assuming 0 is padding)
    mask = mask & (sequences != 0)
    augmented[mask] = unk_idx
    
    return augmented

# Load data
print("Loading data...")
train_texts, train_labels = load_csv(TRAIN_PATH)
test_texts, test_labels = load_csv(TEST_PATH)

# Create validation split from training data (10%)
np.random.seed(42)
n_train = len(train_texts)
val_indices = np.random.choice(n_train, size=int(0.1 * n_train), replace=False)
train_indices = np.setdiff1d(np.arange(n_train), val_indices)

# Split data
val_texts = [train_texts[i] for i in val_indices]
val_labels = [train_labels[i] for i in val_indices]
train_texts = [train_texts[i] for i in train_indices]
train_labels = [train_labels[i] for i in train_indices]

print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")

unique_labels = sorted(set(train_labels))
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}

y_train = np.array([label2id[l] for l in train_labels], dtype=np.int32)
y_val = np.array([label2id[l] for l in val_labels], dtype=np.int32)
y_test = np.array([label2id.get(l, -1) for l in test_labels], dtype=np.int32)

# Preprocess/vocab with improved tokenization
print("Building vocabulary...")
tokenized = [tokenize(clean_text(t)) for t in train_texts]
word2idx = build_vocab(tokenized, min_freq=MIN_FREQ, add_special=True)
vocab_size = len(word2idx)
print("Vocab size:", vocab_size)

X_train = np.array(texts_to_sequences(train_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)
X_val = np.array(texts_to_sequences(val_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)
X_test = np.array(texts_to_sequences(test_texts, word2idx, max_len=MAX_LEN), dtype=np.int32)

# Initialize embeddings with Xavier initialization
def xavier_init_embeddings(vocab_size, emb_dim):
    limit = np.sqrt(6.0 / (vocab_size + emb_dim))
    return np.random.uniform(-limit, limit, (vocab_size, emb_dim)).astype(np.float32)

embeddings = xavier_init_embeddings(vocab_size, EMB_DIM)
model = Transformer(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, 
                   output_dim=len(unique_labels), max_len=MAX_LEN)

# combine params
params = {"Emb": embeddings}
params.update(model.params)

optimizer = AdamWeightDecay(params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = LRScheduler(D_MODEL, WARMUP_STEPS)

# Helper functions
def accumulate_embedding_grads(grad_X_emb, X_batch, vocab_size, emb_dim):
    grad_emb = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    B, T, D = grad_X_emb.shape
    for i in range(B):
        for t in range(T):
            idx = int(X_batch[i, t])
            if 0 <= idx < vocab_size:  # Safety check with proper bounds
                grad_emb[idx] += grad_X_emb[i, t]
    return grad_emb

def predict_on_array(X_arr, batch_size=256):
    preds = []
    for i in range(0, X_arr.shape[0], batch_size):
        xb = X_arr[i:i+batch_size]
        emb = params["Emb"][xb]
        logits, _ = model.forward(emb)
        preds.extend(list(np.argmax(logits, axis=1)))
    return np.array(preds, dtype=np.int32)

def evaluate(X_arr, y_arr, name=""):
    """Improved evaluation function with robust metrics calculation"""
    try:
        y_pred = predict_on_array(X_arr)
        mask = y_arr >= 0  # Filter out invalid labels
        
        if np.sum(mask) == 0:
            print(f"{name} - No valid labels found!")
            return 0.0
        
        y_true_filtered = y_arr[mask]
        y_pred_filtered = y_pred[mask]
        
        # Calculate metrics
        accuracy = compute_accuracy(y_true_filtered, y_pred_filtered)
        precision, recall, f1 = compute_metrics(y_true_filtered, y_pred_filtered, 
                                               len(unique_labels), average='macro')
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return f1
        
    except Exception as e:
        print(f"Error in evaluation for {name}: {e}")
        return 0.0

# Training loop with improvements
print("Starting training...")
N = X_train.shape[0]
indices = np.arange(N)
best_val_f1 = -1.0
patience_counter = 0
train_losses = []
val_f1_scores = []
train_accuracies = []

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    np.random.shuffle(indices)
    total_loss = 0.0
    
    # Training phase
    for i in range(0, N, BATCH):
        batch_idx = indices[i:i+BATCH]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        B = X_batch.shape[0]

        # Apply word dropout for regularization
        if np.random.random() > 0.5:  # 50% chance to apply
            X_batch = word_dropout(X_batch, word2idx, dropout_rate=0.05)

        X_emb = params["Emb"][X_batch]
        logits, cache = model.forward(X_emb)
        loss, dlogits = cross_entropy_loss_and_grad(logits, y_batch)
        total_loss += loss * B

        grads_model, dX_emb = model.backward(dlogits, cache)
        grad_emb = accumulate_embedding_grads(dX_emb, X_batch, vocab_size, EMB_DIM)

        # Build gradients dict
        grads = {"Emb": grad_emb}
        for k in grads_model:
            grads[k] = grads_model[k]
        
        # Gradient clipping
        total_norm = 0
        for k in grads:
            param_norm = np.linalg.norm(grads[k])
            total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)
        
        if total_norm > GRAD_CLIP:
            for k in grads:
                grads[k] *= GRAD_CLIP / total_norm

        # Update with learning rate scheduling
        current_lr = scheduler.get_lr(LR)
        optimizer.step(params, grads, lr=current_lr)

    avg_loss = total_loss / N
    train_losses.append(avg_loss)
    t1 = time.time()
    
    # Validation
    val_f1 = evaluate(X_val, y_val, "Val")
    val_f1_scores.append(val_f1)
    
    # Calculate training accuracy periodically (every 5 epochs to save time)
    if epoch % 5 == 0 or epoch == 1:
        train_f1 = evaluate(X_train[:1000], y_train[:1000], "Train (sample)")  # Sample for speed
    
    print(f"Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.4f}, lr: {current_lr:.6f}, time: {t1-t0:.1f}s, val_F1: {val_f1:.4f}")

    # Save best model and early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        print(f"New best validation F1: {val_f1:.4f}. Saving model...")
        
        save_dict = {"Emb": params["Emb"]}
        for k in model.params:
            save_dict[k] = params[k]
        np.savez(os.path.join(SAVE_DIR, "transformer_best.npz"), **save_dict)
        save_json(word2idx, os.path.join(SAVE_DIR, "word2idx.json"))
        save_json(label2id, os.path.join(SAVE_DIR, "label2id.json"))
        
        # Test on best validation model
        test_f1 = evaluate(X_test, y_test, "Test")
        
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping! No improvement for {PATIENCE} epochs.")
            break

print(f"\nTraining finished!")
print(f"Best validation F1: {best_val_f1:.4f}")

# Final evaluation on test set with best model
print("\n=== Final Test Evaluation ===")
# Load best model
try:
    best_model = np.load(os.path.join(SAVE_DIR, "transformer_best.npz"))
    for k in best_model.files:
        if k in params:
            params[k] = best_model[k]
    
    final_test_f1 = evaluate(X_test, y_test, "Final Test")
    print(f"Final Test F1: {final_test_f1:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': float(best_val_f1),
        'final_test_f1': float(final_test_f1)
    }
    save_json(history, os.path.join(SAVE_DIR, "training_history.json"))
    
except Exception as e:
    print(f"Error loading best model: {e}")
    print("Using current model for final evaluation...")
    final_test_f1 = evaluate(X_test, y_test, "Final Test")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': float(best_val_f1),
        'final_test_f1': float(final_test_f1)
    }
    save_json(history, os.path.join(SAVE_DIR, "training_history.json"))