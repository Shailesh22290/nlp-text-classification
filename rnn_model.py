import numpy as np

def init_embeddings(vocab_size, dim=100, seed=42):
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (vocab_size + dim))
    return rng.uniform(-limit, limit, (vocab_size, dim)).astype(np.float32)

def softmax(logits):
    x = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def cross_entropy_loss_and_grad(logits, y_true):
    B = logits.shape[0]
    probs = softmax(logits)
    correct = probs[np.arange(B), y_true]
    loss = -np.mean(np.log(correct + 1e-12))
    dlogits = probs
    dlogits[np.arange(B), y_true] -= 1
    dlogits /= B
    return loss, dlogits

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        self.params = {}
        self.params["Wx"] = (rng.randn(input_dim, hidden_dim) * 0.01).astype(np.float32)
        self.params["Wh"] = (rng.randn(hidden_dim, hidden_dim) * 0.01).astype(np.float32)
        self.params["b"]  = np.zeros((hidden_dim,), dtype=np.float32)
        self.params["W_out"] = (rng.randn(hidden_dim, output_dim) * 0.01).astype(np.float32)
        self.params["b_out"] = np.zeros((output_dim,), dtype=np.float32)

    def forward(self, X_emb):
        B, T, D = X_emb.shape
        H = self.hidden_dim
        Wx = self.params["Wx"]
        Wh = self.params["Wh"]
        b = self.params["b"]

        hs = np.zeros((B, T, H), dtype=np.float32)
        h_prev = np.zeros((B, H), dtype=np.float32)
        pre_act = np.zeros((B, T, H), dtype=np.float32)  
        for t in range(T):
            x_t = X_emb[:, t, :]         
            z_t = x_t.dot(Wx) + h_prev.dot(Wh) + b  
            h_t = np.tanh(z_t)
            pre_act[:, t, :] = z_t
            hs[:, t, :] = h_t
            h_prev = h_t

        h_last = hs[:, -1, :]  
        logits = h_last.dot(self.params["W_out"]) + self.params["b_out"] 

        cache = {
            "X_emb": X_emb,
            "hs": hs,
            "pre_act": pre_act,
        }
        return logits, cache

    def backward(self, dlogits, cache):
        X_emb = cache["X_emb"]
        hs = cache["hs"]
        pre_act = cache["pre_act"]
        B, T, D = X_emb.shape
        H = self.hidden_dim

        dWx = np.zeros_like(self.params["Wx"])
        dWh = np.zeros_like(self.params["Wh"])
        db = np.zeros_like(self.params["b"])
        dW_out = np.zeros_like(self.params["W_out"])
        db_out = np.zeros_like(self.params["b_out"])

        h_last = hs[:, -1, :] 
        dW_out = h_last.T.dot(dlogits) 
        db_out = np.sum(dlogits, axis=0)  

        dh = dlogits.dot(self.params["W_out"].T) 
        dh_next = np.zeros((B, H), dtype=np.float32)
        dX_emb = np.zeros_like(X_emb, dtype=np.float32)  

        for t in reversed(range(T)):
            h_t = hs[:, t, :]       
            h_prev = hs[:, t-1, :] if t-1 >= 0 else np.zeros_like(h_t)
            if t == T-1:
                dh_total = dh + dh_next
            else:
                dh_total = dh_next

 
            dt = dh_total * (1.0 - h_t * h_t) 

   
            x_t = X_emb[:, t, :]  
            dWx += x_t.T.dot(dt)  
            dWh += h_prev.T.dot(dt)  
            db  += np.sum(dt, axis=0)  
   
            dX_emb[:, t, :] = dt.dot(self.params["Wx"].T)  
            dh_next = dt.dot(self.params["Wh"].T)  

        grads = {
            "Wx": dWx,
            "Wh": dWh,
            "b": db,
            "W_out": dW_out,
            "b_out": db_out
        }
        return grads, dX_emb
