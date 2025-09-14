import numpy as np
from rnn_model import softmax, cross_entropy_loss_and_grad

# ---------------------------
# Utilities
# ---------------------------
def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def layer_norm_forward(x, gamma, beta, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_norm = (x - mean) * inv
    out = gamma * x_norm + beta
    cache = (x_norm, mean, var, gamma, eps)
    return out, cache

def layer_norm_backward(dout, cache):
    x_norm, mean, var, gamma, eps = cache
    D = x_norm.shape[-1]

    dgamma = np.sum(dout * x_norm, axis=tuple(range(dout.ndim - 1)))
    dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))

    L = int(np.prod(dout.shape[:-1]))
    dxnorm2d = (dout * gamma).reshape(L, D)
    xnorm2d = x_norm.reshape(L, D)

    N = D
    sum_dxnorm = np.sum(dxnorm2d, axis=1, keepdims=True)
    sum_dxnorm_xnorm = np.sum(dxnorm2d * xnorm2d, axis=1, keepdims=True)

    dx2d = (dxnorm2d - sum_dxnorm / N - xnorm2d * (sum_dxnorm_xnorm / N)) / np.sqrt(var.reshape(-1,1) + eps)
    dx = dx2d.reshape(*dout.shape)
    return dx, dgamma, dbeta

def split_heads(x, num_heads):
    B, T, D = x.shape
    d_k = D // num_heads
    return x.reshape(B, T, num_heads, d_k).transpose(0, 2, 1, 3)

def combine_heads(x):
    B,h,T,d_k = x.shape
    return x.transpose(0,2,1,3).reshape(B, T, h*d_k)

def scaled_dot_product_attention_forward(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0,1,3,2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    B,h,T,_ = scores.shape
    scores2 = scores.reshape(B*h*T, T)
    scores2 = scores2 - np.max(scores2, axis=1, keepdims=True)
    weights = np.exp(scores2)
    weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-12)
    weights = weights.reshape(B,h,T,T)
    out = weights @ V
    cache = (Q, K, V, weights)
    return out, weights, cache

def scaled_dot_product_attention_backward(dout, cache):
    Q, K, V, weights = cache
    dWeights = np.einsum('bhik,bhjk->bhij', dout, V)
    dV = np.einsum('bhij,bhik->bhjk', weights, dout)
    dScores = np.zeros_like(dWeights)
    B,h,T,_ = dWeights.shape
    for b in range(B):
        for hh in range(h):
            W = weights[b,hh]
            dW = dWeights[b,hh]
            for i in range(T):
                w = W[i]
                dw = dW[i]
                s = np.dot(dw, w)
                dScores[b,hh,i,:] = w * (dw - s)
    inv_sqrt = 1.0 / np.sqrt(Q.shape[-1])
    dQ = np.einsum('bhij,bhjk->bhik', dScores, K) * inv_sqrt
    dK = np.einsum('bhij,bhik->bhjk', dScores, Q) * inv_sqrt
    return dQ, dK, dV

# ---------------------------
# Transformer Encoder Layer
# ---------------------------
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, seed=42):
        rng = np.random.RandomState(seed)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Attention weights
        self.Wq = rng.randn(d_model, d_model).astype(np.float32)*0.01
        self.Wk = rng.randn(d_model, d_model).astype(np.float32)*0.01
        self.Wv = rng.randn(d_model, d_model).astype(np.float32)*0.01
        self.Wo = rng.randn(d_model, d_model).astype(np.float32)*0.01
        # Feed-forward
        self.W1 = rng.randn(d_model, d_ff).astype(np.float32)*0.01
        self.b1 = np.zeros((d_ff,),dtype=np.float32)
        self.W2 = rng.randn(d_ff, d_model).astype(np.float32)*0.01
        self.b2 = np.zeros((d_model,),dtype=np.float32)
        # Layer norms
        self.gamma1 = np.ones((d_model,),dtype=np.float32)
        self.beta1 = np.zeros((d_model,),dtype=np.float32)
        self.gamma2 = np.ones((d_model,),dtype=np.float32)
        self.beta2 = np.zeros((d_model,),dtype=np.float32)

    def forward(self, X, mask=None):
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        Qh, Kh, Vh = split_heads(Q,self.num_heads), split_heads(K,self.num_heads), split_heads(V,self.num_heads)
        attn_out, weights, attn_cache = scaled_dot_product_attention_forward(Qh,Kh,Vh,mask)
        attn_comb = combine_heads(attn_out) @ self.Wo
        resid1 = X + attn_comb
        ln1, ln1_cache = layer_norm_forward(resid1,self.gamma1,self.beta1)
        ff1 = ln1 @ self.W1 + self.b1
        ff1_relu = np.maximum(0, ff1)
        ff2 = ff1_relu @ self.W2 + self.b2
        resid2 = ln1 + ff2
        ln2, ln2_cache = layer_norm_forward(resid2,self.gamma2,self.beta2)
        cache = {"X":X,"Qh":Qh,"Kh":Kh,"Vh":Vh,"attn_cache":attn_cache,"attn_out":attn_out,
                 "attn_comb":attn_comb,"resid1":resid1,"ln1":ln1,"ln1_cache":ln1_cache,
                 "ff1":ff1,"ff1_relu":ff1_relu,"ff2":ff2,"resid2":resid2,"ln2_cache":ln2_cache}
        return ln2, cache

    def backward(self, dln2, cache):
        # --- LayerNorm 2 backward ---
        dresid2, dgamma2, dbeta2 = layer_norm_backward(dln2, cache["ln2_cache"])

        # --- Feed-forward block ---
        dff2 = dresid2.copy()
        dW2 = cache["ff1_relu"].reshape(-1, cache["ff1_relu"].shape[-1]).T @ dff2.reshape(-1, dff2.shape[-1])
        db2 = np.sum(dff2, axis=(0,1))
        dff1_relu = dff2 @ self.W2.T
        dff1 = dff1_relu * (cache["ff1"] > 0)
        dW1 = cache["ln1"].reshape(-1, cache["ln1"].shape[-1]).T @ dff1.reshape(-1, dff1.shape[-1])
        db1 = np.sum(dff1, axis=(0,1))
        dln1 = dresid2 + (dff1 @ self.W1.T)

        # --- LayerNorm 1 backward ---
        dresid1, dgamma1, dbeta1 = layer_norm_backward(dln1, cache["ln1_cache"])

        # --- Attention block ---
        dX = dresid1.copy()
        dWo = combine_heads(cache["attn_out"]).reshape(-1,self.d_model).T @ dresid1.reshape(-1,self.d_model)
        d_attn_comb = dresid1 @ self.Wo.T
        d_attn_out = d_attn_comb.reshape(cache["X"].shape[0], cache["X"].shape[1], self.num_heads, self.d_k).transpose(0,2,1,3)
        dQh,dKh,dVh = scaled_dot_product_attention_backward(d_attn_out, cache["attn_cache"])
        dQ, dK, dV = combine_heads(dQh), combine_heads(dKh), combine_heads(dVh)
        dWq = cache["X"].reshape(-1,self.d_model).T @ dQ.reshape(-1,self.d_model)
        dWk = cache["X"].reshape(-1,self.d_model).T @ dK.reshape(-1,self.d_model)
        dWv = cache["X"].reshape(-1,self.d_model).T @ dV.reshape(-1,self.d_model)
        dX += dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T

        grads = {
            "Wq":dWq,"Wk":dWk,"Wv":dWv,"Wo":dWo,
            "W1":dW1,"b1":db1,"W2":dW2,"b2":db2,
            "gamma1":dgamma1,"beta1":dbeta1,
            "gamma2":dgamma2,"beta2":dbeta2
        }
        return grads, dX

# ---------------------------
# Transformer (with classifier)
# ---------------------------
class Transformer:
    def __init__(self, d_model=100,num_heads=5,d_ff=256,output_dim=3,max_len=50,seed=42):
        self.encoder = TransformerEncoderLayer(d_model,num_heads,d_ff,seed=seed)
        rng = np.random.RandomState(seed+1)
        self.params = {
            "Wq":self.encoder.Wq,"Wk":self.encoder.Wk,"Wv":self.encoder.Wv,"Wo":self.encoder.Wo,
            "W1":self.encoder.W1,"b1":self.encoder.b1,"W2":self.encoder.W2,"b2":self.encoder.b2,
            "gamma1":self.encoder.gamma1,"beta1":self.encoder.beta1,
            "gamma2":self.encoder.gamma2,"beta2":self.encoder.beta2,
            "W_out": rng.randn(d_model,output_dim).astype(np.float32)*0.01,
            "b_out": np.zeros((output_dim,),dtype=np.float32)
        }
        self.positional = positional_encoding(max_len,d_model)

    def forward(self,X_emb):
        B,T,D = X_emb.shape
        X = X_emb + self.positional[:T]
        enc_out, cache_enc = self.encoder.forward(X)
        pooled = np.mean(enc_out, axis=1)
        logits = pooled @ self.params["W_out"] + self.params["b_out"]
        cache = {"enc_cache":cache_enc,"pooled":pooled,"X_emb":X_emb}
        return logits, cache

    def backward(self,dlogits,cache):
        pooled = cache["pooled"]
        dW_out = pooled.T @ dlogits
        db_out = np.sum(dlogits, axis=0)
        dpooled = dlogits @ self.params["W_out"].T
        denc = np.repeat(dpooled[:,np.newaxis,:], cache["enc_cache"]["X"].shape[1], axis=1) / cache["enc_cache"]["X"].shape[1]
        grads_enc,dX = self.encoder.backward(denc, cache["enc_cache"])
        grads = {"W_out":dW_out,"b_out":db_out}
        grads.update(grads_enc)
        return grads, dX
