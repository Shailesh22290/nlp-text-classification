# transformer_model.py
import numpy as np
from rnn_model import softmax, cross_entropy_loss_and_grad  # reuse cross-entropy softmax

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
    # x: (..., D)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_norm = (x - mean) * inv
    out = gamma * x_norm + beta
    cache = (x_norm, mean, var, inv, gamma, beta, eps)
    return out, cache

def layer_norm_backward(dout, cache):
    x_norm, mean, var, inv, gamma, beta, eps = cache
    D = x_norm.shape[-1]

    # grads gamma/beta
    dgamma = np.sum(dout * x_norm, axis=tuple(range(dout.ndim - 1)))
    dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))

    # reshape to 2D (batch*seq, D)
    L = int(np.prod(dout.shape[:-1]))
    dxnorm2d = (dout * gamma).reshape(L, D)
    xnorm2d = x_norm.reshape(L, D)

    # backprop formula: dx = (1/N) * (N*dx_norm - sum(dx_norm) - x_norm*sum(dx_norm*x_norm))
    N = D
    sum_dxnorm = np.sum(dxnorm2d, axis=1, keepdims=True)  # (L,1)
    sum_dxnorm_xnorm = np.sum(dxnorm2d * xnorm2d, axis=1, keepdims=True)  # (L,1)

    dx2d = (dxnorm2d - sum_dxnorm / N - xnorm2d * (sum_dxnorm_xnorm / N)) * (1.0 / np.sqrt(var.reshape(-1,1) + eps))

    dx = dx2d.reshape(*dout.shape)
    return dx, dgamma, dbeta


def split_heads(x, num_heads):
    # x: (B, T, D) -> (B, num_heads, T, d_k)
    B, T, D = x.shape
    assert D % num_heads == 0
    d_k = D // num_heads
    x = x.reshape(B, T, num_heads, d_k)
    return x.transpose(0, 2, 1, 3)

def combine_heads(x):
    # x: (B, num_heads, T, d_k) -> (B, T, D)
    B, h, T, d_k = x.shape
    x = x.transpose(0, 2, 1, 3).reshape(B, T, h * d_k)
    return x

def scaled_dot_product_attention_forward(Q, K, V, mask=None):
    # Q,K,V: (B, h, T, d_k)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0,1,3,2) / np.sqrt(d_k)  # (B,h,T,T)
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    # softmax along last axis
    # reshape to (B*h*T, T) to use stable softmax
    B,h,T,_ = scores.shape
    scores2 = scores.reshape(B*h*T, T)
    scores2 = scores2 - np.max(scores2, axis=1, keepdims=True)
    weights = np.exp(scores2)
    weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-12)
    weights = weights.reshape(B,h,T,T)
    out = weights @ V  # (B,h,T,d_k)
    cache = (Q, K, V, weights, mask)
    return out, weights, cache

def scaled_dot_product_attention_backward(dout, cache):
    # dout: (B,h,T,d_k) gradients wrt attention output
    Q, K, V, weights, mask = cache
    B,h,T,d_k = Q.shape
    # gradients wrt weights: dW = dout @ V^T  (per head)
    # reshape to 2D for easier math: merge B*h*T rows
    dout2 = dout.reshape(B*h*T, d_k)  # (B*h*T, d_k)
    V2 = V.transpose(0,1,2,3).reshape(B*h*T, d_k)  # WRONG shape approach — simpler compute per head using einsum
    # compute dWeights: for each i,j: dW[:,:,i,j] = sum_k dout[:,:,i,k] * V[:,:,j,k]
    # Use einsum:
    dWeights = np.einsum('bhik,bhjk->bhij', dout, V)  # (B,h,T,T)
    # Now backprop through softmax: weights = softmax(scores)
    # dScores = Jacobian_softmax * dWeights
    # For each row (size T): dS_row = diag(w) - w w^T ; multiply by dW_row
    dScores = np.zeros_like(dWeights)
    for b in range(B):
        for hh in range(h):
            Wmat = weights[b, hh]  # (T,T)
            dWmat = dWeights[b, hh]  # (T,T)
            # operate row-wise
            for i in range(T):
                w = Wmat[i]  # (T,)
                dw = dWmat[i]  # (T,)
                # jacobian times dw
                # dscore = w * (dw - sum(dw * w))
                s = np.dot(dw, w)
                dScores[b, hh, i, :] = w * (dw - s)
    # Now dScores corresponds to derivatives wrt scores (QK^T / sqrt(dk))
    inv_sqrt = 1.0 / np.sqrt(d_k)
    # dScores -> dQ and dK
    # scores = Q @ K^T * inv_sqrt ; so
    dQ = np.einsum('bhij,bhjk->bhik', dScores, K) * inv_sqrt
    dK = np.einsum('bhij,bhik->bhjk', dScores, Q) * inv_sqrt
    # dV from dout: dout @ weights^T
    dV = np.einsum('bhik,bhij->bhjk', dout, weights)
    return dQ, dK, dV

# ---------------------------
# Transformer single encoder (one layer)
# ---------------------------
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, seed=42):
        rng = np.random.RandomState(seed)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        # attention projections
        self.Wq = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wk = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wv = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wo = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        # FFN
        self.W1 = rng.randn(d_model, d_ff).astype(np.float32) * 0.01
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = rng.randn(d_ff, d_model).astype(np.float32) * 0.01
        self.b2 = np.zeros((d_model,), dtype=np.float32)
        # LayerNorm params
        self.gamma1 = np.ones((d_model,), dtype=np.float32)
        self.beta1 = np.zeros((d_model,), dtype=np.float32)
        self.gamma2 = np.ones((d_model,), dtype=np.float32)
        self.beta2 = np.zeros((d_model,), dtype=np.float32)

    def forward(self, X, mask=None):
        # X: (B, T, d_model)
        B, T, D = X.shape
        # --- Multi-head attention ---
        Q = X @ self.Wq   # (B,T,D)
        K = X @ self.Wk
        V = X @ self.Wv
        Qh = split_heads(Q, self.num_heads)  # (B,h,T,d_k)
        Kh = split_heads(K, self.num_heads)
        Vh = split_heads(V, self.num_heads)

        attn_out, weights, attn_cache = scaled_dot_product_attention_forward(Qh, Kh, Vh, mask)
        attn_out_comb = combine_heads(attn_out)  # (B,T,D)
        attn_linear = attn_out_comb @ self.Wo  # (B,T,D)
        # residual + layernorm1
        resid1 = X + attn_linear
        ln1_out, ln1_cache = layer_norm_forward(resid1, self.gamma1, self.beta1)

        # --- FFN ---
        ff1 = ln1_out @ self.W1 + self.b1  # (B,T,d_ff)
        ff1_relu = np.maximum(0, ff1)
        ff2 = ff1_relu @ self.W2 + self.b2  # (B,T,D)

        resid2 = ln1_out + ff2
        ln2_out, ln2_cache = layer_norm_forward(resid2, self.gamma2, self.beta2)

        cache = {
            "X": X, "Q": Q, "K": K, "V": V,
            "Qh": Qh, "Kh": Kh, "Vh": Vh,
            "attn_cache": attn_cache, "weights": weights,
            "attn_out_comb": attn_out_comb, "attn_linear": attn_linear,
            "resid1": resid1, "ln1_cache": ln1_cache, "ln1_out": ln1_out,
            "ff1": ff1, "ff1_relu": ff1_relu, "ff2": ff2,
            "resid2": resid2, "ln2_cache": ln2_cache, "mask": mask
        }
        return ln2_out, cache

    def backward(self, dln2, cache):
        # dln2: (B,T,D) gradient w.r.t final output ln2_out
        B,T,D = dln2.shape
        # Unpack cache
        X = cache["X"]
        Q = cache["Q"]; K = cache["K"]; V = cache["V"]
        Qh = cache["Qh"]; Kh = cache["Kh"]; Vh = cache["Vh"]
        attn_cache = cache["attn_cache"]
        attn_out_comb = cache["attn_out_comb"]
        attn_linear = cache["attn_linear"]
        resid1 = cache["resid1"]
        ln1_cache = cache["ln1_cache"]
        ln1_out = cache["ln1_out"]  # Now properly extracted from cache
        ff1 = cache["ff1"]; ff1_relu = cache["ff1_relu"]; ff2 = cache["ff2"]
        resid2 = cache["resid2"]
        ln2_cache = cache["ln2_cache"]
        mask = cache["mask"]
        
        # grads init
        dWq = np.zeros_like(self.Wq)
        dWk = np.zeros_like(self.Wk)
        dWv = np.zeros_like(self.Wv)
        dWo = np.zeros_like(self.Wo)
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dgamma1 = np.zeros_like(self.gamma1)
        dbeta1 = np.zeros_like(self.beta1)
        dgamma2 = np.zeros_like(self.gamma2)
        dbeta2 = np.zeros_like(self.beta2)
        dX = np.zeros_like(X)

        # ---- backprop through LayerNorm2 (ln2_out = LN(resid2)) ----
        dresid2, dgamma2, dbeta2 = layer_norm_backward(dln2, ln2_cache)  # dresid2 shape (B,T,D)
        # resid2 = ln1_out + ff2
        dln1 = dresid2.copy()   # gradient into ln1_out (from resid2)
        dff2 = dresid2.copy()   # gradient into ff2

        # ---- backprop through FFN ----
        # ff2 = ff1_relu @ W2 + b2
        dff1_relu = dff2 @ self.W2.T  # (B,T,d_ff)
        
        # Fixed gradient computation for W2
        # ff2 = ff1_relu @ W2 + b2, where ff1_relu: (B,T,d_ff), W2: (d_ff, d_model)
        # dW2 = ff1_relu^T @ dff2
        ff1_relu_2d = ff1_relu.reshape(-1, self.d_ff)  # (B*T, d_ff)
        dff2_2d = dff2.reshape(-1, D)  # (B*T, d_model)
        dW2 += ff1_relu_2d.T @ dff2_2d  # (d_ff, d_model)
        
        db2 += np.sum(dff2, axis=(0,1))

        # backprop through ReLU
        dff1 = dff1_relu * (ff1 > 0)

        # ff1 = ln1_out @ W1 + b1
        ln1_out_2d = ln1_out.reshape(-1, D)  # (B*T, d_model)
        dff1_2d = dff1.reshape(-1, self.d_ff)  # (B*T, d_ff)
        dW1 += ln1_out_2d.T @ dff1_2d  # (d_model, d_ff)
        db1 += np.sum(dff1, axis=(0,1))

        # gradient into ln1_out accumulates from earlier (dln1 from resid2) + this
        dln1 += dff1 @ self.W1.T  # (B,T,D)

        # ---- backprop through LayerNorm1 (ln1_out = LN(resid1)) ----
        dresid1, dgamma1, dbeta1 = layer_norm_backward(dln1, ln1_cache)
        # resid1 = X + attn_linear
        dX += dresid1.copy()
        dattlin = dresid1.copy()

        # ---- backprop through attn_linear = combine_heads(attn_out) @ Wo ----
        attn_out_comb_2d = attn_out_comb.reshape(-1, D)  # (B*T, d_model)
        dattlin_2d = dattlin.reshape(-1, D)  # (B*T, d_model)
        dWo += attn_out_comb_2d.T @ dattlin_2d  # (d_model, d_model)
        d_attn_out_comb = dattlin @ self.Wo.T  # (B,T,D)

        # ---- backprop through combine_heads -> get attn_out (per-head) grads ----
        # attn_out_comb shape (B,T,D), we need to split into heads
        B, T, D = d_attn_out_comb.shape
        h = self.num_heads
        d_attn_out = d_attn_out_comb.reshape(B, T, h, self.d_k).transpose(0,2,1,3)  # (B,h,T,d_k)

        # ---- backprop through scaled-dot-product-attention ----
        dQh, dKh, dVh = scaled_dot_product_attention_backward(d_attn_out, attn_cache)  # each (B,h,T,d_k)

        # ---- backprop through head combine -> Q,K,V linear projections ----
        # merge heads
        dQ = combine_heads(dQh)  # (B,T,D)
        dK = combine_heads(dKh)
        dV = combine_heads(dVh)

        # Q = X @ Wq  -> dWq += X^T @ dQ ; dX += dQ @ Wq^T
        X_2d = X.reshape(-1, D)  # (B*T, d_model)
        dQ_2d = dQ.reshape(-1, D)  # (B*T, d_model)
        dK_2d = dK.reshape(-1, D)  # (B*T, d_model)
        dV_2d = dV.reshape(-1, D)  # (B*T, d_model)
        
        dWq += X_2d.T @ dQ_2d  # (d_model, d_model)
        dWk += X_2d.T @ dK_2d  # (d_model, d_model)
        dWv += X_2d.T @ dV_2d  # (d_model, d_model)

        dX += dQ @ self.Wq.T
        dX += dK @ self.Wk.T
        dX += dV @ self.Wv.T

        grads = {
            "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo,
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "gamma1": dgamma1, "beta1": dbeta1, "gamma2": dgamma2, "beta2": dbeta2
        }
        return grads, dX

# ---------------------------
# Full Transformer (single encoder)
# ---------------------------
class Transformer:
    def __init__(self, d_model=100, num_heads=5, d_ff=256, output_dim=3, max_len=50, seed=42):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.encoder = TransformerEncoderLayer(d_model, num_heads, d_ff, seed=seed)
        rng = np.random.RandomState(seed+1)
        self.params = {
            # copy encoder params into single dict
            "Wq": self.encoder.Wq, "Wk": self.encoder.Wk, "Wv": self.encoder.Wv, "Wo": self.encoder.Wo,
            "W1": self.encoder.W1, "b1": self.encoder.b1, "W2": self.encoder.W2, "b2": self.encoder.b2,
            "gamma1": self.encoder.gamma1, "beta1": self.encoder.beta1, "gamma2": self.encoder.gamma2, "beta2": self.encoder.beta2,
            "W_out": rng.randn(d_model, output_dim).astype(np.float32) * 0.01,
            "b_out": np.zeros((output_dim,), dtype=np.float32)
        }
        # positional encoding will be added outside (in training script, embeddings dimension must match d_model)
        self.positional = positional_encoding(max_len, d_model)

    def forward(self, X_emb):
        # X_emb: (B, T, d_model)
        B, T, D = X_emb.shape
        X = X_emb + self.positional[:T]
        enc_out, cache_enc = self.encoder.forward(X, mask=None)
        # pool (mean)
        pooled = np.mean(enc_out, axis=1)  # (B, D)
        logits = pooled @ self.params["W_out"] + self.params["b_out"]
        cache = {"enc_cache": cache_enc, "pooled": pooled, "X_emb": X_emb}
        return logits, cache

    def backward(self, dlogits, cache):
        # dlogits: (B, C)
        enc_cache = cache["enc_cache"]
        pooled = cache["pooled"]  # (B, D)
        B = dlogits.shape[0]
        # W_out grads
        dW_out = pooled.T @ dlogits  # (D, C)
        db_out = np.sum(dlogits, axis=0)
        # grad into pooled
        dpooled = dlogits @ self.params["W_out"].T  # (B, D)
        # broadcast to enc_out shape: mean pooling gradient
        denc = np.repeat(dpooled[:, np.newaxis, :], enc_cache["X"].shape[1], axis=1) / enc_cache["X"].shape[1]
        # now backprop through encoder layer
        grads_enc, dX = self.encoder.backward(denc, enc_cache)
        # collect grads into params dict
        grads = {
            "Wq": grads_enc["Wq"], "Wk": grads_enc["Wk"], "Wv": grads_enc["Wv"], "Wo": grads_enc["Wo"],
            "W1": grads_enc["W1"], "b1": grads_enc["b1"], "W2": grads_enc["W2"], "b2": grads_enc["b2"],
            "gamma1": grads_enc["gamma1"], "beta1": grads_enc["beta1"], "gamma2": grads_enc["gamma2"], "beta2": grads_enc["beta2"],
            "W_out": dW_out, "b_out": db_out
        }
        # dX is gradient wrt encoder input (after positional added) ; positional not trainable usually, so return dX_emb = dX
        dX_emb = dX
        return grads, dX_emb