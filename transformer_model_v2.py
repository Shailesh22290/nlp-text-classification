# transformer_model_v2.py
import numpy as np

# --------------------------
# Simple helpers
# --------------------------
def softmax_rowwise(x):
    # x: (..., T)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

def positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe

def layer_norm_forward(x, gamma, beta, eps=1e-6):
    # x: (..., D)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_norm = (x - mean) * inv
    out = gamma * x_norm + beta
    cache = (x_norm, inv, gamma)
    return out, cache

def layer_norm_backward(dout, cache):
    x_norm, inv, gamma = cache
    D = x_norm.shape[-1]
    dgamma = np.sum(dout * x_norm, axis=tuple(range(dout.ndim - 1)))
    dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
    L = int(np.prod(dout.shape[:-1]))
    dxnorm2d = (dout * gamma).reshape(L, D)
    xnorm2d = x_norm.reshape(L, D)
    sum_dxnorm = np.sum(dxnorm2d, axis=1, keepdims=True)
    sum_dxnorm_xnorm = np.sum(dxnorm2d * xnorm2d, axis=1, keepdims=True)
    dx2d = (dxnorm2d - sum_dxnorm / D - xnorm2d * (sum_dxnorm_xnorm / D)) * inv.reshape(-1,1)
    dx = dx2d.reshape(*dout.shape)
    return dx, dgamma, dbeta

def split_heads(x, n_heads):
    # x: (B, T, D) -> (B, n_heads, T, d_k)
    B,T,D = x.shape
    assert D % n_heads == 0
    d_k = D // n_heads
    return x.reshape(B, T, n_heads, d_k).transpose(0,2,1,3)

def combine_heads(x):
    # x: (B, n_heads, T, d_k) -> (B, T, D)
    B,h,T,d_k = x.shape
    return x.transpose(0,2,1,3).reshape(B, T, h*d_k)

def scaled_dot_attn_forward(Q, K, V, mask=None):
    # Q,K,V: (B,h,T,d_k)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0,1,3,2) / np.sqrt(d_k)  # (B,h,T,T)
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    B,h,T,_ = scores.shape
    scores2 = scores.reshape(B*h*T, T)
    w = softmax_rowwise(scores2).reshape(B,h,T,T)
    out = w @ V
    cache = (Q,K,V,w)
    return out, cache

def scaled_dot_attn_backward(dout, cache):
    Q,K,V,w = cache
    B,h,T,d_k = Q.shape
    # dW = dout @ V^T
    dW = np.einsum('bhik,bhjk->bhij', dout, V)  # (B,h,T,T)
    # dV = w^T @ dout
    dV = np.einsum('bhij,bhik->bhjk', w, dout)  # (B,h,T,d_k)
    # backprop through softmax row-wise
    dScores = np.zeros_like(dW)
    for b in range(B):
        for hh in range(h):
            Wmat = w[b,hh]   # (T,T)
            dWmat = dW[b,hh] # (T,T)
            for i in range(T):
                wi = Wmat[i]      # (T,)
                dwi = dWmat[i]
                s = np.dot(dwi, wi)
                dScores[b,hh,i,:] = wi * (dwi - s)
    inv = 1.0 / np.sqrt(d_k)
    dQ = np.einsum('bhij,bhjk->bhik', dScores, K) * inv
    dK = np.einsum('bhij,bhik->bhjk', dScores, Q) * inv
    return dQ, dK, dV

# --------------------------
# Transformer Encoder (pre-LN single layer)
# --------------------------
class TransformerEncoder:
    def __init__(self, vocab_size, embed_dim, d_model, n_heads, d_ff, n_classes, max_len=50, seed=42):
        rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len

        # embeddings + projection to d_model
        self.Emb = rng.randn(vocab_size, embed_dim).astype(np.float32) * 0.01
        self.W_proj = rng.randn(embed_dim, d_model).astype(np.float32) * 0.01

        # positional
        self.positional = positional_encoding(max_len, d_model)

        # attention linear maps
        self.Wq = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wk = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wv = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wo = rng.randn(d_model, d_model).astype(np.float32) * 0.01

        # feed-forward
        self.W1 = rng.randn(d_model, d_ff).astype(np.float32) * 0.01
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = rng.randn(d_ff, d_model).astype(np.float32) * 0.01
        self.b2 = np.zeros((d_model,), dtype=np.float32)

        # pre-LN params (gamma/beta)
        self.gamma_attn = np.ones((d_model,), dtype=np.float32)
        self.beta_attn = np.zeros((d_model,), dtype=np.float32)
        self.gamma_ff = np.ones((d_model,), dtype=np.float32)
        self.beta_ff = np.zeros((d_model,), dtype=np.float32)

        # final classifier
        self.W_out = rng.randn(d_model, n_classes).astype(np.float32) * 0.01
        self.b_out = np.zeros((n_classes,), dtype=np.float32)

    def forward(self, X_tokens):
        """
        X_tokens: (B, T) token ids
        returns: logits (B, C), cache
        """
        B,T = X_tokens.shape
        # Embedding lookup + linear projection to d_model
        X_emb = self.Emb[X_tokens]               # (B,T,embed_dim)
        X = X_emb @ self.W_proj                  # (B,T,d_model)
        X = X + self.positional[:T]              # add positional

        # --- Pre-LN + Self-Attention ---
        ln_in = X
        ln_out, ln_cache = layer_norm_forward(ln_in, self.gamma_attn, self.beta_attn)  # (B,T,d_model)

        Q = ln_out @ self.Wq
        K = ln_out @ self.Wk
        V = ln_out @ self.Wv

        Qh = split_heads(Q, self.n_heads)   # (B,h,T,d_k)
        Kh = split_heads(K, self.n_heads)
        Vh = split_heads(V, self.n_heads)

        attn_out, attn_cache = scaled_dot_attn_forward(Qh, Kh, Vh)  # (B,h,T,d_k)
        attn_comb = combine_heads(attn_out) @ self.Wo   # (B,T,d_model)

        X2 = ln_in + attn_comb   # residual

        # --- Pre-LN + Feedforward ---
        ln2_in = X2
        ln2_out, ln2_cache = layer_norm_forward(ln2_in, self.gamma_ff, self.beta_ff)

        ff1 = ln2_out @ self.W1 + self.b1
        ff1_relu = np.maximum(0, ff1)
        ff2 = ff1_relu @ self.W2 + self.b2

        X3 = ln2_in + ff2  # residual
        pooled = np.mean(X3, axis=1)  # (B,d_model)
        logits = pooled @ self.W_out + self.b_out

        cache = {
            "X_tokens": X_tokens, "X_emb": X_emb, "X": X, "ln_cache": ln_cache,
            "Qh": Qh, "Kh": Kh, "Vh": Vh, "attn_cache": attn_cache, "attn_comb": attn_comb,
            "ln2_cache": ln2_cache, "ff1": ff1, "ff1_relu": ff1_relu, "ff2": ff2,
            "X2": X2, "X3": X3, "pooled": pooled
        }
        return logits, cache

    def backward(self, dlogits, cache):
        """
        Backprop through classifier and single encoder block.
        Returns grads dict (matching attribute names) and dEmb_tokens (grad w.r.t. embedding table entries used)
        """
        B,T = cache["X_tokens"].shape
        grads = {}
        # classifier grads
        dW_out = cache["pooled"].T @ dlogits
        db_out = np.sum(dlogits, axis=0)
        dpooled = dlogits @ self.W_out.T  # (B,d_model)

        # broadcast pooled gradient to sequence (mean)
        dX3 = np.repeat(dpooled[:, None, :], T, axis=1) / T

        # backprop through X3 = ln2_in + ff2
        dln2_in = dX3.copy()
        dff2 = dX3.copy()

        # ff2 = ff1_relu @ W2 + b2
        dW2 = cache["ff1_relu"].reshape(-1, cache["ff1_relu"].shape[-1]).T @ dff2.reshape(-1, dff2.shape[-1])
        db2 = np.sum(dff2, axis=(0,1))
        dff1_relu = dff2 @ self.W2.T
        dff1 = dff1_relu * (cache["ff1"] > 0)

        dW1 = cache["ln2_cache"][0].__array__ if False else None  # placeholder unused
        dW1 = cache["ln2_cache"][0] if False else None  # keep code flow (we'll compute correctly below)
        # compute dW1 properly:
        ln2_out = layer_norm_forward(cache["X2"], self.gamma_ff, self.beta_ff)[0]  # recompute ln2_out (cheap)
        dW1 = ln2_out.reshape(-1, ln2_out.shape[-1]).T @ dff1.reshape(-1, dff1.shape[-1])
        db1 = np.sum(dff1, axis=(0,1))

        dln2_out = dff1 @ self.W1.T
        dln2_out += dln2_in  # accumulate

        # backprop through pre-LN (ln2_out = LayerNorm(ln2_in)) -> need layer_norm_backward
        dln2_in_from_norm, dgamma_ff, dbeta_ff = layer_norm_backward(dln2_out, cache["ln2_cache"])
        # ln2_in is X2, so grads w.r.t X2:
        dX2 = dln2_in_from_norm.copy()

        # backprop through attention residual: X2 = ln_in + attn_comb
        datt = dX2.copy()
        dln_in = dX2.copy()

        # attn_comb = combine_heads(attn_out) @ Wo
        datt_comb = datt @ np.linalg.pinv(self.Wo.T)  # avoid explicit grad for Wo temporarily (we'll compute Wo grad directly)
        # better compute dWo and d_attn_out:
        # dWo = attn_out_comb^T @ datt  ; attn_out_comb == combine_heads(attn_out)
        attn_out_comb = cache["attn_comb"]
        # attn_out_comb shape: (B,T,d_model)
        dWo = combine_heads(cache["attn_cache"][0]).reshape(-1, self.d_model).T @ datt.reshape(-1, self.d_model) if False else (attn_out_comb.reshape(-1, attn_out_comb.shape[-1]).T @ datt.reshape(-1,datt.shape[-1]))
        # simpler direct:
        dWo = attn_out_comb.reshape(-1, attn_out_comb.shape[-1]).T @ datt.reshape(-1, datt.shape[-1])
        d_attn_comb = datt @ self.Wo.T
        # reshape to heads
        d_attn_out = d_attn_comb.reshape(B, T, self.n_heads, self.d_model//self.n_heads).transpose(0,2,1,3)

        # backprop through attention block (dQh,dKh,dVh)
        dQh, dKh, dVh = scaled_dot_attn_backward(d_attn_out, cache["attn_cache"])
        # combine heads
        dQ = combine_heads(dQh)
        dK = combine_heads(dKh)
        dV = combine_heads(dVh)

        # Q = ln_out @ Wq ; K = ln_out @ Wk ; V = ln_out @ Wv
        dWq = cache["ln_cache"][0].reshape(-1, cache["ln_cache"][0].shape[-1]).T @ dQ.reshape(-1, dQ.shape[-1]) if False else (cache["ln_cache"][0].reshape(-1, cache["ln_cache"][0].shape[-1]).T @ dQ.reshape(-1, dQ.shape[-1]))
        # above line used cache['ln_cache'][0] incorrectly, recompute ln_out properly:
        ln_in = cache["X"]  # note: cache['X'] was pre-projection-to-d_model
        ln_out_recomputed, _ = layer_norm_forward(ln_in, self.gamma_attn, self.beta_attn)
        dWq = ln_out_recomputed.reshape(-1, ln_out_recomputed.shape[-1]).T @ dQ.reshape(-1, dQ.shape[-1])
        dWk = ln_out_recomputed.reshape(-1, ln_out_recomputed.shape[-1]).T @ dK.reshape(-1, dK.shape[-1])
        dWv = ln_out_recomputed.reshape(-1, ln_out_recomputed.shape[-1]).T @ dV.reshape(-1, dV.shape[-1])

        # gradient to ln_out from Q/K/V paths:
        dln_out_from_Q = dQ @ self.Wq.T
        dln_out_from_K = dK @ self.Wk.T
        dln_out_from_V = dV @ self.Wv.T
        dln_out = dln_out_from_Q + dln_out_from_K + dln_out_from_V

        # backprop through pre-LN for attention: ln_out = LayerNorm(ln_in)
        dln_in_from_norm, dgamma_attn, dbeta_attn = layer_norm_backward(dln_out + dln_in, cache["ln_cache"])
        # ln_in was X (input to first LN)
        dX = dln_in_from_norm.copy()

        # now propagate dX back to embedding projection and embedding table
        # X = X_emb @ W_proj + pos
        dW_proj = cache["X_emb"].reshape(-1, cache["X_emb"].shape[-1]).T @ dX.reshape(-1, dX.shape[-1])
        dX_emb = dX @ self.W_proj.T  # (B,T,embed_dim)

        # accumulate grads into grads dict
        grads = {
            "Emb": np.zeros_like(self.Emb),  # we will scatter dX_emb into this per-token
            "W_proj": dW_proj,
            "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo,
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "gamma_attn": dgamma_attn, "beta_attn": dbeta_attn,
            "gamma_ff": dgamma_ff, "beta_ff": dbeta_ff,
            "W_out": dW_out, "b_out": db_out
        }

        # scatter dX_emb into Emb gradients
        X_tokens = cache["X_tokens"]
        B, T = X_tokens.shape
        for i in range(B):
            for t in range(T):
                idx = int(X_tokens[i,t])
                grads["Emb"][idx] += dX_emb[i,t]

        return grads, grads["Emb"]
