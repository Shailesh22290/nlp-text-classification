# lstm_model.py
import numpy as np
from rnn_model import softmax, cross_entropy_loss_and_grad  # reuse utils

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        H, D, C = hidden_dim, input_dim, output_dim

        # combine all gates into single weight matrices for efficiency
        self.params = {}
        self.params["Wx"] = (rng.randn(D, 4*H) * 0.01).astype(np.float32)
        self.params["Wh"] = (rng.randn(H, 4*H) * 0.01).astype(np.float32)
        self.params["b"]  = np.zeros((4*H,), dtype=np.float32)
        self.params["W_out"] = (rng.randn(H, C) * 0.01).astype(np.float32)
        self.params["b_out"] = np.zeros((C,), dtype=np.float32)

    def forward(self, X_emb):
        B, T, D = X_emb.shape
        H = self.hidden_dim
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        hs = np.zeros((B, T, H), dtype=np.float32)
        cs = np.zeros((B, T, H), dtype=np.float32)

        h_t = np.zeros((B, H), dtype=np.float32)
        c_t = np.zeros((B, H), dtype=np.float32)

        gates = np.zeros((B, T, 4*H), dtype=np.float32)

        for t in range(T):
            x_t = X_emb[:, t, :]  # (B,D)
            gates_t = x_t.dot(Wx) + h_t.dot(Wh) + b  # (B,4H)
            i = sigmoid(gates_t[:, 0:H])
            f = sigmoid(gates_t[:, H:2*H])
            o = sigmoid(gates_t[:, 2*H:3*H])
            g = np.tanh(gates_t[:, 3*H:4*H])
            c_t = f * c_t + i * g
            h_t = o * np.tanh(c_t)

            hs[:, t, :] = h_t
            cs[:, t, :] = c_t
            gates[:, t, :] = gates_t

        logits = h_t.dot(self.params["W_out"]) + self.params["b_out"]  # last h
        cache = {"X_emb": X_emb, "hs": hs, "cs": cs, "gates": gates}
        return logits, cache

    def backward(self, dlogits, cache):
        X_emb, hs, cs, gates = cache["X_emb"], cache["hs"], cache["cs"], cache["gates"]
        B, T, D = X_emb.shape
        H = self.hidden_dim

        # grads init
        dWx = np.zeros_like(self.params["Wx"])
        dWh = np.zeros_like(self.params["Wh"])
        db  = np.zeros_like(self.params["b"])
        dW_out = np.zeros_like(self.params["W_out"])
        db_out = np.zeros_like(self.params["b_out"])
        dX_emb = np.zeros_like(X_emb)

        # output layer grads
        h_last = hs[:, -1, :]
        dW_out = h_last.T.dot(dlogits)
        db_out = np.sum(dlogits, axis=0)

        dh_next = dlogits.dot(self.params["W_out"].T)  # (B,H)
        dc_next = np.zeros((B, H), dtype=np.float32)

        for t in reversed(range(T)):
            h_t = hs[:, t, :]
            c_t = cs[:, t, :]
            c_prev = cs[:, t-1, :] if t > 0 else np.zeros_like(c_t)
            h_prev = hs[:, t-1, :] if t > 0 else np.zeros_like(h_t)

            gates_t = gates[:, t, :]
            i = sigmoid(gates_t[:, 0:H])
            f = sigmoid(gates_t[:, H:2*H])
            o = sigmoid(gates_t[:, 2*H:3*H])
            g = np.tanh(gates_t[:, 3*H:4*H])

            # backprop through output h_t = o * tanh(c_t)
            do = dh_next * np.tanh(c_t)
            dc = dh_next * o * (1 - np.tanh(c_t)**2) + dc_next
            di = dc * g
            dg = dc * i
            df = dc * c_prev
            dc_next = dc * f

            # derivatives wrt pre-activations
            di_in = di * i * (1 - i)
            df_in = df * f * (1 - f)
            do_in = do * o * (1 - o)
            dg_in = dg * (1 - g*g)
            dGates = np.hstack((di_in, df_in, do_in, dg_in))  # (B,4H)

            # grads to params
            x_t = X_emb[:, t, :]
            dWx += x_t.T.dot(dGates)
            dWh += h_prev.T.dot(dGates)
            db  += np.sum(dGates, axis=0)

            # grads wrt inputs for embedding
            dX_emb[:, t, :] = dGates.dot(self.params["Wx"].T)

            # pass gradients to previous h
            dh_next = dGates.dot(self.params["Wh"].T)

        grads = {
            "Wx": dWx,
            "Wh": dWh,
            "b": db,
            "W_out": dW_out,
            "b_out": db_out
        }
        return grads, dX_emb
