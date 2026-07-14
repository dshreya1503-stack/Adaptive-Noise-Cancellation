"""
nn.py
-----
A minimal reverse-mode autodiff engine + neural network layers, built from
scratch in NumPy (no PyTorch/TensorFlow available in this offline environment).

This is intentionally small (~250 lines) but implements real backpropagation:
- Tensor: tracks a computation graph, .backward() does topological-order
  reverse-mode differentiation (same core idea as PyTorch autograd).
- Layers: Linear (dense), Conv1d (via im2col + col2im, stride=1, 'same' padding),
  ReLU, Tanh, AvgPoolDownsample1d (factor 2), NearestUpsample1d (factor 2),
  Concat (for U-Net-style skip connections).
- Adam optimizer.

All conv/pool/upsample ops work on batched 1D signals of shape (N, C, L).
"""

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        # Always allocate a grad buffer (even for non-learnable tensors like
        # plain inputs) so backward() can safely accumulate into it; only
        # tensors with requires_grad=True get updated by the optimizer.
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
        # Backward closures capture `out` (to read out.grad), which creates a
        # self-referencing cycle on every intermediate tensor (out -> closure
        # -> out). CPython's refcounter can't free cycles on its own, so they
        # pile up until the cyclic GC runs -- with large numpy arrays behind
        # each node, that pileup is very memory-hungry in tight training loops.
        # Clearing _backward/_prev here breaks the cycles immediately so
        # refcounting can reclaim everything as soon as the caller's local
        # variables (x, pred, loss, ...) go out of scope.
        for v in topo:
            v._backward = lambda: None
            v._prev = ()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ---- basic ops needed for our layers ----
    def __add__(self, other):
        out = Tensor(self.data + other.data, True, (self, other), "+")

        def _backward():
            self._accumulate(self.data.shape, out.grad)
            other._accumulate(other.data.shape, out.grad)
        out._backward = _backward
        return out

    def _accumulate(self, shape, grad):
        # sum-reduce grad down to `shape` (handles bias broadcasting)
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        self.grad = self.grad + grad

    def relu(self):
        out = Tensor(np.maximum(0, self.data), True, (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, True, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out


def mse_loss(pred: Tensor, target: np.ndarray):
    diff = pred.data - target
    out = Tensor(np.mean(diff ** 2), True, (pred,), "mse")

    def _backward():
        n = diff.size
        pred.grad += (2.0 / n) * diff * out.grad
    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_f, out_f):
        limit = np.sqrt(2.0 / in_f)
        self.W = Tensor(np.random.randn(out_f, in_f) * limit, requires_grad=True)
        self.b = Tensor(np.zeros(out_f), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        out_data = x.data @ self.W.data.T + self.b.data
        out = Tensor(out_data, True, (x, self.W, self.b), "linear")

        def _backward():
            x.grad += out.grad @ self.W.data
            self.W.grad += out.grad.T @ x.data
            self.b.grad += out.grad.sum(axis=0)
        out._backward = _backward
        return out

    def parameters(self):
        return [self.W, self.b]


class Conv1d:
    """1D convolution, stride=1, 'same' padding (odd kernel size required)."""

    def __init__(self, in_ch, out_ch, kernel_size):
        assert kernel_size % 2 == 1, "use odd kernel size for 'same' padding"
        self.K = kernel_size
        self.pad = kernel_size // 2
        limit = np.sqrt(2.0 / (in_ch * kernel_size))
        self.W = Tensor(np.random.randn(out_ch, in_ch, kernel_size) * limit, requires_grad=True)
        self.b = Tensor(np.zeros(out_ch), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        N, Cin, L = x.data.shape
        K, pad = self.K, self.pad
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (pad, pad)))

        # im2col: (N, Cin, L, K)
        from numpy.lib.stride_tricks import sliding_window_view
        patches = sliding_window_view(x_padded, K, axis=2)  # (N, Cin, L, K)
        patches_flat = patches.transpose(0, 2, 1, 3).reshape(N, L, Cin * K)  # (N,L,Cin*K)

        W_flat = self.W.data.reshape(self.W.data.shape[0], -1)  # (Cout, Cin*K)
        out_data = patches_flat @ W_flat.T + self.b.data  # (N, L, Cout)
        out_data = out_data.transpose(0, 2, 1)  # (N, Cout, L)

        out = Tensor(out_data, True, (x, self.W, self.b), "conv1d")

        def _backward():
            dOut = out.grad.transpose(0, 2, 1)  # (N, L, Cout)
            dOut_flat = dOut.reshape(N * L, -1)  # (N*L, Cout)
            patches_flat2 = patches_flat.reshape(N * L, Cin * K)

            dW_flat = dOut_flat.T @ patches_flat2  # (Cout, Cin*K)
            self.W.grad += dW_flat.reshape(self.W.data.shape)
            self.b.grad += dOut.sum(axis=(0, 1))

            dPatches_flat = dOut_flat @ W_flat  # (N*L, Cin*K)
            dPatches = dPatches_flat.reshape(N, L, Cin, K)

            dX_padded = np.zeros_like(x_padded)
            for k in range(K):
                dX_padded[:, :, k:k + L] += dPatches[:, :, :, k].transpose(0, 2, 1)
            x.grad += dX_padded[:, :, pad:pad + L]

        out._backward = _backward
        return out

    def parameters(self):
        return [self.W, self.b]


class AvgPoolDownsample1d:
    """Average-pools pairs of samples: (N,C,L) -> (N,C,L//2). No parameters."""

    def __call__(self, x: Tensor) -> Tensor:
        N, C, L = x.data.shape
        L2 = L - (L % 2)
        trimmed = x.data[:, :, :L2]
        out_data = trimmed.reshape(N, C, L2 // 2, 2).mean(axis=3)
        out = Tensor(out_data, True, (x,), "downsample")

        def _backward():
            g = np.repeat(out.grad, 2, axis=2) * 0.5
            full = np.zeros_like(x.data)
            full[:, :, :g.shape[2]] += g
            x.grad += full
        out._backward = _backward
        return out


class NearestUpsample1d:
    """Nearest-neighbor upsample by factor 2: (N,C,L) -> (N,C,2L). No parameters."""

    def __call__(self, x: Tensor) -> Tensor:
        out_data = np.repeat(x.data, 2, axis=2)
        out = Tensor(out_data, True, (x,), "upsample")

        def _backward():
            N, C, L2 = out.grad.shape
            g = out.grad.reshape(N, C, L2 // 2, 2).sum(axis=3)
            x.grad += g
        out._backward = _backward
        return out


def concat_channels(a: Tensor, b: Tensor) -> Tensor:
    """Concatenate two (N,C,L) tensors along the channel axis, cropping the
    longer one to match (handles off-by-one length mismatches from pooling)."""
    L = min(a.data.shape[2], b.data.shape[2])
    a_data, b_data = a.data[:, :, :L], b.data[:, :, :L]
    out_data = np.concatenate([a_data, b_data], axis=1)
    out = Tensor(out_data, True, (a, b), "concat")
    Ca = a.data.shape[1]

    def _backward():
        ga, gb = out.grad[:, :Ca, :], out.grad[:, Ca:, :]
        a.grad[:, :, :L] += ga
        b.grad[:, :, :L] += gb
    out._backward = _backward
    return out


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
