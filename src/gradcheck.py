import sys, os
sys.path.append(os.path.dirname(__file__))
import numpy as np
from nn import Tensor, Linear, Conv1d, AvgPoolDownsample1d, NearestUpsample1d, concat_channels, mse_loss

np.random.seed(0)


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f()
        x[idx] = orig - eps
        f_minus = f()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def check(name, build_fn, param_tensor):
    param_tensor.zero_grad()
    loss = build_fn()
    loss.backward()
    analytic = param_tensor.grad.copy()

    def f():
        out = build_fn()
        return out.data.sum() if out.data.ndim else float(out.data)

    numeric = numerical_grad(f, param_tensor.data)
    err = np.abs(analytic - numeric).max()
    status = "OK" if err < 1e-4 else "FAIL"
    print(f"[{status}] {name}: max abs error = {err:.2e}")


# ---- Linear ----
x = Tensor(np.random.randn(3, 5))
lin = Linear(5, 4)
def f1():
    x2 = Tensor(x.data.copy())
    out = lin(x2)
    return Tensor(out.data.sum())
def build1():
    x2 = Tensor(x.data.copy())
    out = lin(x2)
    s = Tensor(out.data.sum(), True, (out,), "sum")
    def _b():
        out.grad += np.ones_like(out.data)
    s._backward = _b
    return s
check("Linear.W", build1, lin.W)

# ---- Conv1d ----
xc = Tensor(np.random.randn(2, 3, 16))
conv = Conv1d(3, 5, 3)
def build2():
    out = conv(xc)
    s = Tensor(out.data.sum(), True, (out,), "sum")
    def _b():
        out.grad += np.ones_like(out.data)
    s._backward = _b
    return s
check("Conv1d.W", build2, conv.W)

def build2x():
    out = conv(xc)
    s = Tensor(out.data.sum(), True, (out,), "sum")
    def _b():
        out.grad += np.ones_like(out.data)
    s._backward = _b
    return s
check("Conv1d.input(x)", build2x, xc)

# ---- Downsample / Upsample ----
xd = Tensor(np.random.randn(2, 3, 10))
down = AvgPoolDownsample1d()
def build3():
    out = down(xd)
    s = Tensor(out.data.sum(), True, (out,), "sum")
    def _b():
        out.grad += np.ones_like(out.data)
    s._backward = _b
    return s
check("Downsample.input", build3, xd)

xu = Tensor(np.random.randn(2, 3, 8))
up = NearestUpsample1d()
def build4():
    out = up(xu)
    s = Tensor(out.data.sum(), True, (out,), "sum")
    def _b():
        out.grad += np.ones_like(out.data)
    s._backward = _b
    return s
check("Upsample.input", build4, xu)

# ---- Concat ----
xa = Tensor(np.random.randn(2, 3, 6))
xb = Tensor(np.random.randn(2, 2, 6))
def build5():
    out = concat_channels(xa, xb)
    s = Tensor(out.data.sum(), True, (out,), "sum")
    def _b():
        out.grad += np.ones_like(out.data)
    s._backward = _b
    return s
check("Concat.a", build5, xa)

# ---- MSE end to end through Conv ----
xm = Tensor(np.random.randn(2, 2, 12))
conv2 = Conv1d(2, 2, 3)
target = np.random.randn(2, 2, 12)
def build6():
    out = conv2(xm)
    return mse_loss(out, target)
check("MSE-through-Conv1d.W", build6, conv2.W)

print("\nAll gradient checks complete.")
