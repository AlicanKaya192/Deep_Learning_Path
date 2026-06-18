"""
================================================================================
MODÜL 04: GERİ YAYILIM (BACKPROPAGATION)
================================================================================

Bu dosya Deep Learning Path'in dördüncü modülüdür. Backpropagation'ı
matematiksel temelden, adım adım manuel hesaplamadan sıfırdan uygulamaya
kadar kapsamlı şekilde öğretir.

KAPSANAN KONULAR:
  - Zincir kuralı (Chain Rule) — tek ve çok değişkenli
  - Hesaplama grafı (Computational Graph)
  - Forward pass ve backward pass tam implementasyonu
  - Her gradyanın adım adım manuel hesabı
  - Vanishing Gradient — matematiksel kanıt ve görsel
  - Exploding Gradient — sebepleri ve gradient clipping çözümü
  - Sıfırdan mini autograd motoru (NumPy)
  - TensorFlow ve PyTorch autograd karşılaştırması

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

ÇALIŞTIRMA:
  python 04_backpropagation_numpy.py

YAZAR: Deep Learning Path — Modül 04
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  MODÜL 04: GERİ YAYILIM (BACKPROPAGATION)")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: ZİNCİR KURALI — TEMEL
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Zincir Kuralı (Chain Rule)")
print("=" * 65)

print("""
  Zincir Kuralı:
  Eğer  z = f(y)  ve  y = g(x)  ise:
    dz/dx  =  dz/dy · dy/dx

  Derin ağ örneği (3 işlem):
    L = loss(a₂)
    a₂ = sigmoid(z₂)
    z₂ = W₂·a₁ + b₂

  dL/dW₂  =  dL/da₂ · da₂/dz₂ · dz₂/dW₂

  Her terim:
    dL/da₂  : Kayıp fonksiyonunun çıkışa göre türevi
    da₂/dz₂ : Sigmoid'in türevi  =  a₂·(1−a₂)
    dz₂/dW₂ : Ağırlıklı toplamın W₂'ye göre türevi  =  a₁
""")

# ── Sayısal Doğrulama: Numerik vs Analitik Gradyan ──────────────────────────
print("  Sayısal vs Analitik Gradyan Doğrulaması:")
print("  f(x) = x³ + 2x² − 5x + 1   →   f'(x) = 3x² + 4x − 5")

def f(x: float) -> float:
    return x**3 + 2*x**2 - 5*x + 1

def f_analytic_grad(x: float) -> float:
    """Analitik türev: 3x² + 4x − 5"""
    return 3*x**2 + 4*x - 5

def numerical_gradient(func, x: float, h: float = 1e-5) -> float:
    """
    Sayısal gradyan — merkezi fark yöntemi.
    f'(x) ≈ [f(x+h) − f(x−h)] / (2h)
    Neden merkezi fark? İleri fark O(h), merkezi fark O(h²) hassasiyet.
    """
    return (func(x + h) - func(x - h)) / (2 * h)

print(f"\n  {'x':>5} | {'Analitik':>12} | {'Sayısal':>12} | {'Fark':>12}")
print(f"  {'-'*50}")
for x_val in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    analytic = f_analytic_grad(x_val)
    numeric  = numerical_gradient(f, x_val)
    diff     = abs(analytic - numeric)
    print(f"  {x_val:>5.1f} | {analytic:>12.6f} | {numeric:>12.6f} | {diff:>12.2e}")

print("\n  Farklar 1e-10 düzeyinde → analitik türev doğrulanmıştır.")


# ============================================================
# SECTION 2: HESAPLAMA GRAFI
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: Hesaplama Grafı (Computational Graph)")
print("=" * 65)

print("""
  Hesaplama Grafı: Her işlemi bir düğüm (node) olarak gösterir.
  Forward pass: soldan sağa değer hesaplama.
  Backward pass: sağdan sola gradyan iletimi.

  Örnek: L = (w·x − y)²

  FORWARD PASS (değerler):
    w=2, x=3, y=5
    ┌────────────────────────────────────────────────┐
    │  x=3 ──┐                                       │
    │         ├──[×]── z=6 ──[−]── e=1 ──[²]── L=1  │
    │  w=2 ──┘            ↑                          │
    │                     y=5                        │
    └────────────────────────────────────────────────┘

  BACKWARD PASS (gradyanlar):
    dL/dL = 1                            (başlangıç)
    dL/de = 2·e = 2·1 = 2               (kare türevi)
    dL/dz = dL/de · de/dz = 2·1 = 2     (çıkarma: de/dz=1)
    dL/dw = dL/dz · dz/dw = 2·x = 2·3 = 6
    dL/dx = dL/dz · dz/dx = 2·w = 2·2 = 4
""")

# Manuel hesaplama doğrulaması
w, x, y_target = 2.0, 3.0, 5.0
z = w * x
e = z - y_target
L = e ** 2

dL_de = 2 * e
dL_dz = dL_de * 1.0       # de/dz = 1
dL_dw = dL_dz * x          # dz/dw = x
dL_dx = dL_dz * w          # dz/dx = w

print(f"  Sayısal doğrulama: w={w}, x={x}, y={y_target}")
print(f"  L = ({w}·{x} − {y_target})² = {L}")
print(f"  dL/dw = {dL_dw}  (sayısal: {numerical_gradient(lambda ww: (ww*x-y_target)**2, w):.4f})")
print(f"  dL/dx = {dL_dx}  (sayısal: {numerical_gradient(lambda xx: (w*xx-y_target)**2, x):.4f})")


# ============================================================
# SECTION 3: TAM BACKPROP — 2 KATMANLI MLP
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: Tam Backpropagation — 2 Katmanlı MLP")
print("=" * 65)

print("""
  Mimari: Giriş(2) → Gizli(3) → Çıkış(1)
  Aktivasyon: Sigmoid her yerde
  Kayıp: Binary Cross-Entropy

  FORWARD PASS:
    Z1 = X @ W1 + b1          → (n, 3)
    A1 = sigmoid(Z1)          → (n, 3)
    Z2 = A1 @ W2 + b2         → (n, 1)
    A2 = sigmoid(Z2) = ŷ      → (n, 1)
    L  = BCE(y, ŷ)

  BACKWARD PASS (zincir kuralı):
    dL/dA2 = (A2 − y) / n                    [BCE+Sigmoid basitleşmesi]
    dL/dW2 = A1.T @ dL/dA2                   [matris türevi]
    dL/db2 = sum(dL/dA2, axis=0)
    dL/dA1 = (dL/dA2) @ W2.T                 [hatayı geri yay]
    δ1     = dL/dA1 * A1*(1−A1)              [sigmoid türevi]
    dL/dW1 = X.T @ δ1
    dL/db1 = sum(δ1, axis=0)
""")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class BackpropMLP:
    """
    2 Katmanlı MLP — Tam Backpropagation İmplementasyonu.
    Her adım yorumlanmış ve gradyan kontrolü destekli.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 lr: float = 0.1):
        self.lr = lr
        # Xavier başlatma
        self.W1 = np.random.randn(n_in,     n_hidden) * np.sqrt(2/(n_in+n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_out)    * np.sqrt(2/(n_hidden+n_out))
        self.b2 = np.zeros((1, n_out))
        # Cache — forward pass değerleri backprop için saklanır
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        İleri geçiş: tüm ara değerleri cache'e kaydeder.
        Backprop sırasında bu değerlere ihtiyaç duyulur.
        """
        Z1 = X  @ self.W1 + self.b1
        A1 = sigmoid(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid(Z2)

        # Backprop için sakla
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2

    def bce_loss(self, y: np.ndarray, yhat: np.ndarray) -> float:
        eps = 1e-15
        yhat = np.clip(yhat, eps, 1-eps)
        return float(-np.mean(y * np.log(yhat) + (1-y) * np.log(1-yhat)))

    def backward(self, y: np.ndarray) -> dict:
        """
        Geri yayılım: zincir kuralıyla tüm gradyanları hesaplar.
        Her adım açıklanmıştır.
        """
        n   = y.shape[0]
        X   = self.cache['X']
        A1  = self.cache['A1']
        A2  = self.cache['A2']

        # ── Çıkış katmanı ──────────────────────────────────────
        # BCE kaybının A2'ye göre türevi + Sigmoid türevi birleşir:
        # dL/dZ2 = (A2 - y) / n   ← güzel basitleşme!
        dL_dZ2 = (A2 - y) / n                         # (n, 1)

        # W2 ve b2 gradyanları
        dL_dW2 = A1.T @ dL_dZ2                        # (n_hidden, 1)
        dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)# (1, 1)

        # ── Gizli katman ───────────────────────────────────────
        # Hatayı geri yay: dL/dA1 = dL/dZ2 @ W2.T
        dL_dA1 = dL_dZ2 @ self.W2.T                   # (n, n_hidden)

        # Sigmoid türevi: A1*(1-A1) — element-wise çarpım
        dL_dZ1 = dL_dA1 * (A1 * (1 - A1))            # (n, n_hidden)

        # W1 ve b1 gradyanları
        dL_dW1 = X.T @ dL_dZ1                         # (n_in, n_hidden)
        dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)# (1, n_hidden)

        # ── Gradyan güncelleme ─────────────────────────────────
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

        return {
            'dW2': dL_dW2, 'db2': dL_db2,
            'dW1': dL_dW1, 'db1': dL_db1,
        }

    def gradient_check(self, X: np.ndarray, y: np.ndarray,
                        h: float = 1e-5) -> float:
        """
        Gradyan kontrolü: analitik vs sayısal gradyanları karşılaştırır.
        Ağırlıkları DEĞİŞTİRMEZ — sadece okuma yapar.
        Fark < 1e-4 ise backprop implementasyonu doğrudur.
        """
        # Analitik gradyanlar — güncelleme yapmadan hesapla
        yhat   = self.forward(X)
        n      = y.shape[0]
        A1     = self.cache['A1']
        A2     = self.cache['A2']
        dZ2    = (A2 - y) / n
        dA1    = dZ2 @ self.W2.T
        dZ1    = dA1 * (A1 * (1 - A1))
        analytic_dW1 = X.T @ dZ1

        # Sayısal gradyanlar — W1 için merkezi fark
        num_grads = np.zeros_like(self.W1)
        it = np.nditer(self.W1, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = self.W1[idx]

            self.W1[idx] = orig + h
            loss_plus = self.bce_loss(y, self.forward(X))

            self.W1[idx] = orig - h
            loss_minus = self.bce_loss(y, self.forward(X))

            self.W1[idx] = orig  # Geri yükle

            num_grads[idx] = (loss_plus - loss_minus) / (2 * h)
            it.iternext()

        numerator   = np.linalg.norm(analytic_dW1 - num_grads)
        denominator = np.linalg.norm(analytic_dW1) + np.linalg.norm(num_grads)
        return numerator / (denominator + 1e-15)


# ── XOR Eğitimi ve Gradyan Kontrolü ───────────────────────────────────────────
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

model_check = BackpropMLP(n_in=2, n_hidden=3, n_out=1, lr=0.5)

print("\n  Gradyan Kontrolü:")
rel_err = model_check.gradient_check(X_xor, y_xor)
status  = "✓ BAŞARILI" if rel_err < 1e-4 else "✗ HATA"
print(f"  Analitik vs Sayısal gradyan farkı: {rel_err:.2e}  →  {status}")

model = BackpropMLP(n_in=2, n_hidden=4, n_out=1, lr=0.5)

print("\n  XOR Eğitimi (5000 epoch):")
loss_hist = []
for epoch in range(5000):
    yhat = model.forward(X_xor)
    loss_hist.append(model.bce_loss(y_xor, yhat))
    model.backward(y_xor)

preds = (model.forward(X_xor) >= 0.5).astype(int)
acc   = np.mean(preds == y_xor)
print(f"  Final Loss: {loss_hist[-1]:.6f} | Accuracy: {acc:.1%}")


# ============================================================
# SECTION 4: SIFIRDAN MİNİ AUTOGRAD
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: Sıfırdan Mini Autograd Motoru")
print("=" * 65)

print("""
  Autograd: Her işlemi kaydeder, backward() çağrısında
  otomatik olarak zincir kuralını uygular.
  PyTorch'un temelinde tam da bu mantık yatar.
""")

class Value:
    """
    Skalar değer + gradyan takibi.
    Her işlem yeni bir Value nesnesi ve backward fonksiyonu oluşturur.
    Bu, PyTorch'un temel Tensor mekanizmasının sadeleştirilmiş versiyonudur.
    """

    def __init__(self, data: float, _children: tuple = (),
                 _op: str = '', label: str = ''):
        self.data     = float(data)
        self.grad     = 0.0          # Başlangıçta gradyan sıfır
        self._backward = lambda: None
        self._prev    = set(_children)
        self._op      = _op
        self.label    = label

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Toplama: her iki girişe de 1 çarpılarak gradyan iletilir
            self.grad  += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Çarpma: dL/da = dL/dout * b,  dL/db = dL/dout * a
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, exp: float):
        out = Value(self.data ** exp, (self,), f'**{exp}')

        def _backward():
            # d(xⁿ)/dx = n·x^(n-1)
            self.grad += out.grad * exp * (self.data ** (exp - 1))
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return other + (-self)

    def sigmoid(self):
        s   = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # sigmoid türevi: s*(1-s)
            self.grad += out.grad * s * (1 - s)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(np.clip(self.data, 1e-15, None)), (self,), 'log')

        def _backward():
            self.grad += out.grad / (self.data + 1e-15)
        out._backward = _backward
        return out

    def backward(self):
        """
        Topolojik sıralama ile tüm ağdaki gradyanları hesaplar.
        Topo sort: her düğüm, bağlı olduğu tüm düğümlerden sonra gelir.
        """
        topo   = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0  # Çıkış gradyanı = 1
        for node in reversed(topo):
            node._backward()


# ── Mini Autograd Demo ─────────────────────────────────────────────────────────
print("  Örnek: L = (w·x − y)²  için dL/dw ve dL/dx")

w_v = Value(2.0, label='w')
x_v = Value(3.0, label='x')
y_v = Value(5.0, label='y')

z_v = w_v * x_v         # z = w*x = 6
e_v = z_v - y_v         # e = z - y = 1
L_v = e_v ** 2          # L = e² = 1

L_v.backward()

print(f"\n  w={w_v.data}, x={x_v.data}, y={y_v.data}")
print(f"  L = (w·x − y)² = {L_v.data:.1f}")
print(f"  dL/dw = {w_v.grad:.4f}  (beklenen: {2*(w_v.data*x_v.data - y_v.data)*x_v.data:.4f})")
print(f"  dL/dx = {x_v.grad:.4f}  (beklenen: {2*(w_v.data*x_v.data - y_v.data)*w_v.data:.4f})")
print(f"  dL/dz = {z_v.grad:.4f}")
print(f"  dL/de = {e_v.grad:.4f}")

# ── XOR ile mini autograd ──────────────────────────────────────────────────────
print("\n  Mini Autograd ile XOR (tek örnek):")
# w1, w2, b -> h, w3, b2 -> out
np.random.seed(1)
w = [Value(np.random.randn(), label=f'w{i}') for i in range(6)]
b = [Value(0.0, label=f'b{i}') for i in range(2)]

# Giriş: (0, 1) -> beklenen çıkış: 1
inp = [Value(0.0), Value(1.0)]

# Gizli katman (2 nöron)
h1 = (w[0]*inp[0] + w[1]*inp[1] + b[0]).sigmoid()
h2 = (w[2]*inp[0] + w[3]*inp[1] + b[0]).sigmoid()

# Çıkış
out = (w[4]*h1 + w[5]*h2 + b[1]).sigmoid()
y_t = Value(1.0)

# BCE kaybı (yaklaşık)
loss = -(y_t * out.log() + (Value(1.0) - y_t) * (Value(1.0) - out).log())
loss.backward()

print(f"  Çıkış: {out.data:.4f}  |  Loss: {loss.data:.4f}")
print(f"  dL/dw0: {w[0].grad:.6f}  |  dL/dw1: {w[1].grad:.6f}")
print(f"  Gradyanlar hesaplandı — autograd çalışıyor ✓")


# ============================================================
# SECTION 5: VANISHING & EXPLODING GRADIENT
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: Vanishing ve Exploding Gradient")
print("=" * 65)

print("""
  VANİSHİNG GRADIENT:
  Sigmoid türevi maks. 0.25. 20 katmanda: 0.25^20 ≈ 9.1e-13
  İlk katmanlar öğrenemiyor → ağ derinleşemez.

  EXPLODING GRADIENT:
  Büyük ağırlıklar + zincir kuralı → gradyanlar katlanarak büyür.
  RNN'lerde çok yaygın. Sayısal taşma → NaN kayıp.

  ÇÖZÜMLER:
  Vanishing → ReLU ailesi, Batch Normalization, ResNet skip connections
  Exploding → Gradient Clipping, ağırlık başlatma (He, Xavier)
""")

# Gradient Clipping implementasyonu
def clip_gradients(gradients: dict, max_norm: float = 1.0) -> dict:
    """
    Gradient Clipping (Norm bazlı).
    Tüm gradyanların toplam L2 normu max_norm'dan büyükse ölçekle.
    W_clipped = W · (max_norm / total_norm)  eğer total_norm > max_norm
    """
    # Toplam gradyan normunu hesapla
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        return {k: v * scale for k, v in gradients.items()}
    return gradients

# Exploding gradient simülasyonu
print("  Exploding Gradient Simülasyonu (RNN benzeri):")
np.random.seed(0)
w_rnn    = 1.5   # 1'den büyük → gradyanlar büyüyor
grad     = 1.0
grad_hist = [grad]
for step in range(20):
    grad = grad * w_rnn
    grad_hist.append(grad)

print(f"  w={w_rnn}, başlangıç gradyanı=1.0")
for i in [0, 5, 10, 15, 19]:
    status = "⚠ PATLADI" if grad_hist[i] > 1000 else "OK"
    print(f"  Adım {i:>3}: gradyan = {grad_hist[i]:>12.2f}  {status}")


# ============================================================
# SECTION 6: TENSORFLOW / KERAS AUTOGRAD
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: TensorFlow Autograd (GradientTape)")
print("=" * 65)

try:
    import tensorflow as tf

    print(f"\n  TensorFlow {tf.__version__}")
    print("""
  TensorFlow'da GradientTape kullanımı:

  w = tf.Variable(2.0)
  x = tf.constant(3.0)
  y = tf.constant(5.0)

  with tf.GradientTape() as tape:
      z = w * x
      L = (z - y) ** 2

  dL_dw = tape.gradient(L, w)
  print(dL_dw.numpy())   # → 12.0
    """)

    # Gerçek çalıştırma
    w_tf = tf.Variable(2.0)
    x_tf = tf.constant(3.0)
    y_tf = tf.constant(5.0)

    with tf.GradientTape() as tape:
        z_tf  = w_tf * x_tf
        L_tf  = (z_tf - y_tf) ** 2

    dL_dw_tf = tape.gradient(L_tf, w_tf)
    print(f"  TF GradientTape sonucu: dL/dw = {dL_dw_tf.numpy()}")
    print(f"  Beklenen: {2*(2.0*3.0-5.0)*3.0}  ✓")

except ImportError:
    print("  TensorFlow yüklü değil — pip install tensorflow")


# ============================================================
# SECTION 7: PYTORCH AUTOGRAD
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 7: PyTorch Autograd")
print("=" * 65)

try:
    import torch

    print(f"\n  PyTorch {torch.__version__}")
    print("""
  PyTorch'ta requires_grad=True olan tensor'lar izlenir:

  w = torch.tensor(2.0, requires_grad=True)
  x = torch.tensor(3.0)
  y = torch.tensor(5.0)

  z = w * x
  L = (z - y) ** 2
  L.backward()

  print(w.grad)   # → tensor(12.)
    """)

    # Gerçek çalıştırma
    w_pt = torch.tensor(2.0, requires_grad=True)
    x_pt = torch.tensor(3.0)
    y_pt = torch.tensor(5.0)

    z_pt = w_pt * x_pt
    L_pt = (z_pt - y_pt) ** 2
    L_pt.backward()

    print(f"  PyTorch autograd sonucu: dL/dw = {w_pt.grad.item()}")
    print(f"  Beklenen: {2*(2.0*3.0-5.0)*3.0}  ✓")

    # no_grad örneği
    print("""
  NOT: inference sırasında gradyan hesabını kapat:
  with torch.no_grad():
      output = model(input)   # Daha hızlı, bellek tasarrufu
    """)

except ImportError:
    print("  PyTorch yüklü değil — pip install torch")


# ============================================================
# SECTION 8: GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 8: Görselleştirme")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Modül 04: Backpropagation Analizi", fontsize=14, fontweight='bold')

# ── Panel 1: Loss Eğrisi ───────────────────────────────────────────────────────
ax1 = axes[0, 0]
ax1.semilogy(loss_hist, color='#1565C0', lw=2)
ax1.set_title("XOR Eğitim Loss Eğrisi", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss (log)")
ax1.axhline(y=0.01, color='red', ls='--', alpha=0.6, label='Loss=0.01')
ax1.legend(); ax1.grid(True, alpha=0.3)

# ── Panel 2: Gradyan Normu (vanishing) ────────────────────────────────────────
ax2 = axes[0, 1]
n_layers = np.arange(1, 21)
ax2.semilogy(n_layers, 0.25**n_layers, 'b-o', ms=5, lw=2, label='Sigmoid (maks 0.25/katman)')
ax2.semilogy(n_layers, 0.9**n_layers,  'g-s', ms=5, lw=2, label='ReLU tipik (0.9/katman)')
ax2.semilogy(n_layers, np.ones(20),    'r-^', ms=5, lw=2, label='ReLU z>0 (1.0/katman)')
ax2.fill_between(n_layers, 0, 1e-7, alpha=0.1, color='red', label='Vanishing bölge')
ax2.set_title("Vanishing Gradient — Katman Derinliği", fontweight='bold')
ax2.set_xlabel("Katman Sayısı"); ax2.set_ylabel("Gradyan Büyüklüğü")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# ── Panel 3: Exploding Gradient ───────────────────────────────────────────────
ax3 = axes[1, 0]
steps = np.arange(21)
for w_val, color, label in [(0.5,'#2E7D32','w=0.5 (sönümlü)'),
                              (1.0,'#1565C0','w=1.0 (sabit)'),
                              (1.5,'#E65100','w=1.5 (patlayan)'),
                              (2.0,'#B71C1C','w=2.0 (hızlı patlayan)')]:
    grads = [w_val**i for i in steps]
    ax3.semilogy(steps, grads, color=color, lw=2, label=label)

ax3.axhline(y=1000, color='gray', ls=':', alpha=0.7, label='Tehlike eşiği')
ax3.set_title("Exploding Gradient — RNN Benzetimi", fontweight='bold')
ax3.set_xlabel("Adım"); ax3.set_ylabel("Gradyan Büyüklüğü (log)")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
ax3.set_ylim(1e-5, 1e6)

# ── Panel 4: Gradient Clipping Etkisi ─────────────────────────────────────────
ax4 = axes[1, 1]
w_exp  = 1.8
raw    = [w_exp**i for i in range(20)]
clipped = [min(g, 5.0) for g in raw]

ax4.plot(raw,     'r-o', lw=2, ms=5, label='Ham gradyan (w=1.8)')
ax4.plot(clipped, 'b-s', lw=2, ms=5, label='Clipped gradyan (max=5)')
ax4.axhline(y=5, color='blue', ls='--', alpha=0.6, label='Clip sınırı=5')
ax4.set_title("Gradient Clipping Etkisi", fontweight='bold')
ax4.set_xlabel("Adım"); ax4.set_ylabel("Gradyan Değeri")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3); ax4.set_ylim(0, 30)

plt.tight_layout()
plt.savefig(
    '/home/claude/deep_learning_path/04-Geri_Yayilim_Backpropagation/modul04_analiz.png',
    dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")


# ============================================================
# SECTION 9: ÖZET
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 9: Modül 04 Özeti")
print("=" * 65)
print("""
  ✓ Zincir kuralı: dL/dW = dL/dA · dA/dZ · dZ/dW
  ✓ Hesaplama grafı: her işlemi düğüm olarak modelle
  ✓ Forward: değerleri soldan sağa hesapla ve cache'e kaydet
  ✓ Backward: gradyanları sağdan sola zincir kuralıyla ilet
  ✓ Gradyan kontrolü: sayısal ≈ analitik → implementasyon doğru
  ✓ Vanishing: sigmoid derin ağlarda öldürür → ReLU kullan
  ✓ Exploding: büyük ağırlık × derin ağ → gradient clipping
  ✓ Autograd: PyTorch/TF otomatik gradyan hesabını kaydederek yapar

  SONRAKI MODÜL: 05 — Regularization
    Dropout, Batch Normalization, L1/L2, Early Stopping
""")
print("=" * 65)
print("  Modül 04 tamamlandı!")
print("=" * 65)
