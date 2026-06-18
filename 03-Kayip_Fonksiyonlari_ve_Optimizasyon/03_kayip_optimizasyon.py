"""
================================================================================
MODÜL 03: KAYIP FONKSİYONLARI VE OPTİMİZASYON
================================================================================

Bu dosya Deep Learning Path'in üçüncü modülüdür. Kayıp fonksiyonlarını ve
optimizasyon algoritmalarını matematiksel temelden, görsel analizle ve üç
framework karşılaştırmasıyla öğretir.

KAPSANAN KONULAR:
  KAYIP FONKSİYONLARI:
    - MSE (Mean Squared Error) — Ortalama Kare Hata
    - MAE (Mean Absolute Error) — Ortalama Mutlak Hata
    - Huber Loss — MSE + MAE hibrit
    - Binary Cross-Entropy (BCE) — ikili sınıflandırma
    - Categorical Cross-Entropy (CCE) — çok sınıflı
    - KL Divergence — dağılım farkı

  OPTİMİZASYON ALGORİTMALARI:
    - SGD (Stochastic Gradient Descent)
    - SGD + Momentum
    - RMSProp
    - Adam (Adaptive Moment Estimation)
    - AdaGrad
    - Nadam

  ÖĞRENME HIZI PROGRAMLARI:
    - Sabit (constant)
    - Adım azalma (step decay)
    - Üstel azalma (exponential decay)
    - Cosine annealing
    - Warmup + decay

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

ÇALIŞTIRMA:
  python 03_kayip_optimizasyon.py

YAZAR: Deep Learning Path — Modül 03
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  MODÜL 03: KAYIP FONKSİYONLARI VE OPTİMİZASYON")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: KAYIP FONKSİYONLARI — NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Kayıp Fonksiyonları — NumPy Implementasyonu")
print("=" * 65)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE — Mean Squared Error (Ortalama Kare Hata).
    Formül : L = (1/n) · Σ (yᵢ − ŷᵢ)²
    Türev  : dL/dŷ = (2/n) · (ŷ − y)
    Kullanım: Regresyon problemleri.
    Avantaj : Büyük hatalara daha fazla ceza (kare aldığı için).
    Sorun   : Aykırı değerlere (outlier) çok hassas.
    """
    return float(np.mean((y_true - y_pred) ** 2))

def mse_deriv(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (2.0 / len(y_true)) * (y_pred - y_true)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE — Mean Absolute Error (Ortalama Mutlak Hata).
    Formül : L = (1/n) · Σ |yᵢ − ŷᵢ|
    Türev  : dL/dŷ = (1/n) · sign(ŷ − y)
    Kullanım: Aykırı değerlere dayanıklı regresyon.
    Avantaj : Aykırı değerlere karşı MSE'den sağlam.
    Sorun   : Türev y=ŷ'de tanımsız (kırık nokta).
    """
    return float(np.mean(np.abs(y_true - y_pred)))

def mae_deriv(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sign(y_pred - y_true) / len(y_true)

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray,
               delta: float = 1.0) -> float:
    """
    Huber Loss — MSE + MAE hibrit.
    Formül:
      |hata| ≤ δ → (1/2)·hata²      (MSE gibi — küçük hatalar için smooth)
      |hata| > δ → δ·(|hata|−δ/2)   (MAE gibi — büyük hatalar için sağlam)
    Kullanım: Aykırı değer olan regresyon problemleri.
    Avantaj : Hem MSE'nin pürüzsüzlüğü hem MAE'nin sağlamlığı.
    """
    err = np.abs(y_true - y_pred)
    small = 0.5 * err ** 2
    large = delta * (err - 0.5 * delta)
    return float(np.mean(np.where(err <= delta, small, large)))

def binary_cross_entropy(y_true: np.ndarray,
                          y_pred: np.ndarray) -> float:
    """
    Binary Cross-Entropy (BCE) — İkili Sınıflandırma Kaybı.
    Formül : L = -(1/n) · Σ [yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ)]
    Türev  : dL/dŷ = (1/n) · (ŷ−y) / (ŷ·(1−ŷ))  [Sigmoid çıkışıyla basitleşir]
    Kullanım: İkili sınıflandırma (0/1 çıkışı olan problemler).
    Neden?: MSE'den daha hızlı öğrenir — log ölçeği yanlış tahminlere büyük ceza verir.
    """
    eps = 1e-15
    yp = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp)))

def categorical_cross_entropy(y_true: np.ndarray,
                               y_pred: np.ndarray) -> float:
    """
    Categorical Cross-Entropy (CCE) — Çok Sınıflı Sınıflandırma Kaybı.
    Formül : L = -(1/n) · Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)
    Kullanım: Çok sınıflı sınıflandırma (Softmax çıkışıyla birlikte).
    y_true : One-hot encoded → [[1,0,0], [0,1,0], ...]
    y_pred : Softmax çıktısı → [[0.8,0.1,0.1], ...]
    """
    eps = 1e-15
    yp = np.clip(y_pred, eps, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(yp), axis=1)))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL Divergence (Kullback-Leibler Sapması).
    Formül : KL(P‖Q) = Σ P(x) · log(P(x)/Q(x))
    Kullanım: VAE kayıp fonksiyonu, dağılım karşılaştırma.
    NOT: KL(P‖Q) ≠ KL(Q‖P) — simetrik değil!
    Yorum : P dağılımını Q ile kodlamak için gereken ekstra bit sayısı.
    """
    eps = 1e-15
    p_c = np.clip(p, eps, 1.0)
    q_c = np.clip(q, eps, 1.0)
    return float(np.sum(p_c * np.log(p_c / q_c)))


# ── Kayıp fonksiyonlarını test et ─────────────────────────────────────────────
print("\n  Kayıp Fonksiyonu Karşılaştırması:")
y_true_reg  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred_good = np.array([1.1, 1.9, 3.2, 3.8, 5.1])   # İyi tahmin
y_pred_bad  = np.array([1.5, 3.0, 1.5, 6.0, 3.5])   # Kötü tahmin + aykırı

print(f"\n  {'Kayıp':<22} {'İyi tahmin':>12} {'Kötü tahmin':>13}")
print(f"  {'-'*50}")
for name, fn in [('MSE', mse), ('MAE', mae),
                  ('Huber (δ=1)', lambda a,b: huber_loss(a,b,1.0))]:
    good = fn(y_true_reg, y_pred_good)
    bad  = fn(y_true_reg, y_pred_bad)
    print(f"  {name:<22} {good:>12.4f} {bad:>13.4f}")

print("\n  Binary Cross-Entropy Örneği:")
y_cls  = np.array([1.0, 0.0, 1.0, 0.0])
y_conf = np.array([0.9, 0.1, 0.8, 0.2])   # Güvenli tahmin
y_unc  = np.array([0.6, 0.4, 0.55, 0.45]) # Belirsiz tahmin
print(f"  BCE (güvenli): {binary_cross_entropy(y_cls, y_conf):.4f}")
print(f"  BCE (belirsiz): {binary_cross_entropy(y_cls, y_unc):.4f}")


# ============================================================
# SECTION 2: OPTİMİZASYON ALGORİTMALARI — NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: Optimizasyon Algoritmaları — NumPy")
print("=" * 65)

# Test fonksiyonu: Rosenbrock — optimizasyon algoritmalarını test etmek için klasik
# f(x,y) = (1-x)² + 100(y-x²)²   minimum: (1,1) → f=0
def rosenbrock(x: float, y: float) -> float:
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x: float, y: float) -> tuple:
    df_dx = -2*(1-x) - 400*x*(y - x**2)
    df_dy = 200*(y - x**2)
    return df_dx, df_dy

# Daha basit test: Karesel kayıp — y = w*x, veri üretiyoruz
# Gerçek w = 2.0, b = 1.0 — optimizerlar bunu bulacak
np.random.seed(0)
X_opt = np.random.randn(100)
Y_opt = 2.0 * X_opt + 1.0 + 0.1 * np.random.randn(100)

def compute_grad(w: float, b: float) -> tuple:
    """Doğrusal regresyon gradyanları."""
    y_pred = w * X_opt + b
    err    = y_pred - Y_opt
    dw = (2/len(X_opt)) * np.dot(err, X_opt)
    db = (2/len(X_opt)) * np.sum(err)
    return dw, db

def compute_loss_linear(w: float, b: float) -> float:
    return float(np.mean((w * X_opt + b - Y_opt)**2))


class SGD:
    """
    Stochastic Gradient Descent.
    Güncelleme: W ← W − η·∇W
    En basit optimizer. Gürültülü ama basit.
    Sorun: Tüm parametrelere aynı lr uygulanır; yavaş yakınsama.
    """
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, w: float, b: float) -> tuple:
        dw, db = compute_grad(w, b)
        return w - self.lr * dw, b - self.lr * db


class SGDMomentum:
    """
    SGD + Momentum.
    Güncelleme:
      v ← β·v − η·∇W     (hız vektörü — geçmiş gradyanların ağırlıklı ortalaması)
      W ← W + v
    Avantaj: Yerel minimumlardan çıkmaya yardımcı, daha hızlı yakınsama.
    β = 0.9 tipik değer.
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr   = lr
        self.beta = beta
        self.vw   = 0.0
        self.vb   = 0.0

    def update(self, w: float, b: float) -> tuple:
        dw, db  = compute_grad(w, b)
        self.vw = self.beta * self.vw - self.lr * dw
        self.vb = self.beta * self.vb - self.lr * db
        return w + self.vw, b + self.vb


class RMSProp:
    """
    RMSProp (Root Mean Square Propagation — Hinton, 2012).
    Güncelleme:
      s ← ρ·s + (1−ρ)·(∇W)²     (kare gradyan hareketli ortalaması)
      W ← W − (η/√(s+ε))·∇W
    Avantaj: Her parametreye uyarlanmış lr. Tekrarlayan sinir ağlarında iyi.
    ρ = 0.9, ε = 1e-8 tipik değerler.
    """
    def __init__(self, lr: float = 0.01, rho: float = 0.9, eps: float = 1e-8):
        self.lr  = lr
        self.rho = rho
        self.eps = eps
        self.sw  = 0.0
        self.sb  = 0.0

    def update(self, w: float, b: float) -> tuple:
        dw, db  = compute_grad(w, b)
        self.sw = self.rho * self.sw + (1 - self.rho) * dw**2
        self.sb = self.rho * self.sb + (1 - self.rho) * db**2
        w_new   = w - (self.lr / np.sqrt(self.sw + self.eps)) * dw
        b_new   = b - (self.lr / np.sqrt(self.sb + self.eps)) * db
        return w_new, b_new


class Adam:
    """
    Adam — Adaptive Moment Estimation (Kingma & Ba, 2014).
    Momentum + RMSProp kombinasyonu. En yaygın kullanılan optimizer.

    Güncelleme:
      m ← β₁·m + (1−β₁)·∇W           (1. moment — gradyan ortalaması)
      v ← β₂·v + (1−β₂)·(∇W)²        (2. moment — kare gradyan ortalaması)
      m̂ = m / (1−β₁ᵗ)                 (bias düzeltmesi)
      v̂ = v / (1−β₂ᵗ)                 (bias düzeltmesi)
      W ← W − η · m̂ / (√v̂ + ε)

    Varsayılan: η=0.001, β₁=0.9, β₂=0.999, ε=1e-8
    """
    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.mw = self.vw = 0.0
        self.mb = self.vb = 0.0
        self.t  = 0

    def update(self, w: float, b: float) -> tuple:
        self.t += 1
        dw, db = compute_grad(w, b)

        # 1. moment güncelleme
        self.mw = self.beta1 * self.mw + (1 - self.beta1) * dw
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db

        # 2. moment güncelleme
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * dw**2
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * db**2

        # Bias düzeltmesi
        mw_hat = self.mw / (1 - self.beta1**self.t)
        mb_hat = self.mb / (1 - self.beta1**self.t)
        vw_hat = self.vw / (1 - self.beta2**self.t)
        vb_hat = self.vb / (1 - self.beta2**self.t)

        w_new = w - self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
        b_new = b - self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
        return w_new, b_new


class AdaGrad:
    """
    AdaGrad — Adaptive Gradient (Duchi et al., 2011).
    Güncelleme:
      G ← G + (∇W)²
      W ← W − (η/√(G+ε)) · ∇W
    Avantaj: Seyrek (sparse) gradyanlar için iyi — NLP görevlerinde.
    Sorun: G sürekli birikir → lr sonunda sıfıra yaklaşır → öğrenme durur.
    """
    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        self.lr  = lr
        self.eps = eps
        self.Gw  = 0.0
        self.Gb  = 0.0

    def update(self, w: float, b: float) -> tuple:
        dw, db  = compute_grad(w, b)
        self.Gw += dw**2
        self.Gb += db**2
        w_new   = w - (self.lr / np.sqrt(self.Gw + self.eps)) * dw
        b_new   = b - (self.lr / np.sqrt(self.Gb + self.eps)) * db
        return w_new, b_new


# ── Optimizerleri eğit ve karşılaştır ─────────────────────────────────────────
EPOCHS   = 200
W_INIT   = 0.0
B_INIT   = 0.0

optimizers = {
    'SGD (lr=0.1)':       SGD(lr=0.1),
    'SGD+Momentum':       SGDMomentum(lr=0.05, beta=0.9),
    'RMSProp':            RMSProp(lr=0.01),
    'Adam':               Adam(lr=0.05),
    'AdaGrad':            AdaGrad(lr=0.1),
}

results = {}
print("\n  Optimizer Karşılaştırması (Doğrusal Regresyon: y = 2x + 1 bul)")
print(f"  {'Optimizer':<18} {'Final Loss':>10} {'W (gerçek=2)':>14} {'b (gerçek=1)':>14}")
print(f"  {'-'*60}")

for name, opt in optimizers.items():
    w, b      = W_INIT, B_INIT
    loss_hist = []
    for _ in range(EPOCHS):
        loss_hist.append(compute_loss_linear(w, b))
        w, b = opt.update(w, b)
    results[name] = {'loss': loss_hist, 'w': w, 'b': b}
    print(f"  {name:<18} {loss_hist[-1]:>10.6f} {w:>14.4f} {b:>14.4f}")


# ============================================================
# SECTION 3: ÖĞRENME HIZI PROGRAMLARI
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: Öğrenme Hızı Programları (LR Schedulers)")
print("=" * 65)

def step_decay(epoch: int, lr0: float = 0.1,
               drop: float = 0.5, epochs_drop: int = 50) -> float:
    """
    Adım Azalma (Step Decay).
    Her `epochs_drop` epoch'ta lr, `drop` katsayısıyla azalır.
    lr(t) = lr₀ · drop^floor(epoch / epochs_drop)
    """
    return lr0 * (drop ** (epoch // epochs_drop))

def exponential_decay(epoch: int, lr0: float = 0.1,
                       k: float = 0.01) -> float:
    """
    Üstel Azalma (Exponential Decay).
    lr(t) = lr₀ · e^(−k·t)
    """
    return lr0 * np.exp(-k * epoch)

def cosine_annealing(epoch: int, lr_max: float = 0.1,
                      lr_min: float = 0.001, T: int = 200) -> float:
    """
    Cosine Annealing.
    lr(t) = lr_min + 0.5·(lr_max−lr_min)·[1 + cos(π·t/T)]
    Avantaj: Periyodik yeniden başlatmayla yerel minimumlardan çıkar.
    """
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))

def warmup_cosine(epoch: int, lr_max: float = 0.1,
                   warmup_epochs: int = 20, T: int = 200) -> float:
    """
    Warmup + Cosine Annealing.
    İlk warmup_epochs: lr doğrusal artar (0 → lr_max).
    Sonra: cosine annealing.
    Kullanım: Transformer modelleri, büyük batch eğitimi.
    """
    if epoch < warmup_epochs:
        return lr_max * (epoch / warmup_epochs)
    return cosine_annealing(epoch - warmup_epochs, lr_max,
                             0.001, T - warmup_epochs)

epochs = np.arange(200)
schedulers = {
    'Sabit (lr=0.1)':       [0.1] * 200,
    'Adım Azalma':          [step_decay(e) for e in epochs],
    'Üstel Azalma':         [exponential_decay(e) for e in epochs],
    'Cosine Annealing':     [cosine_annealing(e) for e in epochs],
    'Warmup + Cosine':      [warmup_cosine(e) for e in epochs],
}

print(f"\n  {'Scheduler':<22} {'Başlangıç lr':>14} {'50. epoch lr':>14} {'Son lr':>10}")
print(f"  {'-'*65}")
for name, lrs in schedulers.items():
    print(f"  {name:<22} {lrs[0]:>14.4f} {lrs[50]:>14.4f} {lrs[-1]:>10.6f}")


# ============================================================
# SECTION 4: TENSORFLOW / KERAS
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: TensorFlow / Keras")
print("=" * 65)

try:
    import tensorflow as tf
    print(f"\n  TensorFlow {tf.__version__}")
    print("""
  # Keras kayıp fonksiyonları:
  model.compile(loss='mse')                         # Regresyon
  model.compile(loss='binary_crossentropy')         # İkili sınıf
  model.compile(loss='categorical_crossentropy')    # Çok sınıf
  model.compile(loss='sparse_categorical_crossentropy') # Integer label
  model.compile(loss=tf.keras.losses.Huber(delta=1.0))  # Huber

  # Keras optimizerları:
  tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
  tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
  tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
  tf.keras.optimizers.Adagrad(learning_rate=0.01)

  # LR Scheduler:
  lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
      initial_learning_rate=0.1, decay_steps=1000)
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    """)
except ImportError:
    print("  TensorFlow yüklü değil — pip install tensorflow")


# ============================================================
# SECTION 5: PYTORCH
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: PyTorch")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    print(f"\n  PyTorch {torch.__version__}")
    print("""
  # PyTorch kayıp fonksiyonları:
  nn.MSELoss()                          # Regresyon
  nn.L1Loss()                           # MAE
  nn.HuberLoss(delta=1.0)               # Huber
  nn.BCELoss()                          # BCE (sigmoid sonrası)
  nn.BCEWithLogitsLoss()                # BCE + sigmoid birlikte (önerilen)
  nn.CrossEntropyLoss()                 # CCE + softmax birlikte
  nn.KLDivLoss(reduction='batchmean')   # KL Divergence

  # PyTorch optimizerları:
  torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
  torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
  torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
  torch.optim.Adagrad(model.parameters(), lr=0.01)

  # LR Scheduler:
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=100, eta_min=0.001)
  # Her epoch sonunda: scheduler.step()
    """)
except ImportError:
    print("  PyTorch yüklü değil — pip install torch")


# ============================================================
# SECTION 6: GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: Görselleştirme")
print("=" * 65)

COLORS_OPT = {
    'SGD (lr=0.1)': '#1565C0', 'SGD+Momentum': '#00695C',
    'RMSProp': '#E65100',       'Adam': '#AD1457',
    'AdaGrad': '#6A1B9A',
}
COLORS_SCH = {
    'Sabit (lr=0.1)': '#9E9E9E', 'Adım Azalma': '#1565C0',
    'Üstel Azalma': '#E65100',   'Cosine Annealing': '#00695C',
    'Warmup + Cosine': '#AD1457',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Modül 03: Kayıp Fonksiyonları ve Optimizasyon",
             fontsize=14, fontweight='bold')

# ── Panel 1: Kayıp Fonksiyonu Karşılaştırması ─────────────────────────────────
ax1 = axes[0, 0]
errors = np.linspace(-3, 3, 300)
ax1.plot(errors, errors**2,          label='MSE: e²',       lw=2.5, color='#1565C0')
ax1.plot(errors, np.abs(errors),     label='MAE: |e|',      lw=2.5, color='#E65100')
delta = 1.0
huber = np.where(np.abs(errors) <= delta,
                  0.5*errors**2,
                  delta*(np.abs(errors) - 0.5*delta))
ax1.plot(errors, huber, label=f'Huber (δ={delta})', lw=2.5, color='#00695C')
ax1.axvline(x=-1, color='gray', lw=1, linestyle=':', alpha=0.7)
ax1.axvline(x=1,  color='gray', lw=1, linestyle=':', alpha=0.7)
ax1.set_title("Regresyon Kayıp Fonksiyonları", fontweight='bold')
ax1.set_xlabel("Hata  (y − ŷ)")
ax1.set_ylabel("Kayıp Değeri")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 5)

# ── Panel 2: BCE vs MSE Karşılaştırması ───────────────────────────────────────
ax2 = axes[0, 1]
p_range = np.linspace(0.01, 0.99, 300)
bce_y1 = -np.log(p_range)              # y=1 için BCE
bce_y0 = -np.log(1 - p_range)          # y=0 için BCE
mse_y1 = (1 - p_range)**2              # y=1 için MSE
mse_y0 = p_range**2                    # y=0 için MSE

ax2.plot(p_range, bce_y1, label='BCE (y=1)',  lw=2.5, color='#1565C0')
ax2.plot(p_range, bce_y0, label='BCE (y=0)',  lw=2.5, color='#1565C0', linestyle='--')
ax2.plot(p_range, mse_y1, label='MSE (y=1)',  lw=2.5, color='#E65100')
ax2.plot(p_range, mse_y0, label='MSE (y=0)',  lw=2.5, color='#E65100', linestyle='--')
ax2.set_title("BCE vs MSE — Sınıflandırma Kaybı", fontweight='bold')
ax2.set_xlabel("Tahmin olasılığı ŷ")
ax2.set_ylabel("Kayıp")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 4)

# ── Panel 3: Optimizer Yakınsama Eğrileri ─────────────────────────────────────
ax3 = axes[1, 0]
for name, res in results.items():
    ax3.semilogy(res['loss'], label=name,
                 color=COLORS_OPT.get(name, 'gray'), lw=2, alpha=0.9)
ax3.set_title("Optimizer Karşılaştırması — Loss Yakınsama", fontweight='bold')
ax3.set_xlabel("Epoch")
ax3.set_ylabel("MSE Loss (log ölçek)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, EPOCHS)

# ── Panel 4: LR Scheduler Karşılaştırması ─────────────────────────────────────
ax4 = axes[1, 1]
for name, lrs in schedulers.items():
    ax4.plot(lrs, label=name, color=COLORS_SCH.get(name, 'gray'), lw=2)
ax4.set_title("Öğrenme Hızı Programları Karşılaştırması", fontweight='bold')
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Öğrenme Hızı")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/03-Kayip_Fonksiyonlari_ve_Optimizasyon/modul03_analiz.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")


# ============================================================
# SECTION 7: ÖZET
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 7: Modül 03 Özeti")
print("=" * 65)
print("""
  KAYIP FONKSİYONLARI:
  ✓ MSE: Büyük hatalara daha fazla ceza — regresyon standart tercihi
  ✓ MAE: Aykırı değerlere dayanıklı — robust regresyon
  ✓ Huber: MSE + MAE hibrit — aykırı değer olan regresyon
  ✓ BCE: İkili sınıflandırma — MSE'den çok daha hızlı öğrenir
  ✓ CCE: Çok sınıflı — Softmax çıkışıyla birlikte kullan
  ✓ KL: Dağılım farkı — VAE gibi üretici modellerde

  OPTİMİZASYON:
  ✓ SGD: Basit, anlayışlı, ancak yavaş
  ✓ Momentum: SGD + hız → daha hızlı, yerel minimumdan çıkar
  ✓ RMSProp: Uyarlanmış lr → RNN için popüler
  ✓ Adam: Momentum + RMSProp → çoğu durumda en iyi başlangıç tercihi
  ✓ AdaGrad: Sparse NLP → birikimli ölçekleme

  KURAL: Nereden başlayacağını bilmiyorsan → Adam (lr=0.001).

  SONRAKI MODÜL: 04 — Geri Yayılım (Backpropagation)
""")
print("=" * 65)
print("  Modül 03 tamamlandı!")
print("=" * 65)
