"""
================================================================================
MODÜL 02: AKTİVASYON FONKSİYONLARI — Derinlemesine İnceleme
================================================================================

Bu dosya Deep Learning Path'in ikinci modülüdür. Aktivasyon fonksiyonlarını
matematiksel temelden, türevlerini ve zayıf noktalarını, modern alternatifleri
ve hangi durumda hangisini seçeceğini kapsamlı biçimde öğretir.

KAPSANAN KONULAR:
  - Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, SELU, Swish, GELU
  - Her fonksiyon için: formül, türev, aralık, avantaj/dezavantaj
  - Vanishing Gradient problemi — matematiksel kanıt ve grafik
  - Dead Neuron (ReLU ölü nöron) problemi ve çözümleri
  - Hangi aktivasyonu ne zaman seçmeli — karar rehberi tablosu
  - NumPy, TensorFlow ve PyTorch implementasyonları
  - Tüm aktivasyonların görsel karşılaştırması

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

ÇALIŞTIRMA:
  python 02_aktivasyon_fonksiyonlari.py

BEKLENEN ÇIKTI:
  - Tüm aktivasyon fonksiyonlarının sayısal değerleri yazdırılır
  - 3 adet çok panelli grafik görüntülenir:
      1. Aktivasyon fonksiyonları karşılaştırması (8 fonksiyon)
      2. Türevler / gradyan büyüklükleri
      3. Vanishing gradient görsel kanıtı

YAZAR: Deep Learning Path — Modül 02
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  MODÜL 02: AKTİVASYON FONKSİYONLARI")
print("  Deep Learning Path — Derinlemesine İnceleme")
print("=" * 65)


# ============================================================
# SECTION 1: TÜM AKTİVASYON FONKSİYONLARI — NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Aktivasyon Fonksiyonları — NumPy Implementasyonu")
print("=" * 65)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid (Lojistik) Fonksiyon.
    Formül  : σ(z) = 1 / (1 + e^(−z))
    Aralık  : (0, 1)
    Türev   : σ'(z) = σ(z) · (1 − σ(z))
    Sorunlar: Vanishing gradient (z→±∞ türev→0), sıfır merkezli değil.
    Kullanım: Çıkış katmanı — ikili sınıflandırma.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    """Sigmoid türevi. Maksimum değer 0.25 (z=0'da)."""
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh_fn(z: np.ndarray) -> np.ndarray:
    """
    Hiperbolik Tanjant.
    Formül  : tanh(z) = (eᶻ − e^(−z)) / (eᶻ + e^(−z))
    Aralık  : (−1, 1) — sıfır merkezli, sigmoid'den iyi
    Türev   : tanh'(z) = 1 − tanh²(z), maksimum 1.0 (z=0'da)
    Sorunlar: Hâlâ vanishing gradient (sigmoid'den az).
    Kullanım: Gizli katmanlar (özellikle RNN).
    """
    return np.tanh(z)

def tanh_deriv(z: np.ndarray) -> np.ndarray:
    """Tanh türevi: 1 − tanh²(z)"""
    return 1.0 - np.tanh(z) ** 2

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU — Rectified Linear Unit.
    Formül  : ReLU(z) = max(0, z)
    Aralık  : [0, +∞)
    Türev   : 1 eğer z>0, 0 eğer z≤0
    Avantaj : Hızlı, basit, z>0'da vanishing gradient YOK.
    Sorunlar: Dead Neuron — z<0 olan nöron hiç güncellenmiyor.
    Kullanım: Modern gizli katmanların standart tercihi.
    """
    return np.maximum(0.0, z)

def relu_deriv(z: np.ndarray) -> np.ndarray:
    """ReLU türevi: z>0 → 1, z≤0 → 0."""
    return (z > 0).astype(float)

def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU.
    Formül  : max(α·z, z)  — α genellikle 0.01
    Aralık  : (−∞, +∞)
    Türev   : 1 eğer z>0, α eğer z≤0
    Avantaj : Dead Neuron problemini çözer (z<0 hâlâ gradyan alır).
    Sorunlar: α hiperparametresi seçimi; negatif bölge tutarsız olabilir.
    Kullanım: ReLU'nun dead neuron sorunu yaşadığı durumlarda.
    """
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU türevi."""
    return np.where(z > 0, 1.0, alpha)

def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    ELU — Exponential Linear Unit.
    Formül  : z eğer z>0,  α·(eᶻ − 1) eğer z≤0
    Aralık  : (−α, +∞)
    Türev   : 1 eğer z>0,  ELU(z)+α eğer z≤0
    Avantaj : Sıfır merkezli, smooth negatif doyma, dead neuron yok.
    Sorunlar: Üstel hesaplama (exponential) ReLU'dan yavaş.
    Kullanım: Derin ağlarda ReLU alternatifi, daha iyi yakınsama.
    """
    return np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1))

def elu_deriv(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU türevi: z>0→1, z≤0→elu(z)+α"""
    return np.where(z > 0, 1.0, elu(z, alpha) + alpha)

def selu(z: np.ndarray) -> np.ndarray:
    """
    SELU — Scaled Exponential Linear Unit.
    Formül  : λ · ELU(z, α) — sabitler: λ≈1.0507, α≈1.6733
    Aralık  : (−λ·α, +∞) ≈ (−1.758, +∞)
    Avantaj : Self-normalizing! Derin ağlarda otomatik ortalaması→0, varyansı→1.
              Batch Normalization'a gerek kalmaz.
    Sorunlar: Sadece tam bağlantılı katmanlarda tam çalışır.
    Kullanım: Çok derin fully-connected ağlar.
    """
    alpha_s = 1.6732632423543772
    lambda_s = 1.0507009873554804
    return lambda_s * np.where(z > 0, z, alpha_s * (np.exp(np.clip(z, -500, 0)) - 1))

def selu_deriv(z: np.ndarray) -> np.ndarray:
    """SELU türevi."""
    alpha_s = 1.6732632423543772
    lambda_s = 1.0507009873554804
    return lambda_s * np.where(z > 0, 1.0,
                                alpha_s * np.exp(np.clip(z, -500, 0)))

def swish(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish (Google Brain, 2017).
    Formül  : z · σ(β·z)
    Aralık  : (≈−0.278, +∞)
    Türev   : σ(βz) + βz·σ(βz)·(1−σ(βz))
    Avantaj : Pürüzsüz, sıfır merkezli değil ama ReLU'dan iyi.
              Büyük ölçekli ImageNet'te ReLU'yu geçiyor.
    Kullanım: EfficientNet ailesi, büyük CNN'ler.
    """
    return z * sigmoid(beta * z)

def swish_deriv(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish türevi."""
    s = sigmoid(beta * z)
    return s + beta * z * s * (1 - s)

def gelu(z: np.ndarray) -> np.ndarray:
    """
    GELU — Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016).
    Formül  : z · Φ(z)  ≈  0.5·z·[1 + tanh(√(2/π)·(z + 0.044715·z³))]
    Aralık  : (≈−0.17, +∞)
    Avantaj : BERT, GPT-2, GPT-3, ViT — modern transformer mimarilerinin standart tercihi.
              Smooth, stochastic regularization etkisi var.
    Kullanım: Transformer tabanlı modeller, NLP, modern vision modeller.
    """
    # Yaygın yaklaşım (tam formül yerine)
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))

def gelu_deriv(z: np.ndarray) -> np.ndarray:
    """GELU türevi (yaklaşık)."""
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return cdf + z * pdf


# ── Tüm aktivasyonları bir sözlükte topla ──────────────────────────────────────
ACTIVATIONS = {
    'Sigmoid':     (sigmoid,     sigmoid_deriv),
    'Tanh':        (tanh_fn,     tanh_deriv),
    'ReLU':        (relu,        relu_deriv),
    'Leaky ReLU':  (leaky_relu,  leaky_relu_deriv),
    'ELU':         (elu,         elu_deriv),
    'SELU':        (selu,        selu_deriv),
    'Swish':       (swish,       swish_deriv),
    'GELU':        (gelu,        gelu_deriv),
}

# ── Sayısal karşılaştırma tablosu ─────────────────────────────────────────────
print("\n  Tüm aktivasyonlar için z={-2, -1, 0, 1, 2} değerleri:")
z_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\n  {'Fonksiyon':<14} " + "  ".join(f"z={z:>4.0f}" for z in z_test))
print(f"  {'-'*60}")
for name, (fn, _) in ACTIVATIONS.items():
    vals = fn(z_test)
    print(f"  {name:<14} " + "  ".join(f"{v:>7.4f}" for v in vals))

print(f"\n  Türev değerleri (z={0}):")
print(f"  {'Fonksiyon':<14}  Türev@z=0  Türev@z=3  Maksimum Gradyan")
print(f"  {'-'*55}")
for name, (fn, dfn) in ACTIVATIONS.items():
    d0 = dfn(np.array([0.0]))[0]
    d3 = dfn(np.array([3.0]))[0]
    z_range = np.linspace(-5, 5, 1000)
    d_max = np.max(np.abs(dfn(z_range)))
    print(f"  {name:<14}  {d0:>9.4f}  {d3:>9.4f}  {d_max:>15.4f}")


# ============================================================
# SECTION 2: VANİSHİNG GRADİENT — KANIT VE GÖRSEL
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: Vanishing Gradient Problemi — Matematiksel Kanıt")
print("=" * 65)

print("""
  Problem:
  Derin bir ağda backprop sırasında gradyanlar katman katman çarpılır.
  Her katmanda sigmoid türevi ≤ 0.25 çarpılınca:

    10 katmanlı ağda: 0.25^10 ≈ 0.000001  (1 milyonda 1!)

  Bu demek ki ilk katmanın ağırlıkları neredeyse hiç güncellenmez.
  Ağ öğrenemiyor görünür ama aslında sadece ilk katmanlar öğrenemiyor.
""")

# Katman sayısıyla gradyan büyüklüğü değişimi
n_layers = np.arange(1, 21)
sigmoid_max_grad = 0.25  # Sigmoid türevinin maksimumu
tanh_max_grad    = 1.0   # Tanh türevinin maksimumu
relu_max_grad    = 1.0   # ReLU türevinin maksimumu (z>0'da)

# Her katmanla gradyan nasıl küçülür? (en kötü durum değil, tipik durum)
sigmoid_grad_decay = sigmoid_max_grad ** n_layers
tanh_grad_decay    = (0.7) ** n_layers  # Tanh tipik ≈ 0.7
relu_grad_decay    = np.ones_like(n_layers, dtype=float)  # ReLU: 1 (z>0'da)

print("  Katman Sayısı vs Gradyan Büyüklüğü:")
print(f"  {'Katman':>8} | {'Sigmoid gradyanı':>18} | {'Tanh gradyanı':>15} | {'ReLU gradyanı':>14}")
print(f"  {'-'*60}")
for i in [1, 2, 3, 5, 10, 15, 20]:
    sg = sigmoid_max_grad ** i
    tg = (0.7) ** i
    rg = 1.0
    print(f"  {i:>8} | {sg:>18.10f} | {tg:>15.8f} | {rg:>14.4f}")


# ============================================================
# SECTION 3: DEAD NEURON PROBLEMİ
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: Dead Neuron (Ölü Nöron) Problemi")
print("=" * 65)

print("""
  Problem:
  ReLU'da z ≤ 0 olan bir nöron için:
    - Çıkış: max(0, z) = 0
    - Türev: 0
    - Gradyan: 0 × (önceki katman gradyanı) = 0
    
  Bu nöron hiç gradyan almaz → ağırlıkları hiç güncellenmez → "ölü".
  
  Ne zaman olur?
    - Öğrenme hızı çok büyükse
    - Ağırlık başlatma hatalıysa
    - Negatif bias değerleri
  
  Çözümler:
    1. Leaky ReLU: z<0 için α=0.01 gradyan → nöron canlı kalır
    2. ELU: z<0 için smooth negatif değer → gradyan var
    3. Öğrenme hızını düşürmek
    4. He başlatma kullanmak
""")

# Dead neuron simülasyonu
np.random.seed(0)
n_neurons = 1000
z_init = np.random.randn(n_neurons)  # Başlangıç z değerleri

# Büyük lr güncellemesi sonrası z değerleri
big_lr_update = np.random.uniform(1, 5, n_neurons)
z_after_big_lr = z_init - big_lr_update

dead_relu  = np.sum(z_after_big_lr < 0)
dead_leaky = 0  # Leaky ReLU hiçbir zaman "tamamen" ölmüyor

print(f"  Simülasyon: {n_neurons} nöron, büyük öğrenme hızı güncellemesi sonrası:")
print(f"  ReLU'da ölü nöron sayısı  : {dead_relu} / {n_neurons}  ({dead_relu/n_neurons:.1%})")
print(f"  Leaky ReLU'da ölü nöron   : {dead_leaky} / {n_neurons}  (0.0%)")
print(f"  ELU'da ölü nöron          : {dead_leaky} / {n_neurons}  (0.0%)")


# ============================================================
# SECTION 4: HANGİ AKTİVASYONU NE ZAMAN SEÇMEK?
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: Aktivasyon Seçim Rehberi")
print("=" * 65)

print("""
  ┌──────────────────────────────────────────────────────────────┐
  │                 AKTİVASYON SEÇİM KARAR REHBERİ              │
  ├─────────────────────┬────────────────────────────────────────┤
  │ Durum               │ Önerilen Aktivasyon                    │
  ├─────────────────────┼────────────────────────────────────────┤
  │ Çıkış - İkili sınıf │ Sigmoid                               │
  │ Çıkış - Çok sınıf   │ Softmax                               │
  │ Çıkış - Regresyon   │ Linear (aktivasyon yok)               │
  │ Gizli - Genel       │ ReLU (başlangıç noktası)              │
  │ Gizli - Deep CNN    │ ReLU veya Leaky ReLU                  │
  │ Gizli - Transformer │ GELU (BERT, GPT standart)             │
  │ Gizli - EfficientNet│ Swish                                  │
  │ Çok derin FC        │ SELU (self-normalizing)               │
  │ Dead neuron sorunu  │ Leaky ReLU veya ELU                   │
  │ RNN/LSTM gizli      │ Tanh                                   │
  └─────────────────────┴────────────────────────────────────────┘
""")


# ============================================================
# SECTION 5: TENSORFLOW/KERAS
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: TensorFlow / Keras Aktivasyonları")
print("=" * 65)

try:
    import tensorflow as tf
    print(f"\n  TensorFlow {tf.__version__}")
    print("""
  # Keras'ta aktivasyon kullanımı — 3 farklı yöntem:

  # Yöntem 1: String olarak
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='gelu'),
      tf.keras.layers.Dense(1,  activation='sigmoid'),
  ])

  # Yöntem 2: Fonksiyon olarak
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
  ])

  # Yöntem 3: Activation layer olarak (daha esnek)
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(64),
      tf.keras.layers.Activation('relu'),
  ])

  # Leaky ReLU (özel alpha)
  model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

  # ELU
  model.add(tf.keras.layers.ELU(alpha=1.0))
    """)
    # Sayısal doğrulama
    z_tf = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("  TF Doğrulama (z=[-2,-1,0,1,2]):")
    print(f"    ReLU : {tf.keras.activations.relu(z_tf).numpy()}")
    print(f"    GELU : {tf.keras.activations.gelu(z_tf).numpy().round(4)}")
    print(f"    Swish: {tf.keras.activations.swish(z_tf).numpy().round(4)}")
except ImportError:
    print("  TensorFlow yüklü değil — pip install tensorflow")


# ============================================================
# SECTION 6: PYTORCH
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: PyTorch Aktivasyonları")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    print(f"\n  PyTorch {torch.__version__}")
    print("""
  # PyTorch'ta aktivasyon kullanımı:

  import torch.nn as nn

  # nn.Sequential içinde
  model = nn.Sequential(
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.GELU(),          # Transformer'larda standart
      nn.Linear(64, 1),
      nn.Sigmoid(),
  )

  # Fonksiyonel API (F) — forward pass içinde
  import torch.nn.functional as F
  x = F.relu(self.linear1(x))
  x = F.gelu(self.linear2(x))

  # Leaky ReLU
  nn.LeakyReLU(negative_slope=0.01)

  # ELU
  nn.ELU(alpha=1.0)

  # SELU
  nn.SELU()

  # Swish (SiLU olarak PyTorch'ta)
  nn.SiLU()    # Swish = SiLU
    """)
    # Sayısal doğrulama
    z_pt = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("  PyTorch Doğrulama (z=[-2,-1,0,1,2]):")
    print(f"    ReLU : {nn.ReLU()(z_pt).numpy()}")
    print(f"    GELU : {nn.GELU()(z_pt).numpy().round(4)}")
    print(f"    SiLU : {nn.SiLU()(z_pt).numpy().round(4)}")
    print(f"    ELU  : {nn.ELU()(z_pt).numpy().round(4)}")
except ImportError:
    print("  PyTorch yüklü değil — pip install torch")


# ============================================================
# SECTION 7: GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 7: Görselleştirme")
print("=" * 65)

z = np.linspace(-4, 4, 400)

# Renkler
COLORS = {
    'Sigmoid':    '#1565C0',
    'Tanh':       '#00695C',
    'ReLU':       '#E65100',
    'Leaky ReLU': '#6A1B9A',
    'ELU':        '#AD1457',
    'SELU':       '#2E7D32',
    'Swish':      '#F57F17',
    'GELU':       '#880E4F',
}

# ── Grafik 1: Aktivasyon Fonksiyonları ────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Modül 02: Aktivasyon Fonksiyonları Karşılaştırması",
             fontsize=14, fontweight='bold')

for ax, (name, (fn, dfn)) in zip(axes.flatten(), ACTIVATIONS.items()):
    vals = fn(z)
    ax.plot(z, vals, color=COLORS[name], linewidth=2.5, label='f(z)')
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.8, alpha=0.5)
    ax.set_title(name, fontweight='bold', fontsize=11, color=COLORS[name])
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('z', fontsize=9)
    ax.set_ylabel('f(z)', fontsize=9)
    # Aralık bilgisi
    vmin, vmax = np.min(vals), np.max(vals)
    ax.text(0.98, 0.05, f'aralık: ({vmin:.2f}, {vmax:.2f})',
            transform=ax.transAxes, ha='right', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/02-Aktivasyon_Fonksiyonlari/modul02_aktivasyonlar.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── Grafik 2: Türevler ────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
fig2.suptitle("Modül 02: Aktivasyon Türevleri (Gradyan Büyüklükleri)",
              fontsize=14, fontweight='bold')

for ax, (name, (fn, dfn)) in zip(axes2.flatten(), ACTIVATIONS.items()):
    derivs = dfn(z)
    ax.plot(z, derivs, color=COLORS[name], linewidth=2.5, linestyle='--')
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.8, alpha=0.5)
    ax.set_title(f"{name} — Türev", fontweight='bold', fontsize=10, color=COLORS[name])
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('z', fontsize=9)
    ax.set_ylabel("f'(z)", fontsize=9)
    max_d = np.max(np.abs(derivs[np.isfinite(derivs)]))
    ax.text(0.98, 0.95, f'max |grad|: {max_d:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/02-Aktivasyon_Fonksiyonlari/modul02_turevler.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── Grafik 3: Vanishing Gradient Kanıtı ──────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Modül 02: Vanishing Gradient Problemi", fontsize=13, fontweight='bold')

# Sol: Gradient büyüklüğü vs katman sayısı
ax3a = axes3[0]
layers = np.arange(1, 21)
sig_grad   = 0.25 ** layers          # Sigmoid: maks türev 0.25
tanh_grad  = (0.7 ** layers)         # Tanh: tipik ~0.7
relu_grad  = np.ones(len(layers))    # ReLU: z>0'da her zaman 1

ax3a.semilogy(layers, sig_grad,  'b-o', label='Sigmoid (max 0.25)', lw=2, ms=5)
ax3a.semilogy(layers, tanh_grad, 'g-s', label='Tanh (tipik 0.7)',   lw=2, ms=5)
ax3a.semilogy(layers, relu_grad, 'r-^', label='ReLU (z>0 → 1.0)',  lw=2, ms=5)
ax3a.axhline(y=1e-4, color='gray', linestyle=':', alpha=0.7, label='Eşik: 1e-4')
ax3a.set_xlabel('Katman Sayısı')
ax3a.set_ylabel('Gradyan Büyüklüğü (log ölçek)')
ax3a.set_title('Derinlikle Gradyan Yok Oluşu')
ax3a.legend(fontsize=9)
ax3a.grid(True, alpha=0.3)
ax3a.fill_between(layers, 0, 1e-4, alpha=0.1, color='red', label='Vanishing bölge')

# Sağ: Sigmoid türevinin z'ye göre değişimi
ax3b = axes3[1]
z_wide = np.linspace(-8, 8, 400)
ax3b.plot(z_wide, sigmoid_deriv(z_wide), 'b-', lw=2.5, label="σ'(z)")
ax3b.fill_between(z_wide, 0, sigmoid_deriv(z_wide), alpha=0.2, color='blue')
ax3b.axhline(y=0.25, color='red', linestyle='--', lw=1.5, label='Maks = 0.25')
ax3b.axhspan(0, 0.01, alpha=0.15, color='red', label='Etkisiz bölge (<0.01)')
ax3b.set_xlabel('z (ağırlıklı toplam)')
ax3b.set_ylabel("σ'(z)")
ax3b.set_title('Sigmoid Türevi — Çoğu z Değerinde Sıfıra Yakın!')
ax3b.legend(fontsize=9)
ax3b.grid(True, alpha=0.3)
ax3b.set_ylim(-0.02, 0.30)

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/02-Aktivasyon_Fonksiyonlari/modul02_vanishing_gradient.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("  Grafikler kaydedildi.")


# ============================================================
# SECTION 8: ÖZET
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 8: Modül 02 Özeti")
print("=" * 65)
print("""
  ✓ Sigmoid: (0,1) çıkış — sadece çıkış katmanında kullan
  ✓ Tanh: (−1,1) çıkış — RNN gizli katmanları için
  ✓ ReLU: z>0 → vanishing yok — varsayılan gizli katman tercihi
  ✓ Leaky ReLU: dead neuron problemi için ReLU alternatifi
  ✓ ELU: smooth, sıfır merkezli, daha iyi yakınsama
  ✓ SELU: self-normalizing — çok derin FC ağlar için
  ✓ Swish: EfficientNet, büyük CNN'ler
  ✓ GELU: Transformer mimarileri standart tercihi (BERT, GPT)

  KURAL: Nereden başlayacağını bilmiyorsan → ReLU.
         Transformer yazıyorsan → GELU.
         Dead neuron problemi yaşıyorsan → Leaky ReLU veya ELU.

  SONRAKI MODÜL: 03 — Kayıp Fonksiyonları ve Optimizasyon
""")
print("=" * 65)
print("  Modül 02 tamamlandı!")
print("=" * 65)
