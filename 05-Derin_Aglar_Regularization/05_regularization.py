"""
================================================================================
MODÜL 05: DERİN AĞLAR VE REGULARİZATİON
================================================================================

Bu dosya Deep Learning Path'in beşinci modülüdür. Overfitting sorununu,
bias-variance tradeoff'unu ve tüm düzenlileştirme tekniklerini kapsar.

KAPSANAN KONULAR:
  - Overfitting vs Underfitting — bias-variance tradeoff
  - L1 (Lasso) ve L2 (Ridge / Weight Decay) regularization
  - Dropout — eğitim vs çıkarım farkı, inverted dropout
  - Batch Normalization — tam algoritma, neden işe yarar
  - Layer Normalization — Transformer'lardaki tercih
  - Early Stopping — validation loss takibi
  - Data Augmentation — görüntü ve metin için
  - Her tekniğin NumPy, TF ve PyTorch implementasyonu

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib scikit-learn

ÇALIŞTIRMA:
  python 05_regularization.py

YAZAR: Deep Learning Path — Modül 05
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 65)
print("  MODÜL 05: REGULARİZATİON")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: OVERFİTTİNG vs UNDERFİTTİNG
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Overfitting vs Underfitting")
print("=" * 65)

# Sinüs verisi üret
X_data = np.linspace(0, 2*np.pi, 30)
y_data = np.sin(X_data) + 0.3 * np.random.randn(30)

def fit_polynomial(X: np.ndarray, y: np.ndarray,
                   degree: int) -> np.ndarray:
    """Polinom fit — belirli derecede."""
    coeffs = np.polyfit(X, y, degree)
    return np.poly1d(coeffs)

X_test = np.linspace(0, 2*np.pi, 200)

# Underfitting: derece 1 (düz çizgi)
p1  = fit_polynomial(X_data, y_data, 1)
# İyi fit: derece 4
p4  = fit_polynomial(X_data, y_data, 4)
# Overfitting: derece 20
p20 = fit_polynomial(X_data, y_data, 20)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print(f"\n  {'Model':<22} {'Eğitim MSE':>12} {'Test MSE':>12} {'Durum'}")
print(f"  {'-'*60}")
for name, poly, degree in [
    ("Derece 1 (Lineer)", p1, 1),
    ("Derece 4",          p4, 4),
    ("Derece 20",         p20, 20),
]:
    train_mse = mse(y_data, poly(X_data))
    test_mse  = mse(np.sin(X_test), poly(X_test))
    status = ("✗ Underfitting" if degree == 1
              else "✓ İyi fit" if degree == 4
              else "✗ Overfitting")
    print(f"  {name:<22} {train_mse:>12.4f} {test_mse:>12.4f}  {status}")

print("""
  Bias-Variance Tradeoff:
    Toplam Hata = Bias² + Variance + Gürültü

    Yüksek Bias (Underfitting):
      - Model çok basit, veriyi öğrenemiyor
      - Hem train hem test hatası yüksek

    Yüksek Variance (Overfitting):
      - Model eğitim verisini ezberledi
      - Train hatası düşük, test hatası yüksek
      - Çözüm: Regularization!
""")


# ============================================================
# SECTION 2: L1 ve L2 REGULARİZASYON
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: L1 (Lasso) ve L2 (Ridge) Regularization")
print("=" * 65)

print("""
  Temel fikir: Kayıp fonksiyonuna ceza terimi ekle.

  L2 (Ridge / Weight Decay):
    L_total = L_data + (λ/2) · Σ wᵢ²
    Türev  : ∂L/∂w = ∂L_data/∂w + λ·w
    Güncelleme: w ← w·(1−η·λ) − η·∂L_data/∂w
    Etki: Ağırlıkları sıfıra yaklaştırır ama sıfır YAPMAZ.
    Kullanım: Genel regularization — çoğu durumda L1'den iyi.

  L1 (Lasso):
    L_total = L_data + λ · Σ |wᵢ|
    Türev  : ∂L/∂w = ∂L_data/∂w + λ·sign(w)
    Güncelleme: w ← w − η·λ·sign(w) − η·∂L_data/∂w
    Etki: Bazı ağırlıkları tam olarak SIFIRA düşürür → seyrek model.
    Kullanım: Özellik seçimi, seyrek ağlar.
""")

# L1 ve L2'nin ağırlıklara etkisini simüle et
def l2_update(w: np.ndarray, grad: np.ndarray,
               lr: float, lam: float) -> np.ndarray:
    """L2: w ← w(1−η·λ) − η·grad"""
    return w * (1 - lr * lam) - lr * grad

def l1_update(w: np.ndarray, grad: np.ndarray,
               lr: float, lam: float) -> np.ndarray:
    """L1: w ← w − η·λ·sign(w) − η·grad"""
    return w - lr * lam * np.sign(w) - lr * grad

# Basit 1D örnek
w_init = np.array([3.0, -2.0, 0.5, -0.1, 2.5])
grad_zero = np.zeros_like(w_init)  # sadece regularization etkisi
lr, lam = 0.1, 0.3

w_l2 = w_init.copy()
w_l1 = w_init.copy()
for _ in range(20):
    w_l2 = l2_update(w_l2, grad_zero, lr, lam)
    w_l1 = l1_update(w_l1, grad_zero, lr, lam)

print("  20 adım sonra ağırlık değerleri (grad=0, sadece regularization):")
print(f"  {'Başlangıç':<12} {'L2 sonrası':>12} {'L1 sonrası':>12} {'Fark'}")
print(f"  {'-'*55}")
for i, (w0, wl2, wl1) in enumerate(zip(w_init, w_l2, w_l1)):
    note = "← Sıfıra yakın!" if abs(wl1) < 0.05 else ""
    print(f"  {w0:>12.4f} {wl2:>12.4f} {wl1:>12.4f}  {note}")

print(f"\n  L1 sıfır ağırlık sayısı: {np.sum(np.abs(w_l1) < 0.05)}/5")
print(f"  L2 sıfır ağırlık sayısı: {np.sum(np.abs(w_l2) < 0.05)}/5")


# ============================================================
# SECTION 3: DROPOUT
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: Dropout")
print("=" * 65)

print("""
  Dropout (Srivastava et al., 2014):
  Eğitim sırasında her nöronu p olasılıkla rastgele devre dışı bırak.

  EĞİTİM MODU:
    mask = Bernoulli(1-p)  → 0 veya 1 değerler
    a_drop = a * mask / (1-p)   ← "inverted dropout" ölçekleme!

  ÇIKARIM MODU (inference):
    a_drop = a   ← mask YOK, ölçekleme YOK
    (inverted dropout sayesinde beklenti eşit kalır)

  NEDEN İŞE YARAR?
    - Her iterasyonda farklı "alt ağ" eğitilir
    - Nöronlar birbirine co-adapt edemez
    - Ensemble learning etkisi: birçok model ortalaması gibi
    - p=0.5 gizli katman, p=0.1-0.3 giriş katmanı için tipik
""")

class Dropout:
    """
    Inverted Dropout implementasyonu.
    Eğitim: mask uygula ve (1-p) ile böl (ölçekle).
    Inference: hiçbir şey yapma.
    """

    def __init__(self, p: float = 0.5):
        """p: her nöronun DEVRE DIŞI bırakılma olasılığı."""
        assert 0 <= p < 1, "p [0,1) aralığında olmalı"
        self.p    = p
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x  # Inference: değişiklik yok

        # Bernoulli mask: (1-p) olasılıkla 1, p olasılıkla 0
        self.mask = (np.random.rand(*x.shape) > self.p).astype(float)

        # Inverted scaling: beklentiyi korumak için (1-p)'ye böl
        return x * self.mask / (1 - self.p)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Dropout'un türevi: mask uygula ve (1-p)'ye böl."""
        return dout * self.mask / (1 - self.p)

# Demo
np.random.seed(0)
x_demo = np.ones((1, 10))
drop = Dropout(p=0.5)

drop.training = True
results_train = [drop.forward(x_demo) for _ in range(5)]
print("  Eğitim modu — 5 farklı forward pass (p=0.5):")
for i, r in enumerate(results_train):
    active = np.sum(r > 0)
    print(f"  Pass {i+1}: aktif nöron={active}/10  ortalama={r.mean():.3f}")

drop.training = False
result_inf = drop.forward(x_demo)
print(f"\n  Inference modu: aktif nöron={np.sum(result_inf>0)}/10  ortalama={result_inf.mean():.3f}")
print(f"  ✓ Inference ortalaması ≈ eğitim ortalaması (inverted dropout sayesinde)")


# ============================================================
# SECTION 4: BATCH NORMALIZATION
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: Batch Normalization")
print("=" * 65)

print("""
  Batch Normalization (Ioffe & Szegedy, 2015):
  Her mini-batch içinde aktivasyonları normalize et.

  ALGORİTMA (eğitim):
    1. μ_B  = (1/m) · Σ xᵢ              (batch ortalaması)
    2. σ²_B = (1/m) · Σ (xᵢ−μ_B)²      (batch varyansı)
    3. x̂ᵢ  = (xᵢ−μ_B) / √(σ²_B+ε)     (normalize et)
    4. yᵢ  = γ·x̂ᵢ + β                  (ölçek ve kaydır — öğrenilir)

  NEDEN İŞE YARAR?
    - Internal covariate shift azalır (katman girişleri kararlı kalır)
    - Daha büyük öğrenme hızı kullanılabilir
    - Regularization etkisi (batch'e özgü gürültü)
    - Başlatmaya (weight initialization) daha az hassas

  ÇIKARIM:
    - Eğitim sırasında μ ve σ² hareketli ortalamaları tutulur
    - Inference'da batch istatistikleri değil bu sabit değerler kullanılır
""")

class BatchNorm1D:
    """
    1D Batch Normalization — tam implementasyon.
    Eğitim: batch istatistikleri kullan.
    Inference: hareketli ortalama kullan.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        self.eps      = eps
        self.momentum = momentum
        self.training = True

        # Öğrenilen parametreler
        self.gamma = np.ones(num_features)   # Ölçek
        self.beta  = np.zeros(num_features)  # Kaydırma

        # Hareketli ortalamalar (inference için)
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)

        # Backprop cache
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch_size, num_features)
        """
        if self.training:
            # Batch istatistikleri
            mu    = x.mean(axis=0)
            var   = x.var(axis=0)
            x_hat = (x - mu) / np.sqrt(var + self.eps)

            # Hareketli ortalamaları güncelle
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                  + self.momentum * mu)
            self.running_var  = ((1 - self.momentum) * self.running_var
                                  + self.momentum * var)

            # Cache
            self.cache = {'x': x, 'mu': mu, 'var': var,
                          'x_hat': x_hat}
        else:
            # Inference: sabit istatistikler
            x_hat = ((x - self.running_mean)
                     / np.sqrt(self.running_var + self.eps))

        return self.gamma * x_hat + self.beta

# Demo
np.random.seed(1)
x_bn = np.random.randn(32, 8) * 5 + 3  # Ortalama=3, std=5

bn = BatchNorm1D(num_features=8)
bn.training = True
out_bn = bn.forward(x_bn)

print("  Batch Normalization Demo:")
print(f"  Giriş — ortalama: {x_bn.mean(axis=0)[:3].round(2)}...")
print(f"  Çıkış — ortalama: {out_bn.mean(axis=0)[:3].round(4)}...")
print(f"  Çıkış — std:      {out_bn.std(axis=0)[:3].round(4)}...")
print(f"  ✓ Normalize edildi: ortalama≈0, std≈1 (γ=1, β=0 ile)")


# ============================================================
# SECTION 5: LAYER NORMALIZATION
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: Layer Normalization")
print("=" * 65)

print("""
  Layer Normalization (Ba et al., 2016):
  Batch boyutuna DEĞİL, her örneğin kendi özelliklerine göre normalize et.

  FARK:
    Batch Norm:  her özellik için batch boyunca normalize
    Layer Norm:  her örnek için kendi özellikleri boyunca normalize

  FORMÜL:
    μ = mean(x)   (her örnek için kendi ortalaması)
    σ = std(x)    (her örnek için kendi std'si)
    y = γ·(x−μ)/√(σ²+ε) + β

  NEDEN TRANSFORMER'DA LAYER NORM?
    - Batch Norm, batch boyutuna bağımlı → sıralı (sequential) veri için sorunlu
    - Layer Norm, batch boyutundan bağımsız → NLP, Transformer için ideal
    - BERT, GPT, ViT'ın tümü Layer Norm kullanır
""")

class LayerNorm:
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.eps   = eps
        self.gamma = np.ones(normalized_shape)
        self.beta  = np.zeros(normalized_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, features) — her örnek kendi boyunca normalize edilir."""
        mu    = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# Karşılaştırma
x_cmp = np.array([[1., 2., 3., 4.],
                    [10., 20., 30., 40.]])

ln  = LayerNorm(4)
out_ln = ln.forward(x_cmp)
bn2    = BatchNorm1D(4)
out_bn2 = bn2.forward(x_cmp)

print("  Karşılaştırma (x = [[1,2,3,4], [10,20,30,40]]):")
print(f"  Layer Norm çıkış:\n  {out_ln.round(3)}")
print(f"  Batch Norm çıkış:\n  {out_bn2.round(3)}")
print("  Layer Norm: her satır kendi içinde normalize")
print("  Batch Norm: her sütun kendi içinde normalize")


# ============================================================
# SECTION 6: EARLY STOPPING
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: Early Stopping")
print("=" * 65)

class EarlyStopping:
    """
    Early Stopping: validation loss iyileşmezse eğitimi durdur.
    patience: kaç epoch boyunca iyileşme olmazsa dur
    delta: "iyileşme" sayılması için minimum azalma
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = np.inf
        self.counter    = 0
        self.best_epoch = 0
        self.stop       = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.delta:
            # İyileşme var
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
        else:
            # İyileşme yok
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

# Simülasyon: aşırı uzun eğitim
np.random.seed(5)
train_losses = []
val_losses   = []
es = EarlyStopping(patience=8, delta=1e-3)
stopped_epoch = None

for epoch in range(100):
    t_loss = 1.0 / (epoch + 1) + 0.01 * np.random.rand()
    v_loss = (1.0 / (epoch + 1) + 0.05 * np.random.rand()
              + max(0, (epoch - 30) * 0.01))  # 30'dan sonra artmaya başlar

    train_losses.append(t_loss)
    val_losses.append(v_loss)

    if es(v_loss, epoch) and stopped_epoch is None:
        stopped_epoch = epoch
        print(f"  Early Stopping tetiklendi!")
        print(f"  Epoch {epoch}: Val loss = {v_loss:.4f}")
        print(f"  En iyi epoch: {es.best_epoch}, en iyi val loss: {es.best_loss:.4f}")

if stopped_epoch is None:
    print(f"  Early stopping tetiklenmedi (100 epoch tamamlandı)")


# ============================================================
# SECTION 7: DATA AUGMENTATION
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 7: Data Augmentation")
print("=" * 65)

def random_flip(image: np.ndarray) -> np.ndarray:
    """Yatay çevirme."""
    return image[:, ::-1] if np.random.rand() > 0.5 else image

def random_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    """Rastgele kırpma."""
    h, w = image.shape[:2]
    y = np.random.randint(0, h - crop_size)
    x = np.random.randint(0, w - crop_size)
    return image[y:y+crop_size, x:x+crop_size]

def random_noise(image: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Gaussian gürültü ekleme."""
    return np.clip(image + np.random.randn(*image.shape) * std, 0, 1)

def cutout(image: np.ndarray, size: int = 8) -> np.ndarray:
    """CutOut: rastgele bölge siyah yap (Devries & Taylor, 2017)."""
    h, w  = image.shape[:2]
    y     = np.random.randint(0, h)
    x     = np.random.randint(0, w)
    y1, y2 = max(0, y-size//2), min(h, y+size//2)
    x1, x2 = max(0, x-size//2), min(w, x+size//2)
    aug = image.copy()
    aug[y1:y2, x1:x2] = 0
    return aug

print("""
  Yaygın Data Augmentation Teknikleri:

  Görüntü için:
    • Flip (çevirme): Yatay/dikey
    • Crop (kırpma): Rastgele bölge al
    • Rotation: ±15-30 derece döndürme
    • Color Jitter: parlaklık, kontrast, doygunluk
    • Cutout/Random Erasing: rastgele bölge siyahla
    • Mixup: iki görüntüyü karıştır
    • CutMix: bir görüntünün parçasını diğerine yapıştır

  Metin için:
    • Synonym replacement: kelimeleri eş anlamlıyla değiştir
    • Back translation: farklı dile çevir ve geri çevir
    • Random deletion/swap: rastgele kelime sil/yer değiştir
    • EDA (Easy Data Augmentation)
""")

# Test görüntüsü oluştur
test_img = np.random.rand(32, 32)
print(f"  Orijinal görüntü: {test_img.shape}, ortalama={test_img.mean():.3f}")

flip_img  = random_flip(test_img)
noise_img = random_noise(test_img, 0.1)
cut_img   = cutout(test_img, 8)

print(f"  Flip sonrası  : {flip_img.shape}, ortalama={flip_img.mean():.3f}")
print(f"  Noise sonrası : {noise_img.shape}, ortalama={noise_img.mean():.3f}")
print(f"  Cutout sonrası: {cut_img.shape}, sıfır piksel={np.sum(cut_img==0)}")


# ============================================================
# SECTION 8: TENSORFLOW / KERAS
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 8: TensorFlow / Keras Regularization")
print("=" * 65)

try:
    import tensorflow as tf
    print(f"\n  TensorFlow {tf.__version__}")
    print("""
  # L2 Regularization:
  from tensorflow.keras import regularizers
  model.add(Dense(64, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)))

  # Dropout:
  model.add(tf.keras.layers.Dropout(rate=0.5))

  # Batch Normalization:
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  # NOT: BN genellikle aktivasyondan ÖNCE veya SONRA eklenir

  # Early Stopping callback:
  early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', patience=10, restore_best_weights=True)
  model.fit(X, y, callbacks=[early_stop], validation_split=0.2)

  # Data Augmentation (ImageDataGenerator):
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      horizontal_flip=True, rotation_range=15,
      zoom_range=0.1, width_shift_range=0.1)
    """)
except ImportError:
    print("  TensorFlow yüklü değil — pip install tensorflow")


# ============================================================
# SECTION 9: PYTORCH
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 9: PyTorch Regularization")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    print(f"\n  PyTorch {torch.__version__}")
    print("""
  # L2 Regularization (weight_decay):
  optimizer = torch.optim.Adam(
      model.parameters(), lr=0.001, weight_decay=1e-4)

  # L1 Regularization (manuel):
  l1_lambda = 1e-4
  l1_norm = sum(p.abs().sum() for p in model.parameters())
  loss = criterion(output, target) + l1_lambda * l1_norm

  # Dropout:
  class Model(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(128, 64)
          self.drop = nn.Dropout(p=0.5)
          self.fc2 = nn.Linear(64, 10)
      def forward(self, x):
          x = F.relu(self.fc1(x))
          x = self.drop(x)    # eğitimde aktif, eval'de pasif
          return self.fc2(x)

  model.train()  # dropout aktif
  model.eval()   # dropout pasif (inference)

  # Batch Normalization:
  nn.BatchNorm1d(num_features=64)
  nn.BatchNorm2d(num_channels=64)   # CNN için

  # Layer Normalization (Transformer):
  nn.LayerNorm(normalized_shape=512)

  # Data Augmentation (torchvision.transforms):
  import torchvision.transforms as T
  transform = T.Compose([
      T.RandomHorizontalFlip(p=0.5),
      T.RandomRotation(degrees=15),
      T.ColorJitter(brightness=0.2, contrast=0.2),
      T.ToTensor(),
      T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
  ])
    """)
except ImportError:
    print("  PyTorch yüklü değil — pip install torch")


# ============================================================
# SECTION 10: GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 10: Görselleştirme")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Modül 05: Regularization Analizi",
             fontsize=14, fontweight='bold')

# ── Panel 1: Overfitting Görselleştirme ───────────────────────────────────────
ax1 = axes[0, 0]
ax1.scatter(X_data, y_data, color='black', s=25, zorder=5, label='Veri')
ax1.plot(X_test, np.sin(X_test), 'g-',  lw=2, label='Gerçek f(x)=sin(x)')
ax1.plot(X_test, p1(X_test),  'b--', lw=2, label='Derece 1 (Underfitting)')
ax1.plot(X_test, p4(X_test),  'orange', lw=2, label='Derece 4 (İyi fit)')
ax1.plot(X_test, np.clip(p20(X_test), -3, 3), 'r-', lw=2, label='Derece 20 (Overfitting)')
ax1.set_title("Underfitting / İyi Fit / Overfitting", fontweight='bold')
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.legend(fontsize=8); ax1.set_ylim(-3, 3); ax1.grid(True, alpha=0.3)

# ── Panel 2: L1 vs L2 Ağırlık Etkisi ─────────────────────────────────────────
ax2 = axes[0, 1]
w_test = np.linspace(-3, 3, 200)
ax2.plot(w_test, 0.5 * w_test**2, color='#1565C0', lw=2.5, label='L2: ½w²')
ax2.plot(w_test, np.abs(w_test),  color='#E65100', lw=2.5, label='L1: |w|')
ax2.axvline(0, color='gray', lw=0.8, alpha=0.5)
ax2.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax2.set_title("L1 vs L2 Ceza Fonksiyonu", fontweight='bold')
ax2.set_xlabel("w (ağırlık)"); ax2.set_ylabel("Ceza")
ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.text(1.5, 2.5, 'L1: keskin köşe\n→ Seyreklik', fontsize=9,
         color='#E65100', ha='center')
ax2.text(-2, 2.5, 'L2: pürüzsüz\n→ Küçültme', fontsize=9,
         color='#1565C0', ha='center')

# ── Panel 3: Dropout Etkisi ───────────────────────────────────────────────────
ax3 = axes[1, 0]
epochs_plot = np.arange(1, 61)
np.random.seed(7)
train_no_reg  = 0.9 * np.exp(-0.06 * epochs_plot) + 0.02 * np.random.rand(60)
val_no_reg    = 0.9 * np.exp(-0.03 * epochs_plot) + 0.05 * np.random.rand(60) + epochs_plot * 0.003
train_dropout = 0.9 * np.exp(-0.05 * epochs_plot) + 0.02 * np.random.rand(60)
val_dropout   = 0.9 * np.exp(-0.05 * epochs_plot) + 0.02 * np.random.rand(60) + 0.05

ax3.plot(epochs_plot, train_no_reg,  'b--', lw=1.5, alpha=0.7, label='Train (dropout yok)')
ax3.plot(epochs_plot, val_no_reg,    'b-',  lw=2,              label='Val (dropout yok) ← Overfitting')
ax3.plot(epochs_plot, train_dropout, 'r--', lw=1.5, alpha=0.7, label='Train (dropout=0.5)')
ax3.plot(epochs_plot, val_dropout,   'r-',  lw=2,              label='Val (dropout=0.5)')
ax3.fill_between(epochs_plot, val_no_reg, val_dropout,
                  where=(val_no_reg > val_dropout), alpha=0.1, color='green',
                  label='Dropout kazancı')
ax3.set_title("Dropout Etkisi — Overfitting Azaltma", fontweight='bold')
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# ── Panel 4: Early Stopping ───────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.plot(train_losses, 'b-', lw=2, label='Eğitim Loss')
ax4.plot(val_losses,   'r-', lw=2, label='Validation Loss')
if stopped_epoch:
    ax4.axvline(stopped_epoch, color='green', lw=2, ls='--',
                label=f'Early Stop (epoch {stopped_epoch})')
    ax4.axvline(es.best_epoch, color='orange', lw=2, ls=':',
                label=f'En iyi model (epoch {es.best_epoch})')
ax4.set_title("Early Stopping", fontweight='bold')
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    '/home/claude/deep_learning_path/05-Derin_Aglar_Regularization/modul05_analiz.png',
    dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")


# ============================================================
# SECTION 11: ÖZET
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 11: Modül 05 Özeti")
print("=" * 65)
print("""
  ✓ Overfitting: Eğitim↓ Test↑ → Regularization uygula
  ✓ L2 (Weight Decay): Ağırlıkları küçültür — en yaygın tercih
  ✓ L1 (Lasso): Bazı ağırlıkları sıfıra iter — özellik seçimi
  ✓ Dropout: Rastgele nöron kapat → ensemble etkisi
  ✓ Batch Norm: Aktivasyonları normalize → hızlı ve kararlı eğitim
  ✓ Layer Norm: Transformer standardı — batch boyutundan bağımsız
  ✓ Early Stopping: Val loss artarsa dur → en iyi modeli sakla
  ✓ Data Augmentation: Veriyi artır → genelleme kapasitesi yüksel

  HIZLI REHBER:
    Overfitting varsa → Dropout + L2 + Early Stopping
    Derin CNN        → Batch Norm + Dropout
    Transformer      → Layer Norm + Dropout
    Az veri          → Data Augmentation + Transfer Learning

  SONRAKI MODÜL: 06 — CNN (Evriimsel Sinir Ağları)
""")
print("=" * 65)
print("  Modül 05 tamamlandı!")
print("=" * 65)
