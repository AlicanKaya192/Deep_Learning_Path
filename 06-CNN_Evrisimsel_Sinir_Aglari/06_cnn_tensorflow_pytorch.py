"""
================================================================================
MODÜL 06: CNN — EVRİŞİMSEL SİNİR AĞLARI
================================================================================

KAPSANAN KONULAR:
  - Konvolüsyon işlemi — matematiksel tanım, padding, stride, dilation
  - Feature map hesaplama formülü
  - Pooling katmanları (Max, Average, Global)
  - CNN mimarisi: Konv → BN → ReLU → Pool → FC
  - Klasik mimariler: LeNet → AlexNet → VGG → ResNet → EfficientNet
  - ResNet Skip Connection — neden derin ağları kurtardı?
  - Feature map görselleştirme
  - Grad-CAM — model neye bakıyor?
  - NumPy konvolüsyon from scratch
  - TensorFlow ve PyTorch CNN implementasyonu

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

YAZAR: Deep Learning Path — Modül 06
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 65)
print("  MODÜL 06: CNN — EVRİŞİMSEL SİNİR AĞLARI")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: KONVOLÜSYONİŞLEMİ — NUMPY FROM SCRATCH
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Konvolüsyon İşlemi — Sıfırdan NumPy")
print("=" * 65)

print("""
  Konvolüsyon: Filtre (kernel) görüntü üzerinde kaydırılır.
  Her pozisyonda element-wise çarpım + toplam yapılır.

  Çıkış boyutu:
    H_out = floor((H_in + 2P − K) / S) + 1
    W_out = floor((W_in + 2P − K) / S) + 1

  H_in: Giriş yüksekliği   K: Kernel boyutu
  P: Padding               S: Stride
""")

def conv2d(x: np.ndarray, kernel: np.ndarray,
           padding: int = 0, stride: int = 1) -> np.ndarray:
    """
    2D Konvolüsyon — sıfırdan NumPy implementasyonu.
    x     : (H, W) giriş görüntüsü
    kernel: (kH, kW) filtre
    """
    H, W   = x.shape
    kH, kW = kernel.shape

    # Padding uygula
    if padding > 0:
        x = np.pad(x, padding, mode='constant', constant_values=0)

    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    out   = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+kH, j*stride:j*stride+kW]
            out[i, j] = np.sum(patch * kernel)
    return out

def max_pool2d(x: np.ndarray, pool_size: int = 2,
               stride: int = 2) -> np.ndarray:
    """Max Pooling — en büyük değeri al."""
    H, W  = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out   = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch    = x[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i,j] = np.max(patch)
    return out

# ── Demo Görüntü ───────────────────────────────────────────
# 8×8 basit görüntü
image = np.array([
    [0,0,0,0,0,0,0,0],
    [0,1,1,1,1,0,0,0],
    [0,1,0,0,1,0,0,0],
    [0,1,0,0,1,0,0,0],
    [0,1,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
], dtype=float)

# Kenar algılama filtreleri
edge_h = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=float)  # Yatay kenar
edge_v = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=float)  # Dikey kenar
blur   = np.ones((3,3)) / 9                                    # Bulanıklaştırma

fmaps = {
    'Orijinal':          image,
    'Yatay Kenar':       conv2d(image, edge_h, padding=1),
    'Dikey Kenar':       conv2d(image, edge_v, padding=1),
    'Bulanıklaştırma':   conv2d(image, blur,   padding=1),
    'Max Pool (2×2)':    max_pool2d(conv2d(image, edge_v, padding=1)),
}

print("  Konvolüsyon Demo — 8×8 görüntü, çeşitli filtreler:")
for name, fm in fmaps.items():
    print(f"  {name:<22}: {fm.shape}")

# Boyut hesaplama
H_in, W_in = 8, 8
for K, P, S in [(3,0,1), (3,1,1), (5,2,1), (3,0,2)]:
    H_out = (H_in + 2*P - K) // S + 1
    print(f"  H_in={H_in}, K={K}, P={P}, S={S}  →  H_out={H_out}")


# ============================================================
# SECTION 2: KLASİK MİMARİLER — ZAMAN ÇİZELGESİ
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: Klasik CNN Mimarileri")
print("=" * 65)

print("""
  ┌────────────┬──────┬────────────────────────────────────────┐
  │ Model      │ Yıl  │ Temel Yenilik                          │
  ├────────────┼──────┼────────────────────────────────────────┤
  │ LeNet-5    │ 1998 │ İlk başarılı CNN, MNIST                │
  │ AlexNet    │ 2012 │ GPU eğitimi, ReLU, Dropout, ImageNet   │
  │ VGG-16/19  │ 2014 │ 3×3 filtreler, derinlik artışı         │
  │ GoogLeNet  │ 2014 │ Inception modülü, 1×1 konvolüsyon      │
  │ ResNet-50  │ 2015 │ Skip connection, 152 katman mümkün     │
  │ DenseNet   │ 2017 │ Tüm önceki katmanlarla bağlantı        │
  │ MobileNet  │ 2017 │ Depthwise konv, mobil cihazlar         │
  │ EfficientNet│2019 │ Compound scaling, en verimli           │
  └────────────┴──────┴────────────────────────────────────────┘
""")


# ============================================================
# SECTION 3: RESNET SKIP CONNECTION — NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: ResNet Skip Connection")
print("=" * 65)

print("""
  Problem: 50+ katmanlı ağlarda vanishing gradient → degradation
  Çözüm: F(x) + x  (residual learning)

  Normal blok:  x → [Conv→BN→ReLU→Conv→BN] → F(x) → ReLU
  ResNet blok:  x → [Conv→BN→ReLU→Conv→BN] → F(x) + x → ReLU
                                                       ↑
                                               Skip Connection

  Öğrenilecek: F(x) = H(x) - x  (rezidüel = fark)
  Eğer F(x)=0 olursa → H(x)=x (kimlik dönüşümü)
  Bu, derin ağın yüzeysel ağdan kötü olmayacağını garanti eder!
""")

def relu(x): return np.maximum(0, x)

def resnet_block_numpy(x: np.ndarray, W1: np.ndarray,
                        W2: np.ndarray) -> np.ndarray:
    """
    Basit ResNet bloğu — tam bağlantılı versiyonu.
    CNN versiyonunda Conv katmanları olur.
    """
    # İlk dönüşüm
    h = relu(x @ W1)
    # İkinci dönüşüm
    F_x = h @ W2
    # Skip connection: F(x) + x
    out = relu(F_x + x)
    return out

# Demo
np.random.seed(1)
dim = 16
x_res = np.random.randn(1, dim)

# Normal ağırlık başlatma
W1 = np.random.randn(dim, dim) * 0.1
W2 = np.random.randn(dim, dim) * 0.1

# 20 blok ResNet
x_normal = x_res.copy()
x_resnet  = x_res.copy()

for i in range(20):
    # Normal blok
    x_normal = relu(relu(x_normal @ W1) @ W2)
    # ResNet blok
    x_resnet = resnet_block_numpy(x_resnet, W1, W2)

print(f"  20 blok sonrası:")
print(f"  Normal ağ  — sinyal normu: {np.linalg.norm(x_normal):.6f}")
print(f"  ResNet     — sinyal normu: {np.linalg.norm(x_resnet):.6f}")
print(f"  ✓ ResNet sinyali canlı tutuyor!")


# ============================================================
# SECTION 4: GRAD-CAM KAVRAMI
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: Grad-CAM — Model Neye Bakıyor?")
print("=" * 65)

print("""
  Grad-CAM (Gradient-weighted Class Activation Mapping):
  Son konvolüsyon katmanının gradyanlarını kullanarak
  modelin hangi bölgeye odaklandığını görselleştirir.

  ADIMLAR:
  1. İleri geçiş yap → hedef sınıf skoru al
  2. Skor'un son Conv katmanına göre gradyanını hesapla
  3. Her kanal için gradyanların global ortalaması = kanal ağırlığı (αₖ)
  4. Ağırlıklı feature map toplamı → ReLU uygula → ısı haritası
  5. Orijinal görüntü boyutuna ölçekle

  Formül:
    αₖ = (1/Z) · Σᵢⱼ ∂y^c / ∂Aᵢⱼᵏ
    L_Grad-CAM = ReLU(Σₖ αₖ · Aᵏ)
""")

# Basit Grad-CAM simülasyonu
np.random.seed(42)
feature_maps  = np.random.rand(8, 8, 16)   # 8×8, 16 kanal
gradients     = np.random.randn(8, 8, 16)  # Her kanal için gradyan

# Kanal ağırlıkları: her kanalın global ortalama gradyanı
alpha = gradients.mean(axis=(0, 1))         # (16,)

# Ağırlıklı toplam
cam = np.sum(feature_maps * alpha[np.newaxis, np.newaxis, :], axis=2)

# ReLU uygula
cam = np.maximum(cam, 0)

# Normalize
if cam.max() > 0:
    cam = cam / cam.max()

print(f"  Feature maps: {feature_maps.shape}")
print(f"  Kanal ağırlıkları (α): {alpha.shape}, max={alpha.max():.3f}")
print(f"  Grad-CAM haritası: {cam.shape}, max={cam.max():.3f}")
print(f"  En yüksek aktivasyon bölgesi: {np.unravel_index(cam.argmax(), cam.shape)}")


# ============================================================
# SECTION 5: TENSORFLOW / KERAS CNN
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: TensorFlow / Keras CNN")
print("=" * 65)

try:
    import tensorflow as tf
    print(f"\n  TensorFlow {tf.__version__}")
    print("""
  # CIFAR-10 için CNN — Keras Sequential API:

  model = tf.keras.Sequential([
      # Blok 1
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu',
                              input_shape=(32,32,3)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Dropout(0.25),

      # Blok 2
      tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Dropout(0.25),

      # Classifier
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(10, activation='softmax'),
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    """)
except ImportError:
    print("  TensorFlow yüklü değil.")


# ============================================================
# SECTION 6: PYTORCH CNN
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: PyTorch CNN")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    print(f"\n  PyTorch {torch.__version__}")
    print("""
  class CIFAR10_CNN(nn.Module):
      def __init__(self):
          super().__init__()
          self.features = nn.Sequential(
              # Blok 1
              nn.Conv2d(3, 32, 3, padding=1),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.Conv2d(32, 32, 3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(2, 2),
              nn.Dropout2d(0.25),

              # Blok 2
              nn.Conv2d(32, 64, 3, padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 64, 3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(2, 2),
              nn.Dropout2d(0.25),
          )
          self.classifier = nn.Sequential(
              nn.Flatten(),
              nn.Linear(64*8*8, 512),
              nn.ReLU(inplace=True),
              nn.Dropout(0.5),
              nn.Linear(512, 10),
          )
      def forward(self, x):
          return self.classifier(self.features(x))

  # Kullanım
  model = CIFAR10_CNN()
  print(model)
  # Parametre sayısı
  total = sum(p.numel() for p in model.parameters())
  print(f"Toplam parametre: {total:,}")
    """)
except ImportError:
    print("  PyTorch yüklü değil.")


# ============================================================
# SECTION 7: GÖRSELLEŞTİRME
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Modül 06: CNN Analizi", fontsize=14, fontweight='bold')

# Panel 1-4: Feature maps
for ax, (name, fm) in zip(axes.flatten()[:5], fmaps.items()):
    ax.imshow(fm, cmap='gray')
    ax.set_title(name, fontweight='bold', fontsize=10)
    ax.axis('off')

# Panel 5: Grad-CAM simülasyonu
ax6 = axes.flatten()[5]
im  = ax6.imshow(cam, cmap='jet')
ax6.set_title("Grad-CAM Isı Haritası (simülasyon)", fontweight='bold', fontsize=10)
ax6.axis('off')
plt.colorbar(im, ax=ax6, fraction=0.046)

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/06-CNN_Evrisimsel_Sinir_Aglari/modul06_analiz.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")

print("\n" + "=" * 65)
print("  Modül 06 tamamlandı!")
print("=" * 65)
