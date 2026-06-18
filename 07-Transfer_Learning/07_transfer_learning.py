"""
================================================================================
MODÜL 07: TRANSFER LEARNING
================================================================================

KAPSANAN KONULAR:
  - Transfer Learning neden işe yarar? (özellik hiyerarşisi)
  - Feature extraction vs Fine-tuning vs Full training
  - Freeze / Unfreeze katman stratejisi
  - Pretrained VGG16 / ResNet50 kullanımı
  - Domain adaptation kavramı
  - Fine-tuning için öğrenme hızı stratejisi
  - NumPy ile kavramsal implementasyon
  - TensorFlow ve PyTorch implementasyonları

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

YAZAR: Deep Learning Path — Modül 07
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 65)
print("  MODÜL 07: TRANSFER LEARNING")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: NEDEN TRANSFER LEARNING?
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Transfer Learning Neden İşe Yarar?")
print("=" * 65)

print("""
  Derin CNN'ler öğrendiği özellik hiyerarşisi:

  Katman 1–3   : Düşük seviye özellikler
                 Kenarlar, köşeler, renkler, dokular
                 → Tüm görüntü veri setlerinde AYNI

  Katman 4–8   : Orta seviye özellikler
                 Parçalar, şekiller, desenler
                 → Çoğu veri setine transfer edilebilir

  Katman 9–12  : Yüksek seviye özellikler
                 Yüzler, nesneler, anlam
                 → Kaynak veri setine özgü

  SON FC KATMAN : Sınıflandırıcı
                 → Her zaman YENİDEN eğitilmeli

  SONUÇ: ImageNet'te öğrenilen düşük/orta seviye özellikler
         tıbbi görüntü, uydu fotoğrafı, çiçek tanıma...
         çoğu görüntü görevine transfer edilebilir!
""")


# ============================================================
# SECTION 2: 3 STRATEJİ KARŞILAŞTIRMASI
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: 3 Transfer Stratejisi")
print("=" * 65)

print("""
  ┌─────────────────┬──────────────────────────────────────────┐
  │ Strateji        │ Ne Yapılır                               │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Feature Extract.│ Tüm katmanlar dondurulur, sadece yeni    │
  │                 │ FC katman eğitilir. En hızlı.            │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Fine-tuning     │ Alt katmanlar dondurulur, üst katmanlar  │
  │                 │ + yeni FC küçük lr ile eğitilir.         │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Full Training   │ Hiçbir şey dondurulmaz, hepsi sıfırdan   │
  │                 │ eğitilir. Çok veri gerektirir.           │
  └─────────────────┴──────────────────────────────────────────┘

  HANGİSİNİ SEÇMEK?

  Az veri + Benzer domain    → Feature Extraction
  Orta veri + Benzer domain  → Fine-tuning (son 2-4 blok)
  Çok veri + Farklı domain   → Full Training veya derin fine-tune
""")


# ============================================================
# SECTION 3: KAVRAMSAL NUMPY İMPLEMENTASYONU
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: Kavramsal İmplementasyon — NumPy")
print("=" * 65)

class PretrainedLayer:
    """Önceden eğitilmiş katman simülasyonu."""
    def __init__(self, in_dim: int, out_dim: int, frozen: bool = True):
        self.W      = np.random.randn(in_dim, out_dim) * 0.1
        self.b      = np.zeros(out_dim)
        self.frozen = frozen       # Dondurulmuş mu?
        self.grad_W = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x @ self.W + self.b)   # ReLU

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.frozen:
            return dout  # Gradyan hesaplanmaz, sadece iletilir
        dx      = dout @ self.W.T
        self.grad_W = self.x.T @ dout
        return dx

    def update(self, lr: float):
        if not self.frozen and self.grad_W is not None:
            self.W -= lr * self.grad_W


class TransferModel:
    """
    Transfer Learning modeli.
    pretrained_layers: dondurulmuş önceden eğitilmiş katmanlar
    new_head: yeni görev için eğitilen sınıflandırıcı
    """

    def __init__(self, strategy: str = 'feature_extract'):
        self.strategy = strategy
        # Pretrained backbone (simüle edilmiş)
        if strategy == 'feature_extract':
            self.backbone = [
                PretrainedLayer(32, 64, frozen=True),
                PretrainedLayer(64, 128, frozen=True),
            ]
        elif strategy == 'fine_tune':
            self.backbone = [
                PretrainedLayer(32, 64, frozen=True),    # Alt katman: dondurulmuş
                PretrainedLayer(64, 128, frozen=False),   # Üst katman: eğitilecek
            ]
        else:  # full
            self.backbone = [
                PretrainedLayer(32, 64, frozen=False),
                PretrainedLayer(64, 128, frozen=False),
            ]
        # Yeni sınıflandırıcı (her zaman eğitilir)
        self.head_W  = np.random.randn(128, 5) * 0.1   # 5 yeni sınıf
        self.head_b  = np.zeros(5)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.backbone:
            x = layer.forward(x)
        self.feat = x
        logits    = x @ self.head_W + self.head_b
        return self.softmax(logits)

    def loss_and_grad(self, x: np.ndarray, y: np.ndarray,
                       lr: float = 0.01) -> float:
        probs     = self.forward(x)
        n         = y.shape[0]
        # Cross-entropy loss
        eps       = 1e-15
        loss      = -np.mean(np.sum(y * np.log(probs + eps), axis=1))

        # Backprop — sadece head
        dlogits   = (probs - y) / n
        dW_head   = self.feat.T @ dlogits
        self.head_W -= lr * dW_head
        self.head_b -= lr * dlogits.sum(axis=0)

        # Backbone (frozen olmayan katmanlara)
        dout = dlogits @ self.head_W.T
        for layer in reversed(self.backbone):
            dout = layer.backward(dout)
            layer.update(lr)

        return float(loss)


# Karşılaştırma
np.random.seed(0)
X_tr = np.random.randn(100, 32)
y_tr = np.eye(5)[np.random.randint(0, 5, 100)]

print("\n  3 Strateji Karşılaştırması (100 örnek, 5 sınıf, 50 epoch):")
print(f"  {'Strateji':<20} {'Eğitilen Param':>16} {'Final Loss':>12}")
print(f"  {'-'*52}")

for strategy in ['feature_extract', 'fine_tune', 'full']:
    model = TransferModel(strategy)
    losses = []
    for _ in range(50):
        l = model.loss_and_grad(X_tr, y_tr, lr=0.005)
        losses.append(l)

    # Eğitilen parametre sayısı
    trainable = sum(
        l.W.size for l in model.backbone if not l.frozen
    ) + model.head_W.size
    print(f"  {strategy:<20} {trainable:>16,} {losses[-1]:>12.4f}")


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
  # ── Feature Extraction ─────────────────────────────────
  base_model = tf.keras.applications.ResNet50(
      weights='imagenet',
      include_top=False,           # FC katmanı dahil etme
      input_shape=(224, 224, 3)
  )
  base_model.trainable = False     # TÜM KATMANLAR DONDURULDU

  inputs = tf.keras.Input(shape=(224, 224, 3))
  x = base_model(inputs, training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
  model = tf.keras.Model(inputs, outputs)

  model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(train_ds, epochs=10)   # Sadece head eğitilir

  # ── Fine-tuning ──────────────────────────────────────────
  base_model.trainable = True

  # Sadece son bloku aç
  for layer in base_model.layers[:-30]:
      layer.trainable = False

  # Fine-tuning için çok küçük lr!
  model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy')
  model.fit(train_ds, epochs=5)
    """)
except ImportError:
    print("  TensorFlow yüklü değil.")


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
  import torchvision.models as models

  # ── Feature Extraction ──────────────────────────────────
  model = models.resnet50(pretrained=True)

  # Tüm katmanları dondur
  for param in model.parameters():
      param.requires_grad = False

  # Yeni sınıflandırıcı ekle (otomatik eğitilir)
  num_features = model.fc.in_features
  model.fc = nn.Sequential(
      nn.Linear(num_features, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, num_classes)
  )

  # Sadece fc parametreleri optimizer'a gönder
  optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

  # ── Fine-tuning ──────────────────────────────────────────
  # Son layer4 bloğunu aç
  for param in model.layer4.parameters():
      param.requires_grad = True

  # Diferansiyel lr: backbone için küçük, head için büyük
  optimizer = torch.optim.Adam([
      {'params': model.layer4.parameters(), 'lr': 1e-5},
      {'params': model.fc.parameters(),     'lr': 1e-3},
  ])

  # Eğitim döngüsü
  model.train()
  for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    """)
except ImportError:
    print("  PyTorch yüklü değil.")


# ============================================================
# SECTION 6: GÖRSELLEŞTİRME
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Modül 07: Transfer Learning Analizi",
             fontsize=14, fontweight='bold')

# Panel 1: Özellik hiyerarşisi
ax1 = axes[0]
layers_name = ['Katman 1-3\n(Kenarlar)', 'Katman 4-8\n(Şekiller)',
               'Katman 9-12\n(Nesneler)', 'FC\n(Sınıflandırıcı)']
transferability = [0.95, 0.75, 0.40, 0.05]
colors = ['#2E7D32', '#1565C0', '#E65100', '#B71C1C']
bars = ax1.barh(layers_name, transferability, color=colors, edgecolor='white', lw=1.5)
ax1.set_xlim(0, 1.1)
ax1.set_xlabel('Transfer Edilebilirlik')
ax1.set_title('Özellik Hiyerarşisi ve Transfer', fontweight='bold')
for bar, val in zip(bars, transferability):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.0%}', va='center', fontsize=10, fontweight='bold')
ax1.axvline(0.5, color='gray', ls='--', alpha=0.5, lw=1)
ax1.grid(True, alpha=0.3, axis='x')

# Panel 2: 3 stratejinin kıyaslaması
ax2 = axes[1]
n_layers = 12
strategies = {
    'Feature Extract': ([1]*12, [0]*11 + [1]),
    'Fine-tune (son 4)': ([1]*8 + [0]*4, [0]*8 + [1]*4 + [1]),
    'Full Training': ([0]*12, [1]*12 + [1]),
}
y_pos  = [2.5, 1.5, 0.5]
colors2 = ['#1565C0', '#00695C', '#E65100']

for (name, (frozen, train)), y, c in zip(strategies.items(), y_pos, colors2):
    # Tüm katmanlar + head
    all_layers = frozen + [train[-1]]
    for i, (fr, is_train_) in enumerate(zip(frozen + [0], [is_train_ for is_train_ in ([0]*len(frozen)) + [1]])):
        pass
    for i in range(n_layers + 1):
        is_frozen = (i < n_layers and frozen[i] == 1)
        color_bar = '#BDBDBD' if is_frozen else c
        ax2.barh(y, 0.8, left=i, height=0.6, color=color_bar,
                 edgecolor='white', lw=0.5)

    ax2.text(-0.3, y, name, ha='right', va='center', fontsize=9, fontweight='bold')

ax2.set_xlim(-4, 14)
ax2.set_ylim(0, 3.5)
ax2.set_xlabel('Katman İndeksi (0→12: düşük→yüksek)')
ax2.set_title('Freeze/Unfreeze Stratejileri', fontweight='bold')
ax2.set_yticks([])
ax2.grid(True, alpha=0.3, axis='x')

# Legend
from matplotlib.patches import Patch
legend = [Patch(color='#BDBDBD', label='Dondurulmuş (frozen)'),
          Patch(color='#1565C0', label='Eğitilen (trainable)')]
ax2.legend(handles=legend, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/07-Transfer_Learning/modul07_analiz.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")

print("\n" + "=" * 65)
print("  Modül 07 tamamlandı!")
print("=" * 65)
