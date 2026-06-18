"""
================================================================================
MODÜL 01: SİNİR AĞLARI TEMELLERİ — Sıfırdan Tam İmplementasyon
================================================================================

Bu dosya Deep Learning Path'in ilk modülüdür. Yapay sinir ağlarının temelini
oluşturan kavramları sıfırdan, adım adım ve üç farklı yaklaşımla öğretir:

  1. NUMPY (from scratch) — hiçbir DL kütüphanesi kullanmadan
  2. TENSORFLOW / KERAS    — sektörün en yaygın production framework'ü
  3. PYTORCH               — araştırmada en yaygın kullanılan framework

KAPSANAN KONULAR:
  - Biyolojik nöron → yapay nöron dönüşümü
  - Perceptron: ağırlıklar, bias, aktivasyon
  - İleri besleme (forward pass) matematiği
  - Çok katmanlı perceptron (MLP) mimarisi
  - XOR problemi: neden tek katman yetersiz?
  - Eğitim döngüsü (training loop): loss, gradient, güncelleme
  - Aktivasyon fonksiyonları: Sigmoid, ReLU, Tanh
  - Sonuçların görselleştirilmesi

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib tensorflow torch torchvision scikit-learn

ÇALIŞTIRMA:
  python 01_sinir_aglari_numpy_from_scratch.py

BEKLENEN ÇIKTI:
  - Her section için eğitim kayıpları (loss) yazdırılır
  - Matplotlib ile 4 grafik görüntülenir:
      1. XOR karar sınırı (NumPy MLP)
      2. Eğitim loss eğrisi
      3. Framework karşılaştırma barları
      4. Ağırlık dağılımı histogramı

YAZAR: Deep Learning Path — Modül 01
================================================================================
"""

# ── Gerekli kütüphaneler ──────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# Rastgelelik kontrolü — her çalıştırmada aynı sonuçlar
np.random.seed(42)

print("=" * 65)
print("  MODÜL 01: SİNİR AĞLARI TEMELLERİ")
print("  Deep Learning Path — Sıfırdan İmplementasyon")
print("=" * 65)


# ============================================================
# SECTION 0: TEORİK ARKA PLAN — Yapay Nöron Modeli
# ============================================================
print("\n[SECTION 0] Teorik Arka Plan")
print("-" * 40)

# Bir yapay nöronun çalışması şu formülle özetlenir:
#
#   z = w1*x1 + w2*x2 + ... + wn*xn + b     (ağırlıklı toplam)
#   a = f(z)                                  (aktivasyon fonksiyonu)
#
# Burada:
#   x_i : giriş değerleri (özellikler)
#   w_i : ağırlıklar (öğrenilen parametreler)
#   b   : bias (eşik değeri)
#   f   : aktivasyon fonksiyonu (doğrusal olmayan dönüşüm)
#   a   : nöron çıktısı (aktivasyon)

def artificial_neuron_demo():
    """Tek bir yapay nöronun matematiksel işleyişini gösterir."""
    print("\n  >>> Tek Nöron Örneği:")
    
    # Giriş vektörü: 3 özellik
    x = np.array([0.5, -1.2, 0.8])
    
    # Ağırlıklar (başlangıçta rastgele)
    w = np.array([0.4, -0.3, 0.7])
    
    # Bias
    b = 0.1
    
    # Adım 1: Ağırlıklı toplam (dot product)
    z = np.dot(w, x) + b
    print(f"  Giriş x       : {x}")
    print(f"  Ağırlıklar w  : {w}")
    print(f"  Bias b        : {b}")
    print(f"  z = w·x + b   : {z:.4f}")
    
    # Adım 2: Aktivasyon (Sigmoid kullanıyoruz)
    activation = 1 / (1 + np.exp(-z))
    print(f"  a = σ(z)      : {activation:.4f}")
    print(f"  Yorum: Bu nöron {activation:.2%} olasılıkla 'ateşleniyor'")
    
    return activation

artificial_neuron_demo()


# ============================================================
# SECTION 1: FROM SCRATCH WITH NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: NUMPY — Sıfırdan MLP (XOR Problemi)")
print("=" * 65)

# ── 1.1 Aktivasyon Fonksiyonları ──────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid aktivasyon fonksiyonu.
    Formül: σ(z) = 1 / (1 + e^(-z))
    Çıktı aralığı: (0, 1) — ikili sınıflandırma için idealdir.
    Sorun: z çok büyük/küçük olunca gradyan ≈ 0 (vanishing gradient).
    """
    # Sayısal taşma önlemi: z değerini kırpıyoruz
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """
    Sigmoid'in türevi.
    Formül: σ'(z) = σ(z) * (1 - σ(z)) = a * (1 - a)
    NOT: Girdi olarak sigmoid'in çıktısı 'a' verilir, z değil.
    Bu sayede yeniden hesaplama yapmadan türevi buluruz.
    """
    return a * (1.0 - a)

def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) aktivasyon fonksiyonu.
    Formül: ReLU(z) = max(0, z)
    Avantaj: Basit, hızlı, vanishing gradient'i azaltır.
    Sorun: z < 0 için nöron ölür (dead neuron problemi).
    """
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    ReLU'nun türevi: z > 0 ise 1, z <= 0 ise 0.
    """
    return (z > 0).astype(float)

def tanh(z: np.ndarray) -> np.ndarray:
    """
    Hiperbolik tanjant aktivasyon fonksiyonu.
    Formül: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    Çıktı aralığı: (-1, 1) — sıfır merkezlidir, sigmoid'den daha iyi.
    """
    return np.tanh(z)

def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """tanh türevi: 1 - tanh²(z) = 1 - a²"""
    return 1.0 - a ** 2


# ── 1.2 XOR Veri Seti ─────────────────────────────────────────────────────────
# XOR problemi tek bir nöronla çözülemez (doğrusal ayrılabilir değil).
# Bunu çözmek için en az 1 gizli katman gerekir.
#
# XOR doğruluk tablosu:
#   x1  x2  | y (beklenen çıktı)
#   ---------|------------------
#   0   0   |  0
#   0   1   |  1
#   1   0   |  1
#   1   1   |  0

X_xor = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=float)

y_xor = np.array([[0],
                   [1],
                   [1],
                   [0]], dtype=float)

print(f"\n  XOR Veri Seti:")
print(f"  {'x1':>4} {'x2':>4} | {'y (beklenen)':>12}")
print(f"  {'-'*25}")
for i in range(len(X_xor)):
    print(f"  {int(X_xor[i,0]):>4} {int(X_xor[i,1]):>4} | {int(y_xor[i,0]):>12}")


# ── 1.3 MLP Sınıfı — Tam NumPy İmplementasyonu ───────────────────────────────

class NumpyMLP:
    """
    Çok Katmanlı Perceptron (Multi-Layer Perceptron) — Sıfırdan NumPy ile.
    
    Mimari: Giriş → Gizli Katman → Çıkış
    
    Parametreler:
        input_size   : Giriş özellik sayısı
        hidden_size  : Gizli katman nöron sayısı
        output_size  : Çıkış nöron sayısı
        learning_rate: Öğrenme hızı (adım büyüklüğü)
    
    Ağırlık Başlatma (Weight Initialization):
        Xavier/Glorot başlatma kullanıyoruz. Neden?
        - Çok küçük başlangıç → sinyal yok olur (vanishing)
        - Çok büyük başlangıç → sinyal patlar (exploding)
        - Xavier: sqrt(2 / (fan_in + fan_out)) — dengeyi sağlar
    """
    
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, learning_rate: float = 0.1):
        self.lr = learning_rate
        
        # Xavier başlatma
        # Katman 1: Giriş → Gizli
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        # Katman 2: Gizli → Çıkış
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))
        
        # Eğitim geçmişi
        self.loss_history: list = []
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        İleri besleme (forward pass).
        
        Hesaplama adımları:
          1. z1 = X @ W1 + b1     (birinci katman ağırlıklı toplam)
          2. a1 = sigmoid(z1)     (birinci katman aktivasyon)
          3. z2 = a1 @ W2 + b2   (ikinci katman ağırlıklı toplam)
          4. a2 = sigmoid(z2)     (çıkış aktivasyonu → tahmin)
        
        Ara değerleri saklıyoruz çünkü backprop'ta lazım olacak.
        """
        # Katman 1
        self.z1 = X @ self.W1 + self.b1        # (n_samples, hidden_size)
        self.a1 = sigmoid(self.z1)              # (n_samples, hidden_size)
        
        # Katman 2 (Çıkış)
        self.z2 = self.a1 @ self.W2 + self.b2  # (n_samples, output_size)
        self.a2 = sigmoid(self.z2)              # (n_samples, output_size)
        
        return self.a2
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy kayıp fonksiyonu.
        
        Formül: L = -1/n * Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        Neden bu kayıp?
        - MSE (Ortalama Kare Hata) sınıflandırmada yavaş öğrenir.
        - BCE, olasılık tahminlerine ceza verirken log ölçeği kullanır.
        - Tahmin 0 iken gerçek 1 ise ceza → sonsuz (model çok emin yanlış tahmin yaptı).
        """
        n = y_true.shape[0]
        # Sıfıra bölme önlemi için küçük epsilon
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred_clipped) +
                        (1 - y_true) * np.log(1 - y_pred_clipped))
        return float(loss)
    
    def backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Geri yayılım (backpropagation) — Zincir kuralı ile gradyanlar.
        
        Zincir kuralı sırası (çıkıştan girişe doğru):
        
          dL/dW2 = a1.T @ dL_da2          (2. katman ağırlık gradyanı)
          dL/db2 = Σ dL_da2               (2. katman bias gradyanı)
          dL/dW1 = X.T @ delta1            (1. katman ağırlık gradyanı)
          dL/db1 = Σ delta1                (1. katman bias gradyanı)
        
        Burada delta1, zincir kuralının 1. katmana kadar taşınan gradyanıdır.
        """
        n = X.shape[0]
        
        # ── Çıkış katmanı gradyanı ──
        # dL/da2: BCE + Sigmoid kombinasyonu türevi = (ŷ - y)
        # Bu güzel bir matematiksel basitleşmedir!
        dL_da2 = (self.a2 - y_true) / n        # (n, output_size)
        
        # Çıkış katmanı ağırlık ve bias gradyanları
        dW2 = self.a1.T @ dL_da2               # (hidden_size, output_size)
        db2 = np.sum(dL_da2, axis=0, keepdims=True)
        
        # ── Gizli katman gradyanı (zincir kuralı devam) ──
        # Hatanın gizli katmana geri yayılımı
        dL_da1 = dL_da2 @ self.W2.T            # (n, hidden_size)
        
        # Sigmoid türevi: a1 * (1 - a1)
        delta1 = dL_da1 * sigmoid_derivative(self.a1)  # element-wise çarpım
        
        # Gizli katman ağırlık ve bias gradyanları
        dW1 = X.T @ delta1                     # (input_size, hidden_size)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # ── Ağırlık güncelleme (Gradient Descent) ──
        # W_new = W_old - lr * dL/dW
        # Negatif yönde adım atıyoruz çünkü kaybı azaltmak istiyoruz.
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 10000, verbose_every: int = 2000) -> None:
        """
        Tam eğitim döngüsü.
        Her epoch: forward → loss → backward → güncelleme.
        """
        print(f"\n  MLP Eğitimi Başlıyor...")
        print(f"  Mimari: {X.shape[1]} → {self.W1.shape[1]} → {self.W2.shape[1]}")
        print(f"  Epochs: {epochs} | Öğrenme hızı: {self.lr}")
        print(f"  {'-'*40}")
        
        for epoch in range(1, epochs + 1):
            # 1. İleri geçiş
            predictions = self.forward(X)
            
            # 2. Kayıp hesabı
            loss = self.compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            # 3. Geri yayılım
            self.backward(X, y)
            
            # 4. İlerleme raporu
            if epoch % verbose_every == 0 or epoch == 1:
                accuracy = self.evaluate(X, y)
                print(f"  Epoch {epoch:>6}/{epochs} | "
                      f"Loss: {loss:.6f} | Accuracy: {accuracy:.1%}")
        
        print(f"  {'-'*40}")
        print(f"  Eğitim tamamlandı! Son Loss: {self.loss_history[-1]:.6f}")
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Tahmin — sigmoid çıktısı threshold üstündeyse 1, değilse 0."""
        probabilities = self.forward(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Doğruluk oranını hesaplar."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))


# ── 1.4 Modeli Eğit ───────────────────────────────────────────────────────────
numpy_mlp = NumpyMLP(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=0.5
)
numpy_mlp.train(X_xor, y_xor, epochs=10000, verbose_every=2000)

# Sonuçları göster
print("\n  XOR Tahmin Sonuçları (NumPy MLP):")
print(f"  {'x1':>4} {'x2':>4} | {'Beklenen':>10} | {'Tahmin':>8} | {'Olasılık':>10}")
print(f"  {'-'*45}")
probs = numpy_mlp.forward(X_xor)
preds = numpy_mlp.predict(X_xor)
for i in range(len(X_xor)):
    status = "✓" if preds[i, 0] == int(y_xor[i, 0]) else "✗"
    print(f"  {int(X_xor[i,0]):>4} {int(X_xor[i,1]):>4} | "
          f"{int(y_xor[i,0]):>10} | {preds[i,0]:>8} | "
          f"{probs[i,0]:>9.4f} {status}")

final_acc = numpy_mlp.evaluate(X_xor, y_xor)
print(f"\n  Final Accuracy: {final_acc:.1%}")


# ============================================================
# SECTION 2: TENSORFLOW / KERAS İMPLEMENTASYONU
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: TensorFlow / Keras — XOR Problemi")
print("=" * 65)

try:
    import tensorflow as tf
    from tensorflow import keras

    # Rastgelelik kontrolü
    tf.random.set_seed(42)

    print(f"\n  TensorFlow versiyonu: {tf.__version__}")

    # Keras ile model tanımı — Sequential API
    # Her satır bir katmanı temsil eder; katmanlar otomatik bağlanır.
    tf_model = keras.Sequential([
        # Gizli katman: 4 nöron, sigmoid aktivasyon
        # input_shape parametresi ilk katmanda belirtilmeli
        keras.layers.Dense(
            units=4,
            activation='sigmoid',
            input_shape=(2,),
            name='hidden_layer'
        ),
        # Çıkış katmanı: 1 nöron, ikili sınıflandırma için sigmoid
        keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name='output_layer'
        )
    ])

    # Model derleme — kayıp fonksiyonu ve optimizer seçimi
    tf_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.5),  # NumPy ile aynı lr
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n  Model Mimarisi:")
    tf_model.summary()

    # Eğitim
    print("\n  TF Model Eğitimi...")
    history_tf = tf_model.fit(
        X_xor, y_xor,
        epochs=10000,
        verbose=0,          # Çok fazla çıktı olmaması için 0
        batch_size=4         # Tüm veri seti (XOR sadece 4 örnek)
    )

    # Değerlendirme
    tf_loss, tf_acc = tf_model.evaluate(X_xor, y_xor, verbose=0)
    print(f"\n  TF Model — Loss: {tf_loss:.6f} | Accuracy: {tf_acc:.1%}")

    tf_predictions = (tf_model.predict(X_xor, verbose=0) >= 0.5).astype(int)
    print("\n  TF Tahmin Sonuçları:")
    print(f"  {'x1':>4} {'x2':>4} | {'Beklenen':>10} | {'Tahmin':>8}")
    print(f"  {'-'*35}")
    for i in range(len(X_xor)):
        status = "✓" if tf_predictions[i, 0] == int(y_xor[i, 0]) else "✗"
        print(f"  {int(X_xor[i,0]):>4} {int(X_xor[i,1]):>4} | "
              f"{int(y_xor[i,0]):>10} | {tf_predictions[i,0]:>8} {status}")

    tf_loss_history = history_tf.history['loss']
    tf_available = True

except ImportError:
    print("  [UYARI] TensorFlow yüklü değil. Section 2 atlanıyor.")
    print("  Yüklemek için: pip install tensorflow")
    tf_available = False
    tf_loss_history = []


# ============================================================
# SECTION 3: PYTORCH İMPLEMENTASYONU
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: PyTorch — XOR Problemi")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Rastgelelik kontrolü
    torch.manual_seed(42)

    print(f"\n  PyTorch versiyonu: {torch.__version__}")

    # NumPy dizilerini PyTorch Tensor'a dönüştür
    # float32 PyTorch'un standart veri tipidir
    X_tensor = torch.FloatTensor(X_xor)
    y_tensor = torch.FloatTensor(y_xor)

    # PyTorch'ta model tanımı — nn.Module kalıtımı
    # Bu yaklaşım daha fazla kontrol sağlar ve araştırmada standarttır.
    class PyTorchMLP(nn.Module):
        """
        PyTorch MLP — nn.Module'den türetilmiş.
        
        PyTorch'ta her model nn.Module'den miras alır.
        __init__: Katmanları tanımla
        forward : İleri geçişi tanımla (backward otomatik hesaplanır!)
        """
        
        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            super(PyTorchMLP, self).__init__()
            
            # nn.Linear: tam bağlantılı katman (weights + bias içerir)
            self.hidden = nn.Linear(input_size, hidden_size)
            self.output = nn.Linear(hidden_size, output_size)
            
            # Aktivasyon fonksiyonu (ayrı bir modül olarak)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            İleri geçiş: x → gizli katman → aktivasyon → çıkış → aktivasyon
            PyTorch backward'ı otomatik hesaplar (autograd).
            """
            # Gizli katman
            hidden_out = self.sigmoid(self.hidden(x))
            
            # Çıkış katmanı
            output = self.sigmoid(self.output(hidden_out))
            
            return output

    # Model oluştur
    pt_model = PyTorchMLP(input_size=2, hidden_size=4, output_size=1)
    print(f"\n  PyTorch Model Mimarisi:\n  {pt_model}")

    # Kayıp fonksiyonu ve optimizer
    criterion = nn.BCELoss()          # Binary Cross-Entropy
    optimizer = optim.SGD(
        pt_model.parameters(),
        lr=0.5                         # NumPy ve TF ile aynı lr
    )

    # Eğitim döngüsü
    print("\n  PyTorch Model Eğitimi...")
    pt_loss_history = []

    for epoch in range(10000):
        # 1. Optimizer gradyanları sıfırla
        # (PyTorch gradyanları otomatik toplar, sıfırlamazsak birikir!)
        optimizer.zero_grad()
        
        # 2. İleri geçiş
        predictions = pt_model(X_tensor)
        
        # 3. Kayıp hesapla
        loss = criterion(predictions, y_tensor)
        pt_loss_history.append(loss.item())
        
        # 4. Geri yayılım (autograd ile otomatik)
        loss.backward()
        
        # 5. Ağırlıkları güncelle
        optimizer.step()

    # Değerlendirme
    pt_model.eval()  # Eval moduna geç (dropout gibi katmanlar devre dışı)
    with torch.no_grad():  # Gradyan hesaplamayı durdur (bellek tasarrufu)
        final_preds = pt_model(X_tensor)
        pt_preds_binary = (final_preds >= 0.5).int()

    pt_loss_final = pt_loss_history[-1]
    pt_acc = (pt_preds_binary == y_tensor.int()).float().mean().item()
    print(f"\n  PT Model — Loss: {pt_loss_final:.6f} | Accuracy: {pt_acc:.1%}")

    print("\n  PyTorch Tahmin Sonuçları:")
    print(f"  {'x1':>4} {'x2':>4} | {'Beklenen':>10} | {'Tahmin':>8}")
    print(f"  {'-'*35}")
    for i in range(len(X_xor)):
        beklenen = int(y_xor[i, 0])
        tahmin = pt_preds_binary[i, 0].item()
        status = "✓" if tahmin == beklenen else "✗"
        print(f"  {int(X_xor[i,0]):>4} {int(X_xor[i,1]):>4} | "
              f"{beklenen:>10} | {tahmin:>8} {status}")

    pt_available = True

except ImportError:
    print("  [UYARI] PyTorch yüklü değil. Section 3 atlanıyor.")
    print("  Yüklemek için: pip install torch")
    pt_available = False
    pt_loss_history = []


# ============================================================
# SECTION 4: FRAMEWORK KARŞILAŞTIRMASI
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: Framework Karşılaştırması")
print("=" * 65)

print("""
  ┌─────────────────┬──────────────┬──────────────┬──────────────┐
  │ Özellik         │ NumPy        │ TensorFlow   │ PyTorch      │
  ├─────────────────┼──────────────┼──────────────┼──────────────┤
  │ Kodu kim yazar  │ Biz (sıfır)  │ Keras API    │ nn.Module    │
  │ Autograd        │ Hayır        │ Evet         │ Evet         │
  │ GPU desteği     │ Hayır        │ Evet         │ Evet         │
  │ Esneklik        │ En yüksek    │ Orta         │ Çok yüksek   │
  │ Öğrenme eğrisi  │ Düşük        │ Orta         │ Orta         │
  │ Production      │ Hayır        │ Evet         │ Evet         │
  │ Araştırma       │ -            │ Orta         │ En yaygın    │
  └─────────────────┴──────────────┴──────────────┴──────────────┘

  SONUÇ:
  • NumPy: Matematiği anlamak için. Production'da kullanılmaz.
  • TensorFlow/Keras: Hızlı prototip ve production deployment.
  • PyTorch: Araştırma, özel mimari, maksimum kontrol.
""")


# ============================================================
# SECTION 5: GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: Görselleştirme")
print("=" * 65)
print("  Grafikler oluşturuluyor...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Modül 01: Sinir Ağları Temelleri — Analiz Paneli",
             fontsize=14, fontweight='bold', y=0.98)

# ── Grafik 1: XOR Karar Sınırı ────────────────────────────────────────────────
ax1 = axes[0, 0]

# Izgara oluştur — her noktanın tahminini hesapla
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                      np.linspace(-0.5, 1.5, 200))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
grid_probs = numpy_mlp.forward(grid_points).reshape(xx.shape)

# Renk haritası (karar bölgeleri)
ax1.contourf(xx, yy, grid_probs, levels=50, cmap='RdYlBu', alpha=0.7)
ax1.contour(xx, yy, grid_probs, levels=[0.5], colors='black',
            linewidths=2, linestyles='--')

# Veri noktaları
colors_data = ['royalblue' if y == 0 else 'crimson' for y in y_xor.flatten()]
labels_data = ['XOR=0' if y == 0 else 'XOR=1' for y in y_xor.flatten()]
for i, (point, color) in enumerate(zip(X_xor, colors_data)):
    ax1.scatter(point[0], point[1], c=color, s=200, zorder=5,
                edgecolors='white', linewidth=2)
    ax1.annotate(f"({int(point[0])},{int(point[1])})",
                 (point[0], point[1]), textcoords="offset points",
                 xytext=(10, 10), fontsize=9)

ax1.set_title("XOR Karar Sınırı (NumPy MLP)", fontweight='bold')
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
patch0 = mpatches.Patch(color='royalblue', label='XOR = 0')
patch1 = mpatches.Patch(color='crimson', label='XOR = 1')
ax1.legend(handles=[patch0, patch1], loc='upper right')
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.grid(True, alpha=0.3)

# ── Grafik 2: Eğitim Loss Eğrileri ───────────────────────────────────────────
ax2 = axes[0, 1]

ax2.plot(numpy_mlp.loss_history, label='NumPy (scratch)',
         color='steelblue', linewidth=2, alpha=0.9)
if tf_available and tf_loss_history:
    ax2.plot(tf_loss_history, label='TensorFlow/Keras',
             color='darkorange', linewidth=2, alpha=0.9, linestyle='--')
if pt_available and pt_loss_history:
    ax2.plot(pt_loss_history, label='PyTorch',
             color='crimson', linewidth=2, alpha=0.9, linestyle=':')

ax2.set_title("Eğitim Loss Eğrileri (10,000 Epoch)", fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Binary Cross-Entropy Loss")
ax2.legend()
ax2.set_yscale('log')  # Log ölçek — küçük farklılıkları görmek için
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10000)

# ── Grafik 3: Aktivasyon Fonksiyonları Karşılaştırması ───────────────────────
ax3 = axes[1, 0]

z_range = np.linspace(-5, 5, 300)
ax3.plot(z_range, sigmoid(z_range), label='Sigmoid σ(z)',
         color='steelblue', linewidth=2)
ax3.plot(z_range, tanh(z_range), label='Tanh(z)',
         color='darkorange', linewidth=2)
ax3.plot(z_range, relu(z_range), label='ReLU max(0,z)',
         color='crimson', linewidth=2)

ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax3.set_title("Aktivasyon Fonksiyonları", fontweight='bold')
ax3.set_xlabel("z (ağırlıklı toplam)")
ax3.set_ylabel("f(z) (aktivasyon)")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-1.5, 2.5)
ax3.set_xlim(-5, 5)

# Aralıkları gösteren annotasyonlar
ax3.annotate("Sigmoid: (0,1)\nolasılık için",
             xy=(3, sigmoid(np.array([3]))[0]),
             xytext=(1.5, 1.8),
             fontsize=8, color='steelblue',
             arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

# ── Grafik 4: Ağ Ağırlık Dağılımı ────────────────────────────────────────────
ax4 = axes[1, 1]

all_weights = np.concatenate([
    numpy_mlp.W1.flatten(),
    numpy_mlp.W2.flatten(),
    numpy_mlp.b1.flatten(),
    numpy_mlp.b2.flatten()
])

ax4.hist(all_weights, bins=20, color='steelblue', edgecolor='white',
         alpha=0.8, linewidth=0.5)
ax4.axvline(x=np.mean(all_weights), color='crimson', linestyle='--',
            linewidth=2, label=f'Ortalama: {np.mean(all_weights):.3f}')
ax4.axvline(x=np.median(all_weights), color='darkorange', linestyle=':',
            linewidth=2, label=f'Medyan: {np.median(all_weights):.3f}')

ax4.set_title("Eğitim Sonrası Ağırlık Dağılımı", fontweight='bold')
ax4.set_xlabel("Ağırlık Değerleri")
ax4.set_ylabel("Frekans")
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/01-Sinir_Aglari_Temelleri/modul01_analiz.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi: modul01_analiz.png")


# ============================================================
# SECTION 6: ÖZET VE SONUÇ
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: Modül 01 Özeti")
print("=" * 65)
print("""
  ✓ Yapay nöron: z = Σ(wi*xi) + b, a = f(z)
  ✓ MLP: birden fazla katman → doğrusal olmayan kararlar
  ✓ XOR problemi tek katmanla çözülemez → gizli katman şart
  ✓ Forward pass: girişten çıkışa sıralı hesaplama
  ✓ Loss fonksiyonu: tahmin ile gerçek arasındaki farkı ölçer
  ✓ Backprop: zincir kuralıyla gradyanlar → ağırlık güncellemesi
  
  SONRAKI MODÜL: 02 — Aktivasyon Fonksiyonları (derinlemesine)
    - Sigmoid'in neden sorunlu olduğu
    - Vanishing gradient problemi
    - Modern aktivasyonlar: ReLU, GELU, Swish
""")

print("=" * 65)
print("  Modül 01 tamamlandı!")
print("=" * 65)
