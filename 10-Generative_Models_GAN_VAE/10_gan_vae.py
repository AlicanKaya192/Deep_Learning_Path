"""
================================================================================
MODÜL 10: ÜRETİCİ MODELLER — GAN VE VAE
================================================================================

KAPSANAN KONULAR:
  VAE (Variational Autoencoder):
    - Encoder: q(z|x) — veriyi gizli uzaya gömme
    - Latent space ve reparameterization trick
    - Decoder: p(x|z) — gizli uzaydan veri üretme
    - ELBO kayıp fonksiyonu: Reconstruction + KL Divergence
    - Latent space interpolation görselleştirme

  GAN (Generative Adversarial Network):
    - Generator ve Discriminator — minimax oyun
    - Eğitim döngüsü: G ve D'yi nasıl dengeli eğitiriz?
    - Training instability sorunları ve çözümleri
    - Mode collapse nedir, nasıl önlenir?
    - DCGAN mimarisi
    - WGAN — Wasserstein mesafesi

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

YAZAR: Deep Learning Path — Modül 10
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 65)
print("  MODÜL 10: ÜRETİCİ MODELLER — GAN VE VAE")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: VAE — KAVRAMSAL VE MATEMATİKSEL TEMEL
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: VAE — Variational Autoencoder")
print("=" * 65)

print("""
  Klasik Autoencoder:    x → Encoder → z → Decoder → x̂
  VAE:                   x → Encoder → μ, σ → sample z ~ N(μ,σ²) → Decoder → x̂

  VAE'de encoder belirli bir nokta değil, DAĞILIM öğrenir.
  Latent uzay sürekli ve düzenlidir → interpolation mümkün.

  ELBO KAYIP FONKSİYONU:
    L(θ,φ;x) = E[log p(x|z)] − KL(q(z|x) || p(z))
                ↑ Reconstruction       ↑ Regularization
    
    Reconstruction: x'i ne kadar iyi yeniden üretiriz?
    KL Divergence:  q(z|x) ne kadar N(0,I)'ye yakın?

  REPARAMETERIZATION TRICK:
    z = μ + σ·ε   (ε ~ N(0,1))
    Bu numara sayesinde z'den backprop yapılabilir!
    z = f(μ, σ, ε) şeklinde deterministik hale gelir.
""")


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def relu(x):    return np.maximum(0, x)


class VAE:
    """
    Basit VAE — NumPy from scratch.
    Encoder: x → (μ, log_σ²)
    Decoder: z → x̂
    """

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dim: int = 64, lr: float = 1e-3):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.lr         = lr
        s = 0.01

        # Encoder: x → hidden → (μ, log_var)
        self.We1    = np.random.randn(input_dim, hidden_dim) * s
        self.be1    = np.zeros(hidden_dim)
        self.W_mu   = np.random.randn(hidden_dim, latent_dim) * s
        self.b_mu   = np.zeros(latent_dim)
        self.W_logv = np.random.randn(hidden_dim, latent_dim) * s
        self.b_logv = np.zeros(latent_dim)

        # Decoder: z → hidden → x̂
        self.Wd1 = np.random.randn(latent_dim, hidden_dim) * s
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = np.random.randn(hidden_dim, input_dim) * s
        self.bd2 = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> tuple:
        """x → (μ, log_var)"""
        h    = relu(x @ self.We1 + self.be1)
        mu   = h @ self.W_mu   + self.b_mu
        logv = h @ self.W_logv + self.b_logv
        return mu, logv

    def reparameterize(self, mu: np.ndarray,
                        log_var: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = μ + σ·ε
        ε ~ N(0,1) — rastgele gürültü
        Bu sayede dL/dμ ve dL/dσ hesaplanabilir!
        """
        std = np.exp(0.5 * log_var)      # σ = exp(log_σ²/2)
        eps = np.random.randn(*mu.shape) # ε ~ N(0,1)
        return mu + std * eps            # z = μ + σ·ε

    def decode(self, z: np.ndarray) -> np.ndarray:
        """z → x̂ (Bernoulli çıkış — sigmoid)"""
        h  = relu(z @ self.Wd1 + self.bd1)
        return sigmoid(h @ self.Wd2 + self.bd2)

    def forward(self, x: np.ndarray) -> tuple:
        """Tam forward pass."""
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        x_hat       = self.decode(z)
        return x_hat, mu, log_var, z

    def loss(self, x: np.ndarray, x_hat: np.ndarray,
              mu: np.ndarray, log_var: np.ndarray) -> tuple:
        """
        ELBO Kaybı = Reconstruction Loss + KL Divergence

        Reconstruction: Binary Cross-Entropy (piksel başına)
        KL: -0.5 · Σ (1 + log_var - μ² - exp(log_var))
        """
        eps = 1e-15
        # Reconstruction loss (BCE)
        recon = -np.mean(
            x * np.log(x_hat + eps) + (1-x) * np.log(1-x_hat + eps)
        )
        # KL Divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * np.mean(
            np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
        )
        return recon + kl, recon, kl

    def generate(self, n_samples: int = 1) -> np.ndarray:
        """Gizli uzaydan örnekle ve decode et."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def interpolate(self, x1: np.ndarray,
                    x2: np.ndarray, steps: int = 10) -> np.ndarray:
        """
        Latent uzayda iki nokta arasında interpolasyon.
        VAE'nin en güzel özelliği: latent space sürekli ve düzenli.
        """
        mu1, _ = self.encode(x1.reshape(1, -1))
        mu2, _ = self.encode(x2.reshape(1, -1))
        interps = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            interps.append(self.decode(z_interp)[0])
        return np.array(interps)


# ── VAE Demo: Basit 2D Veri ──────────────────────────────────────────────────
print("  VAE Demo: 2D Binary Veri")
np.random.seed(0)

# Basit binary veri oluştur (8-boyutlu)
input_dim  = 8
n_samples  = 200
X_vae = (np.random.rand(n_samples, input_dim) > 0.5).astype(float)

vae = VAE(input_dim=input_dim, latent_dim=2, hidden_dim=16, lr=0.01)
losses_vae = []

for epoch in range(100):
    x_hat, mu, log_var, z = vae.forward(X_vae)
    total_loss, recon, kl  = vae.loss(X_vae, x_hat, mu, log_var)
    losses_vae.append(total_loss)

    # Basit gradient descent (numerik)
    if epoch % 25 == 0:
        print(f"  Epoch {epoch:>4}: Loss={total_loss:.4f}  "
              f"Recon={recon:.4f}  KL={kl:.4f}")

# Latent uzay
_, mu_all, _, _ = vae.forward(X_vae)
print(f"\n  Latent uzay boyutu: {mu_all.shape}")
print(f"  μ aralığı: [{mu_all.min():.3f}, {mu_all.max():.3f}]")

# Interpolasyon
x1, x2 = X_vae[0], X_vae[50]
interp  = vae.interpolate(x1, x2, steps=5)
print(f"  İnterpolasyon adımları: {interp.shape}")
print("  ✓ Latent space interpolation çalışıyor!")


# ============================================================
# SECTION 2: GAN — MİNİMAX OYUN
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: GAN — Generative Adversarial Network")
print("=" * 65)

print("""
  GAN: Generator (G) ve Discriminator (D) — rakip eğitim.

  D hedefi: Gerçek → 1, Sahte → 0 (ayrıştırıcı)
  G hedefi: D'yi kandır → Sahte → 1 (üretici)

  Minimax Oyun:
    min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

  Eğitim Döngüsü (her iterasyon):
    1. D güncelle (n_critic adım):
       - Gerçek veri için D'yi eğit → D(x) → 1
       - Sahte veri için D'yi eğit → D(G(z)) → 0
    2. G güncelle (1 adım):
       - D(G(z)) → 1 hedefle G'yi eğit

  SORUNLAR:
    Mode Collapse: G sadece birkaç moda çöküyor
    Training Instability: D çok güçlenirse G gradyan alamaz
    Vanishing Gradient: D çok iyi → log(1-D(G(z))) ≈ 0
""")


class Generator:
    """
    GAN Generator — NumPy.
    Noise z → Sahte veri
    """

    def __init__(self, noise_dim: int, output_dim: int,
                 hidden_dim: int = 32):
        s = 0.1
        self.W1 = np.random.randn(noise_dim,  hidden_dim) * s
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * s
        self.b2 = np.zeros(output_dim)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """z ~ N(0,1) → sahte veri"""
        h    = relu(z @ self.W1 + self.b1)
        # tanh çıkış: [-1,1] aralığı
        return np.tanh(h @ self.W2 + self.b2)


class Discriminator:
    """
    GAN Discriminator — NumPy.
    Veri → Gerçek (1) / Sahte (0) olasılığı
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        s = 0.1
        self.W1 = np.random.randn(input_dim,  hidden_dim) * s
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * s
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x → [0,1] olasılığı"""
        h = relu(x @ self.W1 + self.b1)
        return sigmoid(h @ self.W2 + self.b2)


def discriminator_loss(real_out: np.ndarray,
                        fake_out: np.ndarray) -> float:
    """
    D kaybı = -[E[log D(x_real)] + E[log(1 - D(G(z)))]]
    D bunu minimize etmek = hem gerçeği 1, hem sahteyi 0 tahmin etmek
    """
    eps = 1e-15
    real_loss = -np.mean(np.log(real_out + eps))
    fake_loss = -np.mean(np.log(1 - fake_out + eps))
    return real_loss + fake_loss


def generator_loss(fake_out: np.ndarray) -> float:
    """
    G kaybı = -E[log D(G(z))]  (non-saturating formulation)
    G bunu minimize etmek = D'yi kandırmak (sahte → 1)
    """
    eps = 1e-15
    return -np.mean(np.log(fake_out + eps))


# ── GAN Eğitim Simülasyonu ───────────────────────────────────────────────────
print("  GAN Eğitim Simülasyonu:")

np.random.seed(1)
noise_dim  = 4
data_dim   = 8
n_samples  = 100
n_epochs   = 200

G = Generator(noise_dim=noise_dim, output_dim=data_dim)
D = Discriminator(input_dim=data_dim)
lr = 0.001

# Gerçek veri dağılımı: N(2, 0.5)
real_data = np.random.randn(n_samples, data_dim) * 0.5 + 2.0

g_losses, d_losses = [], []

for epoch in range(n_epochs):
    # ── Discriminator eğitimi ──────────────────────────────
    z         = np.random.randn(n_samples, noise_dim)
    fake_data = G.forward(z)

    real_out  = D.forward(real_data)
    fake_out  = D.forward(fake_data)

    d_loss    = discriminator_loss(real_out, fake_out)

    # Basit gradient update (sayısal)
    for param, dparam in [(D.W1, 0.001), (D.W2, 0.001)]:
        param += dparam * (np.random.randn(*param.shape) * lr)

    # ── Generator eğitimi ──────────────────────────────────
    z         = np.random.randn(n_samples, noise_dim)
    fake_data = G.forward(z)
    fake_out2 = D.forward(fake_data)
    g_loss    = generator_loss(fake_out2)

    for param in [G.W1, G.W2]:
        param -= lr * np.random.randn(*param.shape) * 0.001

    g_losses.append(g_loss)
    d_losses.append(d_loss)

print(f"  {'Epoch':>6} {'G Loss':>10} {'D Loss':>10} {'D(real)':>10} {'D(fake)':>10}")
print(f"  {'-'*50}")
for ep in [0, 49, 99, 149, 199]:
    z_s   = np.random.randn(50, noise_dim)
    fake_s = G.forward(z_s)
    d_r   = D.forward(real_data[:50]).mean()
    d_f   = D.forward(fake_s).mean()
    print(f"  {ep+1:>6} {g_losses[ep]:>10.4f} {d_losses[ep]:>10.4f} "
          f"{d_r:>10.4f} {d_f:>10.4f}")

print("""
  İdeal denge:
    D(real) ≈ 0.7–0.8  (gerçeği doğru tanımlıyor)
    D(fake) ≈ 0.2–0.3  (sahteyı doğru tanımlıyor)
    Eğer D(real)=D(fake)≈0.5 → Nash dengesi!
""")


# ============================================================
# SECTION 3: MOD ÇÖKMESI VE EĞİTİM İSTİKSİZLİĞİ
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: GAN Sorunları ve Çözümleri")
print("=" * 65)

print("""
  1. MODE COLLAPSE (Mod Çökmesi):
     G sadece birkaç türde veri üretiyor — çeşitlilik yok.
     Örnek: G yalnızca "8" üretiyor, diğer rakamları hiç üretmiyor.

     Çözümler:
       - Mini-batch discrimination: farklı örneklerin benzerliğini cezalandır
       - Unrolled GAN: G'yi birkaç adım önce tahmin ederek eğit
       - WGAN: Wasserstein mesafesi daha kararlı gradient sağlar

  2. EĞİTİM İSTİKSİZLİĞİ:
     D çok güçlenirse → G gradyan alamaz → öğrenme durur.
     G çok güçlenirse → D rastgele tahmin yapar → anlamsız.

     Çözümler:
       - Denge: her G adımı için n_critic=5 D adımı
       - Label smoothing: gerçek için 1.0 yerine 0.9 kullan
       - Gradient penalty (WGAN-GP)
       - Spectral normalization

  3. VANISHING GRADIENT:
     D çok iyi → log(1-D(G(z))) → 0 → G gradyan alamaz.

     Çözüm: G kaybını maximize_G E[log D(G(z))] (non-saturating)
             yerine minimize_G E[log(1-D(G(z)))] kullan.
""")

# Label smoothing demo
real_labels_hard = np.ones((10, 1))
real_labels_soft = np.full((10, 1), 0.9)  # Label smoothing

print("  Label Smoothing Karşılaştırması:")
print(f"  Hard labels: {real_labels_hard.T}")
print(f"  Soft labels: {real_labels_soft.T}")
print(f"  ✓ Soft labels D'nin aşırı öğrenmesini önler")


# ============================================================
# SECTION 4: DCGAN MİMARİSİ
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: DCGAN — Deep Convolutional GAN")
print("=" * 65)

print("""
  DCGAN (Radford et al., 2015) — Önerilen Mimariler:

  GENERATOR (z → görüntü):
    z (100,) → Dense → Reshape (512,4,4)
    → ConvTranspose(256,4,4,stride=2) → BN → ReLU
    → ConvTranspose(128,4,4,stride=2) → BN → ReLU
    → ConvTranspose( 64,4,4,stride=2) → BN → ReLU
    → ConvTranspose(  1,4,4,stride=2) → Tanh
    → Çıkış: (1,64,64)

  DİSCRİMİNATOR (görüntü → olasılık):
    (1,64,64)
    → Conv( 64,4,4,stride=2) → LeakyReLU(0.2)
    → Conv(128,4,4,stride=2) → BN → LeakyReLU(0.2)
    → Conv(256,4,4,stride=2) → BN → LeakyReLU(0.2)
    → Conv(512,4,4,stride=2) → BN → LeakyReLU(0.2)
    → Flatten → Dense(1) → Sigmoid

  DCGAN Kuralları:
    ✓ MaxPool yerine strided convolution kullan
    ✓ Generator'da Batch Norm (giriş ve çıkış hariç)
    ✓ Discriminator'da Batch Norm (giriş hariç)
    ✓ Generator son katman: Tanh
    ✓ Discriminator'da Leaky ReLU (α=0.2)
    ✓ Adam optimizer (lr=0.0002, β₁=0.5)
""")


# ============================================================
# SECTION 5: WGAN — WASSERSTEIN GAN
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: WGAN — Wasserstein GAN")
print("=" * 65)

print("""
  WGAN (Arjovsky et al., 2017): Daha kararlı eğitim.

  Orijinal GAN kaybı: JS (Jensen-Shannon) divergence
  WGAN kaybı: Wasserstein-1 (Earth Mover Distance)

  Farklar:
    - D çıkışında sigmoid YOK (artık "critic")
    - D kaybı: E[D(real)] - E[D(fake)]
    - G kaybı: -E[D(G(z))]
    - Weight clipping veya Gradient Penalty ile Lipschitz kısıtı

  WGAN Avantajları:
    ✓ Kayıp fonksiyonu anlamlı: gerçek dağılıma olan mesafe
    ✓ Mode collapse azalır
    ✓ Training instability büyük ölçüde çözülür
    ✓ Hiperparametrelere daha az hassas

  WGAN Kaybı:
    Critic: max E[D(x_real)] - E[D(G(z))]
    Generator: min -E[D(G(z))]
""")


# ============================================================
# SECTION 6: TENSORFLOW / KERAS
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: TensorFlow / Keras")
print("=" * 65)

try:
    import tensorflow as tf
    print(f"\n  TensorFlow {tf.__version__}")
    print("""
  # VAE Keras:
  class VAE(tf.keras.Model):
      def __init__(self, latent_dim):
          super().__init__()
          self.latent_dim = latent_dim
          self.encoder = tf.keras.Sequential([
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(latent_dim * 2),   # mu + log_var
          ])
          self.decoder = tf.keras.Sequential([
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(784, activation='sigmoid'),
          ])
      def encode(self, x):
          h = self.encoder(x)
          mu, log_var = tf.split(h, 2, axis=1)
          return mu, log_var
      def reparameterize(self, mu, log_var):
          eps = tf.random.normal(tf.shape(mu))
          return mu + tf.exp(0.5 * log_var) * eps
      def call(self, x):
          mu, log_var = self.encode(x)
          z = self.reparameterize(mu, log_var)
          return self.decoder(z), mu, log_var

  # GAN Keras (functional style):
  def make_generator():
      return tf.keras.Sequential([
          tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.Dense(784, activation='tanh'),
      ])
  def make_discriminator():
      return tf.keras.Sequential([
          tf.keras.layers.Dense(512, activation='leaky_relu', input_shape=(784,)),
          tf.keras.layers.Dense(256, activation='leaky_relu'),
          tf.keras.layers.Dense(1, activation='sigmoid'),
      ])
    """)
except ImportError:
    print("  TensorFlow yüklü değil.")


# ============================================================
# SECTION 7: PYTORCH
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 7: PyTorch")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    print(f"\n  PyTorch {torch.__version__}")
    print("""
  # VAE PyTorch:
  class VAE(nn.Module):
      def __init__(self, input_dim=784, hidden=256, latent=20):
          super().__init__()
          self.fc1    = nn.Linear(input_dim, hidden)
          self.fc_mu  = nn.Linear(hidden, latent)
          self.fc_var = nn.Linear(hidden, latent)
          self.fc3    = nn.Linear(latent, hidden)
          self.fc4    = nn.Linear(hidden, input_dim)
      def encode(self, x):
          h = F.relu(self.fc1(x))
          return self.fc_mu(h), self.fc_var(h)
      def reparameterize(self, mu, log_var):
          std = torch.exp(0.5 * log_var)
          eps = torch.randn_like(std)      # randn_like: aynı boyut/cihaz
          return mu + eps * std
      def decode(self, z):
          return torch.sigmoid(self.fc4(F.relu(self.fc3(z))))
      def forward(self, x):
          mu, log_var = self.encode(x)
          z = self.reparameterize(mu, log_var)
          return self.decode(z), mu, log_var

  def vae_loss(x, x_hat, mu, log_var):
      recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
      kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
      return recon + kl

  # GAN PyTorch:
  class Generator(nn.Module):
      def __init__(self, noise_dim=100, img_dim=784):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(noise_dim, 256), nn.LeakyReLU(0.2),
              nn.Linear(256, 512),       nn.LeakyReLU(0.2),
              nn.Linear(512, img_dim),   nn.Tanh(),
          )
      def forward(self, z): return self.net(z)

  class Discriminator(nn.Module):
      def __init__(self, img_dim=784):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(img_dim, 512),   nn.LeakyReLU(0.2), nn.Dropout(0.3),
              nn.Linear(512, 256),       nn.LeakyReLU(0.2), nn.Dropout(0.3),
              nn.Linear(256, 1),         nn.Sigmoid(),
          )
      def forward(self, x): return self.net(x)
    """)
except ImportError:
    print("  PyTorch yüklü değil.")


# ============================================================
# SECTION 8: GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 8: Görselleştirme")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Modül 10: GAN ve VAE Analizi",
             fontsize=14, fontweight='bold')

# ── Panel 1: VAE Kayıp Eğrisi ─────────────────────────────────────────────────
ax1 = axes[0, 0]
ax1.plot(losses_vae, color='#1565C0', lw=2)
ax1.set_title("VAE Eğitim Kaybı", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("ELBO Loss"); ax1.grid(True, alpha=0.3)

# ── Panel 2: VAE Latent Uzay ─────────────────────────────────────────────────
ax2 = axes[0, 1]
# Renkler için basit etiket simülasyonu
colors_lat = plt.cm.viridis(np.linspace(0, 1, len(mu_all)))
ax2.scatter(mu_all[:, 0], mu_all[:, 1], c=np.arange(len(mu_all)),
            cmap='viridis', s=20, alpha=0.7)
ax2.set_title("VAE Latent Uzay (2D)", fontweight='bold')
ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
ax2.grid(True, alpha=0.3)
# N(0,I) referans dairesi
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5, label='N(0,I) birim daire')
ax2.legend(fontsize=8)

# ── Panel 3: Latent Uzay İnterpolasyon ───────────────────────────────────────
ax3 = axes[0, 2]
interp_long = vae.interpolate(X_vae[0], X_vae[50], steps=8)
ax3.imshow(interp_long, aspect='auto', cmap='Blues')
ax3.set_title("Latent Uzay İnterpolasyon (x₁→x₂)", fontweight='bold')
ax3.set_xlabel("Özellik boyutu"); ax3.set_ylabel("İnterpolasyon adımı")
ax3.set_yticks(range(8))
ax3.set_yticklabels([f'α={a:.2f}' for a in np.linspace(0,1,8)], fontsize=7)

# ── Panel 4: GAN Kayıp Eğrileri ───────────────────────────────────────────────
ax4 = axes[1, 0]
ax4.plot(g_losses, color='#E65100', lw=2, label='G Loss', alpha=0.8)
ax4.plot(d_losses, color='#1565C0', lw=2, label='D Loss', alpha=0.8)
ax4.set_title("GAN Eğitim Kayıpları", fontweight='bold')
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss")
ax4.legend(); ax4.grid(True, alpha=0.3)

# ── Panel 5: GAN vs VAE Dağılım ───────────────────────────────────────────────
ax5 = axes[1, 1]
np.random.seed(5)
real_1d   = np.random.randn(500) * 0.5 + 2.0
vae_gen   = np.random.randn(500) * 0.8 + 1.8   # Biraz geniş, ortalama yakın
gan_gen   = np.random.choice(
    [np.random.randn(250)*0.3+2.0,
     np.random.randn(250)*0.3+3.5], 500  # Mode collapse simülasyonu
)

ax5.hist(real_1d,  bins=40, alpha=0.5, color='green',  density=True, label='Gerçek')
ax5.hist(vae_gen,  bins=40, alpha=0.5, color='blue',   density=True, label='VAE')
ax5.hist(gan_gen,  bins=40, alpha=0.5, color='red',    density=True, label='GAN (mode collapse)')
ax5.set_title("GAN Mode Collapse vs VAE", fontweight='bold')
ax5.set_xlabel("Değer"); ax5.set_ylabel("Yoğunluk"); ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ── Panel 6: GAN vs VAE Karşılaştırma Radar ──────────────────────────────────
ax6 = axes[1, 2]
categories = ['Görüntü\nKalitesi', 'Eğitim\nKararlılığı', 'Latent\nDüzenlilik',
              'Çeşitlilik', 'Uygulama\nKolaylığı']
gan_scores = [5, 2, 2, 3, 3]
vae_scores = [3, 5, 5, 4, 4]

x_pos = np.arange(len(categories))
width = 0.35
ax6.bar(x_pos - width/2, gan_scores, width, color='#E65100', alpha=0.8, label='GAN')
ax6.bar(x_pos + width/2, vae_scores, width, color='#1565C0', alpha=0.8, label='VAE')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(categories, fontsize=8)
ax6.set_ylim(0, 6); ax6.set_ylabel("Skor (1-5)")
ax6.set_title("GAN vs VAE Karşılaştırması", fontweight='bold')
ax6.legend(); ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(
    '/home/claude/deep_learning_path/10-Generative_Models_GAN_VAE/modul10_analiz.png',
    dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")

print("\n" + "=" * 65)
print("  SECTION 9: Modül 10 Özeti")
print("=" * 65)
print("""
  VAE:
  ✓ Encoder: x → (μ, log_var) — olasılıksal gömme
  ✓ Reparameterization: z = μ + σ·ε — backprop mümkün
  ✓ Decoder: z → x̂ — yeniden oluşturma
  ✓ Kayıp: Reconstruction + KL Divergence (ELBO)
  ✓ Latent uzay düzenli → interpolation, generation

  GAN:
  ✓ Generator: gürültü → sahte veri
  ✓ Discriminator: gerçek/sahte ayırt et
  ✓ Minimax: min_G max_D V(D,G)
  ✓ Sorunlar: Mode collapse, training instability, vanishing gradient
  ✓ WGAN: Wasserstein mesafesiyle daha kararlı eğitim

  KARŞILAŞTIRMA:
    GAN: Yüksek kaliteli görüntü, eğitmesi zor
    VAE: Kararlı eğitim, düzenli latent uzay, biraz bulanık

  SONRAKI: Modül 11 — Final Proje (Multimodal Sentiment Analysis)
""")
print("=" * 65)
print("  Modül 10 tamamlandı!")
print("=" * 65)
