"""
================================================================================
MODÜL 08: RNN, LSTM VE GRU
================================================================================

KAPSANAN KONULAR:
  - Vanilla RNN — forward pass, gizli durum, BPTT
  - Vanishing gradient in RNNs — matematiksel kanıt
  - LSTM — 4 kapı (forget, input, output, cell), tam diyagram
  - GRU — 2 kapı, LSTM'den fark
  - Bidirectional RNN/LSTM
  - Sequence-to-Sequence (Encoder-Decoder) mimarisi
  - Zaman serisi tahmini (NumPy from scratch)
  - Metin üretimi (character-level)
  - TensorFlow ve PyTorch implementasyonları

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

YAZAR: Deep Learning Path — Modül 08
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 65)
print("  MODÜL 08: RNN, LSTM VE GRU")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: VANILLA RNN — FROM SCRATCH
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Vanilla RNN — Sıfırdan NumPy")
print("=" * 65)

print("""
  RNN: Her adımda önceki gizli durumu (hidden state) taşır.

  Forward Pass:
    hₜ = tanh(Wₓₓ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)   [gizli durum güncelleme]
    yₜ = Wᵧₕ·hₜ + bᵧ                        [çıkış hesaplama]

  Parametreler:
    Wₓₓ : giriş→gizli ağırlıkları  (n_hidden × n_input)
    Wₕₕ : gizli→gizli ağırlıkları  (n_hidden × n_hidden)
    Wᵧₕ : gizli→çıkış ağırlıkları  (n_output × n_hidden)
""")

class VanillaRNN:
    """Vanilla RNN — NumPy from scratch."""

    def __init__(self, n_input: int, n_hidden: int, n_output: int,
                 lr: float = 0.01):
        self.n_hidden = n_hidden
        self.lr       = lr
        scale = 0.01
        # Ağırlıklar
        self.Wxh = np.random.randn(n_hidden, n_input)  * scale
        self.Whh = np.random.randn(n_hidden, n_hidden) * scale
        self.Why = np.random.randn(n_output, n_hidden) * scale
        self.bh  = np.zeros((n_hidden, 1))
        self.by  = np.zeros((n_output, 1))

    def forward(self, inputs: list, h_prev: np.ndarray) -> tuple:
        """
        inputs : T adet (n_input, 1) vektör listesi
        h_prev : (n_hidden, 1) başlangıç gizli durumu
        """
        xs, hs, ys = {}, {}, {}
        hs[-1] = h_prev.copy()
        loss = 0.0

        for t, x in enumerate(inputs):
            xs[t] = x
            # Gizli durum: tanh(Wxh·x + Whh·h_prev + bh)
            hs[t] = np.tanh(self.Wxh @ x + self.Whh @ hs[t-1] + self.bh)
            # Çıkış (ham logit)
            ys[t] = self.Why @ hs[t] + self.by

        self.cache = (xs, hs, ys)
        return hs, ys

    def softmax(self, y: np.ndarray) -> np.ndarray:
        e = np.exp(y - y.max())
        return e / e.sum()

    def sample(self, h: np.ndarray, seed_ix: int,
               n: int, vocab_size: int) -> list:
        """Eğitilmiş modelden karakter örnekle."""
        x   = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for _ in range(n):
            h    = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y    = self.Why @ h + self.by
            p    = self.softmax(y)
            ix   = np.random.choice(range(vocab_size), p=p.ravel())
            x    = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes


# ── Basit Dizi Tahmini ──────────────────────────────────────────────────────
# Sinüs dalgası tahmini: t anındaki değerden t+1'i tahmin et
T        = 100
t_vals   = np.linspace(0, 4*np.pi, T)
series   = np.sin(t_vals)

# Veriyi giriş/hedef çiftlerine çevir
seq_len  = 10
X_seq    = [series[i:i+seq_len] for i in range(T - seq_len - 1)]
y_seq    = [series[i+seq_len]   for i in range(T - seq_len - 1)]

rnn = VanillaRNN(n_input=1, n_hidden=16, n_output=1, lr=0.01)
h0  = np.zeros((16, 1))

print("  Sinüs Serisi Tahmini (Vanilla RNN, 20 epoch):")
losses = []
for epoch in range(20):
    epoch_loss = 0.0
    h = h0.copy()
    for x_seq, y_target in zip(X_seq[:50], y_seq[:50]):
        inputs = [np.array([[v]]) for v in x_seq]
        hs, ys = rnn.forward(inputs, h)
        h      = hs[seq_len - 1]
        # MSE kaybı (basit)
        pred  = ys[seq_len - 1][0, 0]
        err   = pred - y_target
        epoch_loss += err ** 2
        # Basit güncelleme (tam BPTT yerine)
        dy = np.array([[2 * err]])
        rnn.Why -= rnn.lr * dy @ hs[seq_len-1].T
        rnn.by  -= rnn.lr * dy
    losses.append(epoch_loss / 50)

print(f"  Epoch 1  Loss: {losses[0]:.4f}")
print(f"  Epoch 10 Loss: {losses[9]:.4f}")
print(f"  Epoch 20 Loss: {losses[-1]:.4f}")


# ============================================================
# SECTION 2: LSTM — TAM İMPLEMENTASYON
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: LSTM — Long Short-Term Memory")
print("=" * 65)

print("""
  LSTM 4 Kapı Mekanizması:

  Forget Gate  : fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)   → Eski bilgiyi ne kadar unut?
  Input Gate   : iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)   → Yeni bilgiyi ne kadar ekle?
  Cell Update  : c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc) → Aday hücre durumu
  Cell State   : cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ         → Uzun vadeli bellek
  Output Gate  : oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)   → Çıkışı kontrol et
  Hidden State : hₜ = oₜ⊙tanh(cₜ)              → Kısa vadeli bellek
""")

class LSTMCell:
    """Tek LSTM hücresi — NumPy from scratch."""

    def __init__(self, n_input: int, n_hidden: int):
        self.n_hidden = n_hidden
        n = n_input + n_hidden
        scale = 0.1

        # 4 kapı için ağırlıklar (birleştirilmiş: [h, x] → gate)
        self.Wf = np.random.randn(n_hidden, n) * scale  # forget
        self.Wi = np.random.randn(n_hidden, n) * scale  # input
        self.Wc = np.random.randn(n_hidden, n) * scale  # cell
        self.Wo = np.random.randn(n_hidden, n) * scale  # output
        self.bf = np.zeros((n_hidden, 1))
        self.bi = np.zeros((n_hidden, 1))
        self.bc = np.zeros((n_hidden, 1))
        self.bo = np.zeros((n_hidden, 1))

    @staticmethod
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray, h_prev: np.ndarray,
                c_prev: np.ndarray) -> tuple:
        """
        x     : (n_input, 1)
        h_prev: (n_hidden, 1)
        c_prev: (n_hidden, 1)
        Returns: h_next, c_next, gate_values (cache için)
        """
        # Birleştirilmiş giriş
        concat = np.vstack([h_prev, x])   # (n_input+n_hidden, 1)

        # 4 kapı hesabı
        f  = self.sigmoid(self.Wf @ concat + self.bf)   # Forget gate
        i  = self.sigmoid(self.Wi @ concat + self.bi)   # Input gate
        c_tilde = np.tanh(self.Wc @ concat + self.bc)   # Aday hücre
        o  = self.sigmoid(self.Wo @ concat + self.bo)   # Output gate

        # Hücre durumu güncelleme
        c_next = f * c_prev + i * c_tilde                # Uzun vadeli bellek
        h_next = o * np.tanh(c_next)                     # Kısa vadeli bellek

        return h_next, c_next, (f, i, c_tilde, o, concat, c_prev)

# ── LSTM Demo ──────────────────────────────────────────────────────────────
np.random.seed(0)
lstm_cell = LSTMCell(n_input=4, n_hidden=8)
h0  = np.zeros((8, 1))
c0  = np.zeros((8, 1))

# 5 adım ileri geçiş
h, c = h0.copy(), c0.copy()
print("  LSTM 5 Adım Forward Pass:")
print(f"  {'Adım':>5} {'||h||':>10} {'||c||':>10} {'f_mean':>10} {'i_mean':>10}")
print(f"  {'-'*50}")
for t in range(5):
    x     = np.random.randn(4, 1)
    h, c, (f, i, c_t, o, *_) = lstm_cell.forward(x, h, c)
    print(f"  {t+1:>5} {np.linalg.norm(h):>10.4f} {np.linalg.norm(c):>10.4f} "
          f"{f.mean():>10.4f} {i.mean():>10.4f}")

print("\n  Forget gate ortalaması: ~1 → bilgiyi koru, ~0 → unut")
print("  Input gate ortalaması:  ~1 → yeni bilgi ekle, ~0 → ekleme")


# ============================================================
# SECTION 3: GRU — NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: GRU — Gated Recurrent Unit")
print("=" * 65)

print("""
  GRU — LSTM'in basitleştirilmiş versiyonu (2 kapı):

  Reset Gate  : rₜ = σ(Wr·[hₜ₋₁, xₜ])   → Geçmişi ne kadar sıfırla?
  Update Gate : zₜ = σ(Wz·[hₜ₋₁, xₜ])   → Eski/yeni karışım oranı
  Aday       : h̃ₜ = tanh(Wh·[rₜ⊙hₜ₋₁, xₜ])
  Çıkış      : hₜ = (1−zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ

  LSTM vs GRU:
    LSTM: Cell state + Hidden state (2 ayrı durum)
    GRU : Tek Hidden state (daha basit)
    GRU genellikle LSTM'e yakın performans, daha az parametre.
""")

class GRUCell:
    """Tek GRU hücresi — NumPy from scratch."""

    def __init__(self, n_input: int, n_hidden: int):
        n = n_input + n_hidden
        scale = 0.1
        self.Wr = np.random.randn(n_hidden, n) * scale  # reset
        self.Wz = np.random.randn(n_hidden, n) * scale  # update
        self.Wh = np.random.randn(n_hidden, n) * scale  # candidate
        self.br = np.zeros((n_hidden, 1))
        self.bz = np.zeros((n_hidden, 1))
        self.bh = np.zeros((n_hidden, 1))

    @staticmethod
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray,
                h_prev: np.ndarray) -> np.ndarray:
        concat = np.vstack([h_prev, x])

        r      = self.sigmoid(self.Wr @ concat + self.br)  # Reset gate
        z      = self.sigmoid(self.Wz @ concat + self.bz)  # Update gate
        # Aday: reset gate geçmiş gizli durumu filtreler
        concat_r = np.vstack([r * h_prev, x])
        h_tilde  = np.tanh(self.Wh @ concat_r + self.bh)  # Aday
        # Güncelleme: eski ve yeni karıştır
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next, (r, z, h_tilde)

np.random.seed(0)
gru_cell = GRUCell(n_input=4, n_hidden=8)
h_gru = np.zeros((8, 1))

print("  GRU 5 Adım Forward Pass:")
print(f"  {'Adım':>5} {'||h||':>10} {'r_mean':>10} {'z_mean':>10}")
print(f"  {'-'*40}")
for t in range(5):
    x = np.random.randn(4, 1)
    h_gru, (r, z, _) = gru_cell.forward(x, h_gru)
    print(f"  {t+1:>5} {np.linalg.norm(h_gru):>10.4f} "
          f"{r.mean():>10.4f} {z.mean():>10.4f}")


# ============================================================
# SECTION 4: KARAKTER DÜZEYİNDE METİN ÜRETİMİ
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: Karakter Düzeyinde Dil Modeli")
print("=" * 65)

# Küçük bir metin
text    = "deep learning is fun and powerful"
chars   = sorted(set(text))
vocab   = {c: i for i, c in enumerate(chars)}
ix2ch   = {i: c for c, i in vocab.items()}
vocab_size = len(chars)

print(f"  Metin: '{text}'")
print(f"  Kelime haznesi: {vocab_size} karakter: {chars}")

# LSTM tabanlı karakter modeli — basit tek katman
class CharLSTM:
    """Karakter düzeyinde LSTM dil modeli."""

    def __init__(self, vocab_size: int, hidden_size: int, lr: float = 0.01):
        self.V    = vocab_size
        self.H    = hidden_size
        self.lr   = lr
        n         = vocab_size + hidden_size
        scale     = 0.1

        # 4 kapı ağırlıkları (birleştirilmiş)
        self.W    = np.random.randn(4 * hidden_size, n) * scale
        self.b    = np.zeros((4 * hidden_size, 1))
        # Çıkış katmanı
        self.Wy   = np.random.randn(vocab_size, hidden_size) * scale
        self.by   = np.zeros((vocab_size, 1))

    def sigmoid(self, x): return 1/(1+np.exp(-np.clip(x,-500,500)))

    def forward_step(self, x: np.ndarray,
                      h: np.ndarray, c: np.ndarray) -> tuple:
        concat = np.vstack([h, x])
        gates  = self.W @ concat + self.b     # (4H, 1)
        H      = self.H
        f  = self.sigmoid(gates[:H])
        i  = self.sigmoid(gates[H:2*H])
        ct = np.tanh(gates[2*H:3*H])
        o  = self.sigmoid(gates[3*H:])
        c_new = f * c + i * ct
        h_new = o * np.tanh(c_new)
        y_raw = self.Wy @ h_new + self.by
        # Softmax
        e = np.exp(y_raw - y_raw.max())
        p = e / e.sum()
        return h_new, c_new, p

    def sample(self, seed_char: str, n_chars: int = 50) -> str:
        h = np.zeros((self.H, 1))
        c = np.zeros((self.H, 1))
        x = np.zeros((self.V, 1))
        if seed_char in vocab:
            x[vocab[seed_char]] = 1
        result = seed_char
        for _ in range(n_chars):
            h, c, p = self.forward_step(x, h, c)
            ix = np.random.choice(self.V, p=p.ravel())
            x  = np.zeros((self.V, 1))
            x[ix] = 1
            result += ix2ch[ix]
        return result

# Hızlı demo (eğitim değil — sadece sampling yapısı)
model = CharLSTM(vocab_size=vocab_size, hidden_size=32)
sampled = model.sample('d', n_chars=30)
print(f"\n  Rastgele sampling (eğitimsiz): '{sampled}'")
print(f"  (Eğitim sonrası anlamlı metin üretilir)")


# ============================================================
# SECTION 5: BİDİRECTİONAL RNN
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: Bidirectional RNN")
print("=" * 65)

print("""
  Bidirectional RNN: İki RNN — biri ileri, biri geri yönde tarar.
  Her adımın çıktısı: [h_ileri, h_geri] birleşimi

  x₁ → x₂ → x₃ → x₄   (İleri yön)
  x₁ ← x₂ ← x₃ ← x₄   (Geri yön)

  hₜ = [h_forward_t ; h_backward_t]

  Avantaj: Her token için hem sol hem sağ bağlam görülür.
  BERT'in pre-training'i çift yönlü Transformer kullanır.
  Dezavantaj: Gerçek zamanlı (online) tahmin yapılamaz — tüm dizi gerekli.
""")

# Basit bidirectional demo
def simple_rnn_step(x, h, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x + W_hh @ h + b_h)

n_input, n_hidden = 4, 8
Wxh_f = np.random.randn(n_hidden, n_input)  * 0.1
Whh_f = np.random.randn(n_hidden, n_hidden) * 0.1
bh_f  = np.zeros((n_hidden, 1))
Wxh_b = np.random.randn(n_hidden, n_input)  * 0.1
Whh_b = np.random.randn(n_hidden, n_hidden) * 0.1
bh_b  = np.zeros((n_hidden, 1))

seq = [np.random.randn(n_input, 1) for _ in range(5)]

# İleri geçiş
h_f = np.zeros((n_hidden, 1))
h_forwards = []
for x in seq:
    h_f = simple_rnn_step(x, h_f, Wxh_f, Whh_f, bh_f)
    h_forwards.append(h_f.copy())

# Geri geçiş
h_b = np.zeros((n_hidden, 1))
h_backwards = []
for x in reversed(seq):
    h_b = simple_rnn_step(x, h_b, Wxh_b, Whh_b, bh_b)
    h_backwards.insert(0, h_b.copy())

# Birleştir
bidir_outputs = [np.vstack([hf, hb])
                 for hf, hb in zip(h_forwards, h_backwards)]

print(f"  Giriş dizisi: T=5, n_input={n_input}")
print(f"  İleri RNN gizli boyutu: {n_hidden}")
print(f"  Bidirectional çıkış boyutu: {bidir_outputs[0].shape[0]} "
      f"(= {n_hidden} × 2)")


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
  # Vanilla RNN:
  model = tf.keras.Sequential([
      tf.keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(T, n_features)),
      tf.keras.layers.SimpleRNN(32),
      tf.keras.layers.Dense(1),
  ])

  # LSTM:
  model = tf.keras.Sequential([
      tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(T, n_features)),
      tf.keras.layers.LSTM(32, return_sequences=False),
      tf.keras.layers.Dense(num_classes, activation='softmax'),
  ])

  # GRU:
  tf.keras.layers.GRU(64, return_sequences=True)

  # Bidirectional:
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))

  # return_sequences=True  : Her adımın çıktısı → (batch, T, hidden)
  # return_sequences=False : Sadece son adım    → (batch, hidden)
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
  # LSTM:
  lstm = nn.LSTM(input_size=10, hidden_size=64,
                 num_layers=2, batch_first=True,
                 dropout=0.3, bidirectional=False)

  # İleri geçiş:
  x = torch.randn(32, 20, 10)   # (batch, seq_len, input_size)
  h0 = torch.zeros(2, 32, 64)   # (num_layers, batch, hidden)
  c0 = torch.zeros(2, 32, 64)
  output, (hn, cn) = lstm(x, (h0, c0))
  # output: (32, 20, 64) — tüm adımlar
  # hn    : (2, 32, 64)  — son gizli durum
  # cn    : (2, 32, 64)  — son hücre durumu

  # GRU:
  gru = nn.GRU(10, 64, num_layers=2, batch_first=True)
  output, hn = gru(x)

  # Bidirectional:
  blstm = nn.LSTM(10, 64, bidirectional=True, batch_first=True)
  output, _ = blstm(x)
  # output: (32, 20, 128) — her adım için 2×64

  # Sequence classification modeli:
  class TextClassifier(nn.Module):
      def __init__(self, vocab_size, embed_dim, hidden, n_class):
          super().__init__()
          self.embed = nn.Embedding(vocab_size, embed_dim)
          self.lstm  = nn.LSTM(embed_dim, hidden, batch_first=True)
          self.fc    = nn.Linear(hidden, n_class)
      def forward(self, x):
          x = self.embed(x)
          _, (hn, _) = self.lstm(x)
          return self.fc(hn[-1])
    """)
except ImportError:
    print("  PyTorch yüklü değil.")


# ============================================================
# SECTION 8: GÖRSELLEŞTİRME
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Modül 08: RNN / LSTM / GRU Analizi",
             fontsize=14, fontweight='bold')

# ── Panel 1: RNN Training Loss ────────────────────────────────────────────────
ax1 = axes[0, 0]
ax1.plot(losses, color='#1565C0', lw=2)
ax1.set_title("Vanilla RNN — Sinüs Serisi Eğitim Loss", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss"); ax1.grid(True, alpha=0.3)

# ── Panel 2: LSTM Kapı Değerleri Dağılımı ────────────────────────────────────
ax2 = axes[0, 1]
np.random.seed(5)
n_steps = 50
f_vals, i_vals, o_vals = [], [], []
h = np.zeros((8,1)); c = np.zeros((8,1))
lstm_demo = LSTMCell(n_input=4, n_hidden=8)
for _ in range(n_steps):
    x = np.random.randn(4,1)
    h, c, (f, i_g, c_t, o, *_) = lstm_demo.forward(x, h, c)
    f_vals.append(f.mean()); i_vals.append(i_g.mean()); o_vals.append(o.mean())

ax2.plot(f_vals, color='#E65100', lw=2, label='Forget Gate (f)')
ax2.plot(i_vals, color='#1565C0', lw=2, label='Input Gate (i)')
ax2.plot(o_vals, color='#2E7D32', lw=2, label='Output Gate (o)')
ax2.set_title("LSTM Kapı Aktivasyonları (50 adım)", fontweight='bold')
ax2.set_xlabel("Adım"); ax2.set_ylabel("Ortalama Aktivasyon")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1)

# ── Panel 3: LSTM vs GRU Parametre Karşılaştırması ────────────────────────────
ax3 = axes[1, 0]
hidden_sizes = [16, 32, 64, 128, 256]
input_size   = 32
lstm_params  = [4*(h*(h+input_size)+h) for h in hidden_sizes]
gru_params   = [3*(h*(h+input_size)+h) for h in hidden_sizes]
rnn_params   = [h*(h+input_size)+h for h in hidden_sizes]

ax3.plot(hidden_sizes, lstm_params, 'b-o', lw=2, ms=6, label='LSTM (4 kapı)')
ax3.plot(hidden_sizes, gru_params,  'g-s', lw=2, ms=6, label='GRU (3 kapı)')
ax3.plot(hidden_sizes, rnn_params,  'r-^', lw=2, ms=6, label='Vanilla RNN (1 kapı)')
ax3.set_title("Parametre Sayısı Karşılaştırması", fontweight='bold')
ax3.set_xlabel("Gizli Boyut"); ax3.set_ylabel("Parametre Sayısı")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

# ── Panel 4: Vanishing Gradient in RNN ────────────────────────────────────────
ax4 = axes[1, 1]
steps = np.arange(1, 51)
# tanh türevi maks 1 ama tipik ~0.5
rnn_grad  = (0.5) ** steps
lstm_grad = np.ones(50) * 0.9  # LSTM neredeyse sabit (cell state highway)
gru_grad  = (0.85) ** steps

ax4.semilogy(steps, rnn_grad,  'r-', lw=2, label='Vanilla RNN (tipik 0.5/adım)')
ax4.semilogy(steps, lstm_grad, 'b-', lw=2, label='LSTM (cell state koruması)')
ax4.semilogy(steps, gru_grad,  'g-', lw=2, label='GRU (update gate koruması)')
ax4.set_title("Uzun Vadeli Bağımlılık — Gradyan Akışı", fontweight='bold')
ax4.set_xlabel("Geri Adım Sayısı"); ax4.set_ylabel("Gradyan Büyüklüğü (log)")
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/08-RNN_LSTM_GRU/modul08_analiz.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")

print("\n" + "=" * 65)
print("  Modül 08 tamamlandı!")
print("=" * 65)
