"""
================================================================================
MODÜL 09: TRANSFORMER VE ATTENTION MEKANİZMASI
================================================================================

KAPSANAN KONULAR:
  - Attention mekanizması — scaled dot-product attention
  - Multi-head attention — neden birden fazla kafa?
  - Positional encoding — sinüzoidal ve öğrenilen
  - Transformer encoder bloğu (tam implementasyon)
  - Transformer decoder bloğu
  - BERT ve GPT kavramsal özet (pre-training hedefleri)
  - Mini Transformer from scratch — PyTorch
  - Attention ağırlığı görselleştirme

GEREKLİ KÜTÜPHANELER:
  pip install numpy matplotlib

YAZAR: Deep Learning Path — Modül 09
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 65)
print("  MODÜL 09: TRANSFORMER VE ATTENTION")
print("  Deep Learning Path")
print("=" * 65)


# ============================================================
# SECTION 1: SCALED DOT-PRODUCT ATTENTION — NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1: Scaled Dot-Product Attention")
print("=" * 65)

print("""
  Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V

  Q (Query)  : "Ne arıyorum?" matrisi
  K (Key)    : "Neyle eşleşirim?" matrisi
  V (Value)  : "Eşleşince ne döneceğim?" matrisi
  dₖ         : Key/Query boyutu (√dₖ ile ölçekleme — kararlılık için)

  Adımlar:
    1. Skorlar: S = Q·Kᵀ / √dₖ       → her sorgu-anahtar uyumu
    2. Softmax: A = softmax(S)         → olasılık dağılımı (attention ağırlıkları)
    3. Çıkış  : O = A · V             → ağırlıklı değer toplamı
""")

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                   V: np.ndarray,
                                   mask: np.ndarray = None) -> tuple:
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    mask: (seq_len, seq_len) — None veya -inf değerli maske
    """
    d_k     = Q.shape[-1]
    scores  = Q @ K.T / np.sqrt(d_k)      # (seq_len, seq_len)

    if mask is not None:
        scores = scores + mask

    attn_weights = softmax(scores, axis=-1) # (seq_len, seq_len)
    output       = attn_weights @ V         # (seq_len, d_v)
    return output, attn_weights

# ── Demo ──────────────────────────────────────────────────────────────────────
seq_len, d_k, d_v = 5, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, attn = scaled_dot_product_attention(Q, K, V)
print(f"  Q boyutu: {Q.shape}  K: {K.shape}  V: {V.shape}")
print(f"  Attention ağırlıkları: {attn.shape}")
print(f"  Çıkış boyutu: {output.shape}")
print(f"\n  Her satırın toplamı 1 mi? {np.allclose(attn.sum(axis=-1), 1.0)}")
print(f"  Attention ağırlıkları (satır 0): {attn[0].round(3)}")

# Causal (decoder) mask örneği
causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
_, attn_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"\n  Causal mask uygulandı — üst üçgen sıfır:")
print(f"  {(attn_causal < 1e-6).astype(int)}")


# ============================================================
# SECTION 2: MULTI-HEAD ATTENTION — NUMPY
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 2: Multi-Head Attention")
print("=" * 65)

print("""
  Multi-Head Attention: h farklı attention hesabını paralel çalıştır.
  Her kafa farklı ilişkileri öğrenir (sözdizimi, anlam, konum...).

  head_i  = Attention(Q·Wᵢᴿ, K·Wᵢᴷ, V·Wᵢᵛ)
  MHA(Q,K,V) = Concat(head₁,...,headₕ) · Wᴼ

  d_model = h × d_k  (boyut değişmez)
""")

class MultiHeadAttention:
    """Multi-Head Attention — NumPy from scratch."""

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        scale = 0.1
        # Her kafa için Q, K, V projeksiyon ağırlıkları
        self.WQ = np.random.randn(d_model, d_model) * scale
        self.WK = np.random.randn(d_model, d_model) * scale
        self.WV = np.random.randn(d_model, d_model) * scale
        self.WO = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """(seq_len, d_model) → (n_heads, seq_len, d_k)"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 0, 2)   # (n_heads, seq_len, d_k)

    def forward(self, Q: np.ndarray, K: np.ndarray,
                V: np.ndarray, mask: np.ndarray = None) -> tuple:
        # Projeksiyon
        Q_proj = Q @ self.WQ   # (seq_len, d_model)
        K_proj = K @ self.WK
        V_proj = V @ self.WV

        # Kafa bölümü
        Q_h = self.split_heads(Q_proj)  # (n_heads, seq_len, d_k)
        K_h = self.split_heads(K_proj)
        V_h = self.split_heads(V_proj)

        # Her kafa için attention
        head_outputs = []
        all_attn     = []
        for h in range(self.n_heads):
            out_h, attn_h = scaled_dot_product_attention(
                Q_h[h], K_h[h], V_h[h], mask)
            head_outputs.append(out_h)
            all_attn.append(attn_h)

        # Birleştir ve final projeksiyon
        concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
        output = concat @ self.WO

        return output, np.array(all_attn)  # (n_heads, seq_len, seq_len)

# Demo
d_model, n_heads = 16, 4
seq_len          = 6
mha  = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
X    = np.random.randn(seq_len, d_model)
out, all_attn = mha.forward(X, X, X)

print(f"  d_model={d_model}, n_heads={n_heads}, d_k={d_model//n_heads}")
print(f"  Giriş: {X.shape}")
print(f"  MHA çıkışı: {out.shape}")
print(f"  Attention ağırlıkları: {all_attn.shape}  (n_heads, seq_len, seq_len)")


# ============================================================
# SECTION 3: POZİSYONEL ENCODİNG
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 3: Positional Encoding")
print("=" * 65)

print("""
  Transformer'da kelime sırası bilgisi yoktur.
  Positional Encoding bu bilgiyi ekler.

  Sinüzoidal PE (Vaswani et al., 2017):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  Avantajlar:
    - Eğitim sırasında görülmemiş dizi uzunluklarına genelleşir
    - Her pozisyon benzersiz — model konumu ayırt edebilir
    - Göreceli konum: PE(pos+k), PE(pos)'un doğrusal dönüşümü
""")

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinüzoidal positional encoding matrisi hesapla."""
    PE  = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len).reshape(-1, 1)       # (seq_len, 1)
    i   = np.arange(0, d_model, 2)                # Çift indeksler
    div = np.exp(i * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(pos * div)   # Çift boyutlar
    PE[:, 1::2] = np.cos(pos * div)   # Tek boyutlar
    return PE

PE = positional_encoding(seq_len=20, d_model=32)
print(f"  PE boyutu: {PE.shape}  (seq_len=20, d_model=32)")
print(f"  İlk pozisyon PE: {PE[0, :4].round(3)}...")
print(f"  İkinci pozisyon PE: {PE[1, :4].round(3)}...")
print(f"  PE değerleri [-1, 1] arasında mı? "
      f"{PE.min():.3f} ≤ PE ≤ {PE.max():.3f}")


# ============================================================
# SECTION 4: TRANSFORMER ENCODER BLOĞU
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4: Transformer Encoder Bloğu")
print("=" * 65)

print("""
  Transformer Encoder Bloğu:

  x → [Multi-Head Self-Attention] → Add & Norm → [FFN] → Add & Norm → çıkış

  Add & Norm: Residual connection + Layer Normalization
  FFN: Feed-Forward Network (2 katman, genellikle 4× genişleme)
    FFN(x) = max(0, xW₁+b₁)W₂+b₂
""")

class FeedForward:
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff)   * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)   * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2

class LayerNorm:
    """Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta  = np.zeros(d_model)
        self.eps   = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu    = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True)
        return self.gamma * (x - mu) / (sigma + self.eps) + self.beta

class TransformerEncoderBlock:
    """
    Tek Transformer Encoder Bloğu:
    Self-Attention → Add&Norm → FFN → Add&Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.mha  = MultiHeadAttention(d_model, n_heads)
        self.ffn  = FeedForward(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)

    def forward(self, x: np.ndarray,
                mask: np.ndarray = None) -> tuple:
        # 1. Self-Attention + Residual + LayerNorm
        attn_out, attn_weights = self.mha.forward(x, x, x, mask)
        x = self.ln1.forward(x + attn_out)   # Add & Norm

        # 2. FFN + Residual + LayerNorm
        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)    # Add & Norm

        return x, attn_weights

# ── Encoder Demo ──────────────────────────────────────────────────────────────
d_model, n_heads, d_ff = 16, 4, 32
seq_len                = 6

encoder_block = TransformerEncoderBlock(d_model, n_heads, d_ff)
X_enc         = np.random.randn(seq_len, d_model)
out_enc, attn_enc = encoder_block.forward(X_enc)

print(f"  Encoder bloğu: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
print(f"  Giriş:  {X_enc.shape}")
print(f"  Çıkış:  {out_enc.shape}  (boyut değişmedi!)")
print(f"  Attention ağırlıkları: {attn_enc.shape}")

# 2 encoder bloğu üst üste
encoder2 = TransformerEncoderBlock(d_model, n_heads, d_ff)
out_enc2, _ = encoder2.forward(out_enc)
print(f"\n  2 Encoder bloğu sonrası: {out_enc2.shape}  ✓")


# ============================================================
# SECTION 5: BERT VE GPT ÖZET
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5: BERT ve GPT — Kavramsal Özet")
print("=" * 65)

print("""
  ┌─────────────┬──────────────────────────────────────────────┐
  │             │ BERT                    │ GPT                │
  ├─────────────┼─────────────────────────┼────────────────────┤
  │ Mimari      │ Transformer Encoder     │ Transformer Decoder│
  │ Yön         │ Çift yönlü (BiDir.)     │ Tek yönlü (Causal) │
  │ Pre-training│ Masked LM + NSP         │ Causal LM          │
  │ Hedef       │ Anlama (Understanding)  │ Üretim (Generation)│
  │ Fine-tuning │ Sınıflandırma, NER, QA  │ Tamamlama, Özet    │
  │ Tokenizer   │ WordPiece               │ BPE                │
  │ Versiyonlar │ BERT-base/large, RoBERTa│ GPT-2/3/4, ChatGPT │
  └─────────────┴─────────────────────────┴────────────────────┘

  BERT Masked LM:
    "The cat [MASK] on the mat" → predict: "sat"

  GPT Causal LM:
    "The cat sat" → predict next: "on"

  Her ikisi de Transformer tabanlı ama farklı maskeleme stratejisi!
""")


# ============================================================
# SECTION 6: PYTORCH MINI TRANSFORMER
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 6: PyTorch Mini Transformer")
print("=" * 65)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"\n  PyTorch {torch.__version__}")
    print("""
  class MiniTransformer(nn.Module):
      def __init__(self, vocab_size, d_model=64, n_heads=4,
                   n_layers=2, d_ff=128, max_seq=512):
          super().__init__()
          self.embed    = nn.Embedding(vocab_size, d_model)
          self.pos_enc  = nn.Embedding(max_seq, d_model)
          self.layers   = nn.ModuleList([
              nn.TransformerEncoderLayer(
                  d_model=d_model, nhead=n_heads,
                  dim_feedforward=d_ff, dropout=0.1,
                  batch_first=True
              ) for _ in range(n_layers)
          ])
          self.fc_out   = nn.Linear(d_model, vocab_size)

      def forward(self, x):
          # x: (batch, seq_len)
          seq_len = x.size(1)
          pos     = torch.arange(seq_len).unsqueeze(0).to(x.device)
          x       = self.embed(x) + self.pos_enc(pos)
          for layer in self.layers:
              x = layer(x)
          return self.fc_out(x)   # (batch, seq_len, vocab_size)

  # PyTorch built-in Transformer:
  encoder_layer = nn.TransformerEncoderLayer(
      d_model=512, nhead=8, dim_feedforward=2048,
      dropout=0.1, batch_first=True)
  transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    """)

    # Gerçek PyTorch kodu çalıştır
    class MiniTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=32, n_heads=4,
                      n_layers=2, d_ff=64, max_seq=50):
            super().__init__()
            self.embed   = nn.Embedding(vocab_size, d_model)
            self.pos_enc = nn.Embedding(max_seq, d_model)
            self.layers  = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=0.1,
                    batch_first=True, norm_first=True)
                for _ in range(n_layers)
            ])
            self.fc_out = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            pos = torch.arange(x.size(1)).unsqueeze(0)
            x   = self.embed(x) + self.pos_enc(pos)
            for layer in self.layers:
                x = layer(x)
            return self.fc_out(x)

    vocab_size = 50
    model      = MiniTransformer(vocab_size=vocab_size)
    x_pt       = torch.randint(0, vocab_size, (2, 10))  # batch=2, seq=10
    logits     = model(x_pt)
    total_p    = sum(p.numel() for p in model.parameters())
    print(f"\n  MiniTransformer — giriş: {tuple(x_pt.shape)}")
    print(f"  Çıkış logits: {tuple(logits.shape)}")
    print(f"  Toplam parametre: {total_p:,}")

except ImportError:
    print("  PyTorch yüklü değil.")


# ============================================================
# SECTION 7: GÖRSELLEŞTİRME
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Modül 09: Transformer ve Attention Analizi",
             fontsize=14, fontweight='bold')

# ── Panel 1: Attention Isı Haritası ──────────────────────────────────────────
ax1 = axes[0, 0]
words = ['the', 'cat', 'sat', 'on', 'mat', 'end']
np.random.seed(1)
Q2 = np.random.randn(6, 8); K2 = np.random.randn(6, 8); V2 = np.random.randn(6, 8)
_, attn_vis = scaled_dot_product_attention(Q2, K2, V2)
im1 = ax1.imshow(attn_vis, cmap='Blues', vmin=0, vmax=1)
ax1.set_xticks(range(6)); ax1.set_yticks(range(6))
ax1.set_xticklabels(words, rotation=45); ax1.set_yticklabels(words)
ax1.set_title("Attention Ağırlıkları Isı Haritası", fontweight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046)

# ── Panel 2: Positional Encoding ─────────────────────────────────────────────
ax2 = axes[0, 1]
PE_vis = positional_encoding(50, 64)
im2 = ax2.imshow(PE_vis.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
ax2.set_title("Positional Encoding (50 pozisyon, d=64)", fontweight='bold')
ax2.set_xlabel("Pozisyon"); ax2.set_ylabel("Encoding boyutu")
plt.colorbar(im2, ax=ax2, fraction=0.046)

# ── Panel 3: Multi-Head Attention (4 kafa) ────────────────────────────────────
ax3 = axes[1, 0]
# Her kafanın attention haritasını yan yana göster
all_attn_4head = attn_enc  # (4, 6, 6)
combined = np.zeros((6, 6*4 + 3*1))  # boşluklu
col = 0
for h in range(4):
    combined[:, col:col+6] = all_attn_4head[h]
    col += 6
    if h < 3:
        combined[:, col] = 0.5   # Ayraç
        col += 1
im3 = ax3.imshow(combined, cmap='Blues', vmin=0, vmax=1)
ax3.set_title("Multi-Head Attention (4 kafa, 6 token)", fontweight='bold')
ax3.set_xlabel("Token (her 6'lık blok = 1 kafa)")
ax3.set_ylabel("Token")
ax3.axvline(5.5, color='red', lw=1, alpha=0.7)
ax3.axvline(12.5, color='red', lw=1, alpha=0.7)
ax3.axvline(19.5, color='red', lw=1, alpha=0.7)

# ── Panel 4: Transformer vs RNN Karşılaştırması ────────────────────────────────
ax4 = axes[1, 1]
categories = ['Uzun Bağımlılık', 'Paralel Hesap', 'Parametre Sayısı',
              'Bellek (sekans)', 'Hız (eğitim)', 'Yorumlanabilirlik']
rnn_scores  = [2, 1, 4, 3, 2, 2]
trans_scores= [5, 5, 3, 2, 5, 5]
x_cat = np.arange(len(categories))
width = 0.35
ax4.bar(x_cat - width/2, rnn_scores,  width, color='#E65100', alpha=0.8, label='RNN/LSTM')
ax4.bar(x_cat + width/2, trans_scores, width, color='#1565C0', alpha=0.8, label='Transformer')
ax4.set_xticks(x_cat)
ax4.set_xticklabels(categories, rotation=25, ha='right', fontsize=8)
ax4.set_ylim(0, 6)
ax4.set_title("Transformer vs RNN/LSTM Karşılaştırma", fontweight='bold')
ax4.set_ylabel("Skor (1=düşük, 5=yüksek)")
ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/deep_learning_path/09-Transformer_ve_Attention/modul09_analiz.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik kaydedildi.")

print("\n" + "=" * 65)
print("  Modül 09 tamamlandı!")
print("=" * 65)
