# 🧠 Deep Learning Path

> Sıfırdan ileri seviyeye kapsamlı bir derin öğrenme yol haritası.  
> Her modül: **PDF konu anlatımı** + **Python implementasyonu** + **Jupyter Notebook**

---

## 📌 Hakkında

Bu repo, derin öğrenmeyi temelden öğrenmek isteyenler için hazırlanmış yapılandırılmış bir öğrenme yolu içerir. Her konu kendi klasöründe; teorik PDF, çalışabilir Python kodu ve kapsamlı Jupyter Notebook ile birlikte sunulmuştur.

Tüm implementasyonlar **üç framework** ile karşılaştırmalı olarak yapılmıştır:
- 🔢 **NumPy** — sıfırdan, matematiği anlamak için
- 🟠 **TensorFlow / Keras** — hızlı prototip ve production
- 🔴 **PyTorch** — araştırma ve özel mimari

---

## 📂 Repo Yapısı

```
Deep-Learning-Path/
│
├── 01-Sinir_Aglari_Temelleri/
│   ├── 01_Sinir_Aglari_Temelleri.pdf
│   ├── 01_sinir_aglari_numpy_from_scratch.py
│   └── 01_sinir_aglari_kapsamli.ipynb
│
├── 02-Aktivasyon_Fonksiyonlari/
│   ├── 02_Aktivasyon_Fonksiyonlari.pdf
│   ├── 02_aktivasyon_fonksiyonlari.py
│   └── 02_aktivasyon_fonksiyonlari_kapsamli.ipynb
│
├── 03-Kayip_Fonksiyonlari_ve_Optimizasyon/
├── 04-Geri_Yayilim_Backpropagation/
├── 05-Derin_Aglar_Regularization/
├── 06-CNN_Evrisimsel_Sinir_Aglari/
├── 07-Transfer_Learning/
├── 08-RNN_LSTM_GRU/
├── 09-Transformer_ve_Attention/
├── 10-Generative_Models_GAN_VAE/
│
└── 11-FINAL_PROJE_Multimodal_Sentiment/
    ├── multimodal_sentiment_analysis.py
    ├── multimodal_sentiment_notebook.ipynb
    └── FINAL_PROJE_RAPORU.pdf
```

---

## 🗺️ Modül Listesi

| # | Modül | Konular |
|---|-------|---------|
| 01 | Sinir Ağları Temelleri | Perceptron, MLP, Forward Pass, Backprop, XOR |
| 02 | Aktivasyon Fonksiyonları | Sigmoid, ReLU, GELU, Vanishing Gradient, Dead Neuron |
| 03 | Kayıp Fonk. & Optimizasyon | MSE, BCE, Adam, SGD, LR Scheduler |
| 04 | Backpropagation | Zincir Kuralı, Hesaplama Grafı, Sıfırdan Impl. |
| 05 | Regularization | Dropout, Batch Norm, L1/L2, Early Stopping |
| 06 | CNN | Konvolüsyon, Pooling, ResNet, Grad-CAM |
| 07 | Transfer Learning | Feature Extraction, Fine-tuning, VGG/ResNet |
| 08 | RNN / LSTM / GRU | Zaman Serisi, BPTT, Seq2Seq |
| 09 | Transformer & Attention | Self-Attention, BERT, GPT, Positional Encoding |
| 10 | GAN & VAE | Üretici Modeller, Latent Space, DCGAN |
| 11 | Final Proje | Multimodal Sentiment Analysis (CNN + BERT fusion) |

---

## 🚀 Kurulum

```bash
git clone https://github.com/AlicanKaya192/Deep-Learning-Path.git
cd Deep-Learning-Path

pip install numpy matplotlib scikit-learn tensorflow torch torchvision jupyter
```

---

## 📋 Her Modülde Ne Var?

### 📄 PDF
- Konu anlatımı (teorik temel, formüller)
- Görsel şemalar ve mimari diyagramlar
- Adım adım sayısal örnekler
- Framework karşılaştırma tabloları
- Yaygın hatalar ve çözümleri
- Özet tablo + kaynaklar

### 🐍 Python (.py)
- NumPy ile sıfırdan implementasyon
- TensorFlow/Keras implementasyonu
- PyTorch implementasyonu
- Matplotlib görselleştirme
- PEP8 uyumlu, tam yorumlanmış

### 📓 Jupyter Notebook (.ipynb)
- LaTeX matematiksel formüller
- İnteraktif görselleştirmeler
- Hiperparametre deneyleri
- Gerçek veri seti uygulaması
- Alıştırmalar (ipuçlu, çözümsüz)

---

## 🎯 Kimler İçin?

- Derin öğrenmeye sıfırdan başlayanlar
- Matematiği anlayarak ilerlemek isteyenler
- Framework'leri karşılaştırmalı öğrenmek isteyenler
- Bilgisayar Mühendisliği / Veri Bilimi öğrencileri

---

## 📊 İlerleme

![%20](https://img.shields.io/badge/İlerleme-2%2F11%20Modül-blue)

- [x] Modül 01 — Sinir Ağları Temelleri
- [x] Modül 02 — Aktivasyon Fonksiyonları
- [ ] Modül 03 — Kayıp Fonksiyonları & Optimizasyon
- [ ] Modül 04 — Backpropagation
- [ ] Modül 05 — Regularization
- [ ] Modül 06 — CNN
- [ ] Modül 07 — Transfer Learning
- [ ] Modül 08 — RNN / LSTM / GRU
- [ ] Modül 09 — Transformer & Attention
- [ ] Modül 10 — GAN & VAE
- [ ] Modül 11 — Final Proje

---

## 🔗 İlgili Repo

[📦 Data-Science-RoadMap](https://github.com/AlicanKaya192/Data-Science-RoadMap) — Tam veri bilimi yol haritası

---

## 📜 Lisans

MIT License — özgürce kullanabilir, katkıda bulunabilirsiniz.
