# Vision Transformer Model Comparison

Proyek ini adalah implementasi perbandingan model Vision Transformer untuk tugas Deep Learning - Eksplorasi Vision Transformer.

## 📋 Deskripsi

Program ini membandingkan 3 model Vision Transformer:
- **ViT (Vision Transformer Base)** - Transformer murni untuk visi
- **Swin Transformer Tiny** - Hierarchical transformer dengan shifted window attention
- **DeiT (Data-efficient Image Transformer Small)** - ViT dengan knowledge distillation

## 🎯 Fitur

✅ **Training dengan Transfer Learning** - Fine-tuning dari pre-trained weights  
✅ **Metrik Lengkap** - Accuracy, Precision, Recall, F1-Score  
✅ **Visualisasi Komprehensif** - Learning curves, confusion matrix, comparison plots  
✅ **Analisis Parameter** - Jumlah parameter dan ukuran model  
✅ **Pengukuran Inference Time** - Waktu inferensi dan throughput  
✅ **Comparison Tables** - Tabel perbandingan dalam format CSV  

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (opsional, untuk GPU acceleration)

## 🚀 Instalasi

### 1. Clone atau Download Repository

```bash
git clone https://github.com/username/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Opsional) Gunakan Virtual Environment

```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi (Windows)
venv\Scripts\activate

# Aktivasi (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📂 Struktur Dataset

Program ini mendukung dataset dengan struktur folder berikut:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── classN/
│       └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── classN/
└── test/
    ├── class1/
    ├── class2/
    └── classN/
```

### Dataset yang Dapat Digunakan:

1. **Indonesian Food Dataset** (dari tugas sebelumnya)
2. **CIFAR-10** (tersedia di torchvision)
3. **CIFAR-100**
4. **Food-101**
5. Dataset custom Anda sendiri (harus mengikuti struktur di atas)

## 🎮 Cara Menggunakan

### 1. Persiapan Dataset

Edit konfigurasi di notebook pada bagian **Configuration**:

```python
CONFIG = {
    'data_dir': 'path/to/your/dataset',  # GANTI dengan path dataset Anda
    'img_size': 224,
    'batch_size': 32,
    'num_epochs': 20,
    # ...
}
```

### 2. Jalankan Notebook

#### Opsi A: Jupyter Notebook (Lokal)

```bash
jupyter notebook vision_transformer_comparison.ipynb
```

#### Opsi B: Google Colab

1. Upload notebook ke Google Drive
2. Buka dengan Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Upload dataset atau mount Google Drive

#### Opsi C: Kaggle Notebooks

1. Upload notebook ke Kaggle
2. Enable GPU/TPU di settings
3. Add dataset dari Kaggle Datasets atau upload sendiri

### 3. Eksekusi Cell-by-Cell

Jalankan cell secara berurutan:

1. ✅ Setup dan Import Libraries
2. ✅ Configuration
3. ✅ Data Preparation
4. ✅ Model Definition
5. ✅ Training (ini akan memakan waktu terlama)
6. ✅ Evaluation
7. ✅ Visualizations
8. ✅ Comparative Analysis

### 4. Hasil akan Tersimpan di Folder `figure/`

```
figure/
├── ViT-Base_training_history.png
├── ViT-Base_confusion_matrix.png
├── ViT-Base_per_class_metrics.png
├── Swin-Tiny_training_history.png
├── Swin-Tiny_confusion_matrix.png
├── Swin-Tiny_per_class_metrics.png
├── DeiT-Small_training_history.png
├── DeiT-Small_confusion_matrix.png
├── DeiT-Small_per_class_metrics.png
├── model_comparison_overview.png
├── parameter_comparison.csv
├── performance_comparison.csv
├── inference_comparison.csv
└── summary_report.txt
```

## ⚙️ Konfigurasi

### Mengubah Model yang Dibandingkan

Edit bagian `models` di CONFIG:

```python
'models': {
    'ViT-Base': 'vit_base_patch16_224',
    'Swin-Tiny': 'swin_tiny_patch4_window7_224',
    'DeiT-Small': 'deit_small_patch16_224'
}
```

Model lain yang tersedia dari `timm`:
- `'vit_small_patch16_224'` - ViT Small
- `'vit_large_patch16_224'` - ViT Large
- `'swin_small_patch4_window7_224'` - Swin Small
- `'deit_base_patch16_224'` - DeiT Base
- Dan banyak lagi...

### Hyperparameter Tuning

```python
CONFIG = {
    'num_epochs': 20,           # Jumlah epoch training
    'learning_rate': 1e-4,      # Learning rate
    'batch_size': 32,           # Batch size
    'weight_decay': 1e-4,       # Weight decay untuk regularization
    'patience': 5,              # Early stopping patience
}
```

## 📊 Output dan Metrik

Program ini akan menghasilkan:

### 1. Parameter Information
- Total parameters
- Trainable parameters
- Non-trainable parameters
- Model size (MB)

### 2. Performance Metrics
- Accuracy
- Precision (macro & per-class)
- Recall (macro & per-class)
- F1-Score (macro & per-class)
- Confusion Matrix

### 3. Inference Time
- Average time per image (ms)
- Standard deviation
- Throughput (images/sec)

### 4. Visualizations
- Training/Validation Loss curves
- Training/Validation Accuracy curves
- Confusion Matrix heatmap
- Per-class metrics bar chart
- Model comparison charts

### 5. Comparison Tables
- Parameter comparison (CSV)
- Performance comparison (CSV)
- Inference time comparison (CSV)

## 💡 Tips dan Troubleshooting

### Out of Memory (OOM) Error

Jika terjadi OOM error saat training:

1. **Kurangi batch size**:
   ```python
   'batch_size': 16,  # atau 8
   ```

2. **Gunakan model yang lebih kecil**:
   ```python
   'models': {
       'ViT-Tiny': 'vit_tiny_patch16_224',
       'Swin-Tiny': 'swin_tiny_patch4_window7_224',
   }
   ```

3. **Gunakan mixed precision training** (tambahkan di training loop):
   ```python
   scaler = torch.cuda.amp.GradScaler()
   ```

### Training Terlalu Lama

1. **Kurangi jumlah epoch**:
   ```python
   'num_epochs': 10,
   ```

2. **Gunakan dataset lebih kecil**

3. **Gunakan GPU** (Google Colab gratis atau Kaggle)

### Dataset Loading Error

Pastikan struktur folder dataset benar:

```python
# Cek struktur
import os
print(os.listdir('path/to/dataset/train'))  # Harus menampilkan nama-nama class
```

## 🎓 Untuk Laporan

Gunakan hasil-hasil berikut untuk laporan Anda:

1. **Tabel Parameter** → `figure/parameter_comparison.csv`
2. **Tabel Performa** → `figure/performance_comparison.csv`
3. **Tabel Inference Time** → `figure/inference_comparison.csv`
4. **Semua Visualisasi** → folder `figure/`
5. **Summary Report** → `figure/summary_report.txt`
6. **Model Weights** → `*_model.pth` files

## 📚 Referensi

1. **Vision Transformer (ViT)**:
   - Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   - https://arxiv.org/abs/2010.11929

2. **Swin Transformer**:
   - Liu et al. (2021) "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
   - https://arxiv.org/abs/2103.14030

3. **DeiT**:
   - Touvron et al. (2021) "Training data-efficient image transformers & distillation through attention"
   - https://arxiv.org/abs/2012.12877

4. **TIMM Library**:
   - https://github.com/huggingface/pytorch-image-models

## 🤝 Kontribusi

Jika menemukan bug atau ingin menambahkan fitur:

1. Fork repository
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Lisensi

Project ini dibuat untuk keperluan akademik - Tugas Deep Learning ITERA.

## 👤 Author

**[Abraham Ganda Napitu]**  
NIM: [122140095]  
Prodi Teknik Informatika - ITERA  
Mata Kuliah: Deep Learning