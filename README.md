# 🔬 NutritionNet — AI-Powered Nail & Skin Health Scanner

> **Non-invasive nutritional deficiency detection from nail and skin photographs using dual deep learning models, with personalised 7-day Indian diet plans and downloadable PDF reports.**

---

## 🧠 Overview

NutritionNet is a deep learning-powered web application that detects nutritional deficiencies from close-up photographs of fingernails and skin. It runs **two AI models simultaneously**, compares their confidence scores through a calibrated arbitration mechanism, and delivers:

- An instant deficiency diagnosis with confidence score and severity rating
- A personalised **7-day Indian diet plan** (vegetarian and non-vegetarian)
- A downloadable **PDF health report** — no server-side data storage

Built as a final year M.Sc. Data Science project at **CHRIST (Deemed to be University), Bengaluru**.

---

## 🎥 Demo

| Upload | Analysis | Diet Plan |
|--------|----------|-----------|
| Drag & drop nail or skin image | Instant AI diagnosis with severity gauge | 7-day Indian meal plan personalised to deficiency |

---

## ✨ Features

- **Dual-model parallel inference** — both models run on every image; no pre-selection required from the user
- **Smart routing** — confidence arbitration with asymmetric nail penalty (−20%) prevents false positives
- **8 conditions detected** — 3 nail conditions + 5 vitamin/mineral deficiency classes
- **Severity scoring** — each prediction mapped to a 0–100 clinical severity scale
- **Confidence breakdown** — top 2 class probabilities shown as visual bars
- **Food grid** — 6 nutrient-specific food recommendations per condition
- **7-day Indian diet plan** — affordable, locally available foods with veg/non-veg toggle
- **Automatic veg swaps** — chicken/fish/eggs replaced with paneer, rajma, soya, tofu
- **PDF report generation** — ReportLab A4 document with image, diagnosis, diet, and tips
- **No data retention** — images and results are never stored server-side
- **Medical disclaimer** — prominently displayed on every result

---

## ⚙️ How It Works

```
User uploads image
        ↓
Both models run in parallel
        ↓
 ┌─────────────────┐     ┌──────────────────┐
 │  Nail Model     │     │  Skin Model      │
 │  MobileNetV3    │     │  EfficientNet-B4 │
 │  224×224 · TF   │     │  380×380 · PT    │
 └────────┬────────┘     └────────┬─────────┘
          └──────────┬────────────┘
                     ↓
         Confidence Arbitration
         (nail_conf − 20% penalty)
                     ↓
         Winner selected or "unknown"
                     ↓
      Severity Score + Condition Metadata
                     ↓
     ┌───────────────┴───────────────┐
     ↓                               ↓
Result Card + Diet Plan          PDF Report
```

**Confidence threshold:** Both models must score above 40% individually. The nail model wins only if its confidence exceeds the skin model's by 20% or more.

---

## 🤖 Model Architecture

### Nail Model — MobileNetV3Large (TensorFlow / Keras)

| Property | Value |
|----------|-------|
| Architecture | MobileNetV3Large |
| Pretrained on | ImageNet |
| Input size | 224 × 224 × 3 |
| Classes | healthy · onychomycosis · psoriasis |
| Stage 1 | Head only · Adam LR=1e-3 · 15 epochs |
| Stage 2 | Top 30 layers · LR=1e-5 · 30 epochs |
| Best Val Accuracy | ~91% |
| Output file | `best_model.keras` |

**Custom head:** `Dropout(0.4) → Dense(256, ReLU, L2=1e-4) → Dropout(0.3) → Dense(3, Softmax)`

---

### Skin Model — EfficientNet-B4 (PyTorch)

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B4 |
| Pretrained on | ImageNet |
| Input size | 380 × 380 × 3 |
| Classes | Vit A · Vit B-12 · Vit C · Vit D · Zinc/Iron/Biotin |
| Frozen phase | Epochs 1–15 · Head LR=5e-5 |
| Gradual unfreeze | Epoch 16+ · Body LR=5e-6 |
| Schedule | Warmup (3ep) + Cosine Annealing |
| Techniques | AMP float16 · MixUp(α=0.3) · GradClip(0.5) · AdamW |
| Best Val Accuracy | ~79% |
| Output file | `best_vitamin_model.pth` |

**Custom head:** `LayerNorm → Dropout(0.3) → Linear(512, GELU) → LayerNorm → Dropout(0.15) → Linear(5)`

---

## 📁 Project Structure

```
NutritionNet/
│
├── app.py                      # Main Streamlit application
│
├── models/
│   ├── best_model.keras         # Trained nail model weights (TF/Keras)
│   └── best_vitamin_model.pth   # Trained skin model weights (PyTorch)
│
├── notebooks/
│   ├── Nail_model.ipynb         # Nail model training notebook
│   └── Skin_model.ipynb         # Skin model training notebook
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** Model weight files are not included in this repository due to size. See [Installation](#-installation) for setup instructions.

---

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/NutritionNet.git
cd NutritionNet
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model weights

Place your trained model files in the `models/` directory:

```
models/
├── best_model.keras
└── best_vitamin_model.pth
```

Then update the paths in `app1.py`:

```python
NAIL_MODEL_PATH = "models/best_model.keras"
SKIN_MODEL_PATH = "models/best_vitamin_model.pth"
```

### 5. Run the app

```bash
streamlit run app1.py
```

Open your browser at `http://localhost:8501`

---

## 🚀 Usage

1. Open the app in your browser
2. Click **Browse files** or drag and drop a nail or skin photograph
3. Press **Analyse now →**
4. View your diagnosis, severity rating, confidence breakdown, and food recommendations
5. Select **Non-Veg** or **Vegetarian** to get your personalised diet plan
6. Download your **PDF health report**

### Supported image formats
`JPG · JPEG · PNG · BMP · WEBP`

### Tips for best results
- Use a **close-up** photograph — the affected area should fill most of the frame
- Ensure **good lighting** — natural daylight or bright indoor light
- Keep the camera **steady** — blurry images reduce accuracy

---

## 📊 Dataset

### Nail Disease Dataset
- **Total images:** 1,465
- **Classes:** healthy (~480) · onychomycosis (~680) · psoriasis (~305)
- **Split:** 80% train / 20% validation + held-out test set
- **Source:** Nail Disease Image Classification Dataset (Kaggle)

### Skin Vitamin Deficiency Dataset
- **Total images:** 13,077
- **Classes:** Vitamin A · B-12 · C · D · Zinc/Iron/Biotin
- **Split:** 80% train / 20% validation (StratifiedShuffleSplit, seed=42)
- **Source:** Skin Diseases DermNet Dataset

---

## 📈 Results

| Model | Architecture | Val Accuracy | Epochs |
|-------|-------------|-------------|--------|
| Nail | MobileNetV3Large | ~89% (peak ~91%) | 23 |
| Skin | EfficientNet-B4 | ~79% | 50 |

**Key observations:**
- Nail model validation accuracy consistently exceeded training accuracy — no overfitting
- Skin model training and validation curves converged within 1% at the final epoch
- Most significant skin model gains occurred between epochs 15–30 (gradual unfreeze phase)

---

## 🧰 Tech Stack

| Category | Technology |
|----------|------------|
| Web Framework | Streamlit |
| Deep Learning (Nail) | TensorFlow 2.16.1 + Keras |
| Deep Learning (Skin) | PyTorch 2.4+ |
| Image Processing | Pillow · OpenCV |
| PDF Generation | ReportLab |
| Data / ML | NumPy · scikit-learn |
| Training Platform | Google Colab (T4 GPU) |
| Local Development | MacBook Air M4 (Apple MPS) |

---

## ⚕️ Disclaimer

> NutritionNet is for **educational and informational purposes only**. It does not constitute medical advice, diagnosis, or treatment. AI confidence scores are not clinical accuracy measures. Always consult a qualified dermatologist or doctor for any health concern. No user images or data are stored by this application.

---

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ for accessible nutritional health screening in India</p>
