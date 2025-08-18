 Drug Labels and Side Effects Prediction

## 📌 Project Overview
This project uses the **Drug Labels and Side Effects Dataset** (Kaggle, ~1,436 records) to predict **multi‑label side effects** from structured (tabular) drug information.

The dataset includes:
- **Drug details** (name, class, manufacturer, approval year, dosage, price)
- **Regulatory info** (approval status, expiry date)
- **Usage constraints** (administration route, side effect severity)
- **Safety and warnings** (warnings, contraindications)
- **Target variable**: `side_effects` (435 unique effects, multi-label format)

Our pipeline is designed as a clean, step‑by‑step workflow: data loading, EDA, preprocessing, feature engineering, modeling, and evaluation.

---

## 🛠️ Pipeline Steps

### Step 0 — Load & Inspect
- Loaded CSV and checked dataset shape, columns, and data types
- Verified no missing values per column
- High uniqueness in drug_name (~1337 unique) and side_effects (~435 unique, comma-separated)

### Step 1 — Exploratory Data Analysis (EDA)
- Balanced target distributions:
  - **Approval Status**: Approved, Pending, Rejected (~equal counts)
  - **Severity**: Mild, Moderate, Severe (~equal counts)
- Price range largely similar across major drug classes
- Short `indications` text (15–21 chars) — noted for NLP enrichment
- Top side effects: blurred vision, dry mouth, headache, rash, diarrhea

### Step 2 — Preprocessing
- **Text Cleaning**: lowercase, punctuation removal, whitespace normalization
- **Text Merge**: combined `indications`, `contraindications`, and `warnings` into `combined_text`
- **Multi‑Label Encoding**: split `side_effects` by comma → binarized with `MultiLabelBinarizer`
- **Categorical Encoding**: `drug_class`, `administration_route`, `approval_status`, `side_effect_severity`, `manufacturer` → `LabelEncoder`
- **Date Feature Engineering**: computed `days_until_expiry` from `expiry_date`

### Step 3 — Tabular Feature Engineering & Baseline Models
_(NLP skipped for now)_
- Features: numeric (approval_year, dosage_mg, price_usd, days_until_expiry) + encoded categoricals
- Multi‑model baseline with `MultiOutputClassifier`:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - K‑Nearest Neighbors
- Metrics:
  - Macro F1
  - Micro F1
  - Hamming Loss

---

## 📊 Example Step 3 Results (Tabular Only)

| Model               | Macro F1 | Micro F1 | Hamming Loss |
|---------------------|----------|----------|--------------|
| Logistic Regression | 0.0000   | 0.0000   | 0.1986       |
| Random Forest       | 0.0723   | 0.0718   | 0.2243       |
| KNN                 | 0.1036   | 0.1056   | 0.2236       |

_(Gradient Boosting pending in test bench)_

---

## 🚀 Planned Next Steps
- Apply probability threshold tuning for multi‑label recall improvement
- Add Gradient Boosting, XGBoost, LightGBM to baseline comparison
- Optionally reintroduce NLP features (`combined_text`) using TF‑IDF or transformer embeddings (DistilBERT)
- Explore hybrid tabular + NLP models

---

## 📂 Repository Structure
