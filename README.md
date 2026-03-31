# 🩺 AI Disease Prediction System

Ever wondered if you could build something that guesses what disease you might have just from your symptoms? That's exactly what this project does. It's a command-line tool that takes in symptoms you type and predicts the most likely disease using a Decision Tree model trained on real medical data.

It's not perfect (and definitely not a replacement for a doctor), but it was a great way to learn how ML pipelines actually work end-to-end.

---

## What's inside

- **train.py** — the training script. Feed it the dataset and it spits out a trained model + all the supporting files.
- **predict.py** — the fun part. Run this, type your symptoms, and see what it thinks you might have.
- **archive.zip** — the dataset ZIP (put this here before training)
---

## What it can do

- **Predicts diseases** from symptoms using a Decision Tree that hits ~97.6% accuracy on the test set
- **Handles typos** — type `headach` instead of `headache`? No problem, fuzzy matching sorts it out
- **Shows top 3 predictions** with confidence percentages, not just one guess
- **Describes the disease** so you actually know what it is
- **Suggests precautions** — practical stuff you can do
- **Rates severity** of your symptoms (Low / Medium / High) based on weighted scores
- **Auto-extracts the dataset** from the ZIP file so you don't have to manually unzip anything

---

## Project layout

```
ai ml project/
│
├── train.py                  # Training script
├── predict.py                # Prediction tool
├── archive.zip               # Dataset ZIP (put this here before training)
│
├── model.pkl                 # Trained model (auto-generated)
├── symptoms.pkl              # Symptom list (auto-generated)
├── disease_description.pkl   # Descriptions (auto-generated)
├── disease_precaution.pkl    # Precautions (auto-generated)
├── symptom_severity.pkl      # Severity weights (auto-generated)
│
└── README.md                 # You're reading it
```

---

## About the dataset

The data comes from [this Kaggle dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset). It's bundled as `archive.zip` and contains:

| File | What's in it |
|---|---|
| `Training.csv` | 306 rows, 134 symptom columns, plus the disease label |
| `Testing.csv` | 42 rows, same format |
| `disease_description.csv` | Short descriptions for each disease |
| `disease_precaution.csv` | Up to 4 precautions per disease |
| `symptom_severity.csv` | A severity score (1–7) for each symptom |

There are **42 disease classes** (everything from common stuff like Acne to serious things like AIDS and Dengue) and **134 binary symptom features** (1 = you have it, 0 = you don't).

---

## Getting started

You'll need **Python 3.10+** and pip.

Install the dependencies:

```bash
pip install pandas scikit-learn thefuzz
```

Then:

**1. Drop `archive.zip` into the project folder** (if it's not already there).

**2. Train the model:**

```bash
python3 train.py
```

This extracts the CSVs, trains the Decision Tree, evaluates it, and saves everything as `.pkl` files.

**3. Run the predictor:**

```bash
python3 predict.py
```

Type your symptoms separated by commas when it asks:

```
Enter symptoms (comma separated): itching, skin rash, nodal skin eruptions
```

That's it. You'll get predictions, descriptions, precautions, the whole deal.

---

## How it works under the hood

### Training (`train.py`)

Straightforward pipeline:

1. Finds and unzips `archive.zip`
2. Loads `Training.csv` and `Testing.csv`, fills any missing values
3. Splits features (symptoms) from labels (disease names)
4. Trains a `DecisionTreeClassifier` on the training set
5. Tests accuracy on the held-out test set
6. Dumps the model + supporting data as pickle files

### Prediction (`predict.py`)

This is where the magic happens:

1. You type in symptoms as free text
2. Each symptom gets fuzzy-matched against the known symptom list:
   - **90%+ match** → auto-corrects silently
   - **70–89% match** → auto-corrects but tells you
   - **50–69% match** → shows you the closest options and lets you pick
   - **Below 50%** → skips it (probably not a real symptom)
3. Builds a binary feature vector (134 slots, 1s where your symptoms match)
4. Runs `predict_proba()` to get confidence scores for all 42 diseases
5. Shows the top 3 predictions, plus description and precautions for the #1 pick
6. Calculates an overall severity rating based on your symptom weights

---

## Model performance

| | |
|---|---|
| Algorithm | Decision Tree Classifier |
| Training data | 306 samples |
| Test data | 42 samples |
| Classes | 42 diseases |
| Features | 134 binary symptoms |
| **Accuracy** | **97.62%** |

The accuracy is high partly because it's a clean, structured dataset where each disease has a fairly distinct symptom pattern. Real-world medical data would be messier and harder to classify.

---

## Example output

Here's what a typical run looks like:

```
Enter symptoms (comma separated): itching, skin rash, nodal skin eruptionss

  🔍 Auto-corrected  'nodal_skin_eruptionss'  →  'nodal_skin_eruptions'

⚡ Symptom Severity: 🟢 Low (avg score: 2.7/7)

══════════════════════════════════════════════════
🩺  TOP PREDICTIONS
══════════════════════════════════════════════════

🥇  Fungal infection  (100.0% confidence)

📖 Description:
   In humans, fungal infections occur when an invading fungus takes over an
   area of the body and is too much for the immune system to handle...

🛡️  Precautions:
   1. Bath twice
   2. Use detol or neem in bathing water
   3. Keep the infected area dry
   4. Use clean cloths

══════════════════════════════════════════════════
⚠️  This tool is for educational purposes only.
   Always consult a qualified doctor for medical advice.
══════════════════════════════════════════════════
```
---

## Known limitations

Let's be real about what this can't do:

- **Small dataset** — 306 training samples is tiny. A production system would need thousands.
- **Overlapping symptoms** — some diseases share similar symptoms, so predictions can get confused.
- **Static data** — new diseases or newly discovered symptoms aren't covered unless you retrain.
- **Not medical advice** — seriously, don't use this to diagnose yourself. Go see a doctor.

---

## Ideas for the future

Some things I'd like to add eventually:

- A proper web interface (Flask + React maybe)
- A REST API with a `/predict` endpoint
- Try other models — Random Forest, SVM, XGBoost — and compare
- Live autocomplete for symptoms as you type
- Pull in larger, real-world clinical datasets
- Support for entering symptoms in different languages

---

## Author

Ankesh Raj
