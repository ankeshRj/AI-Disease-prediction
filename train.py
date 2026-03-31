import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import zipfile
import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import numpy as np


zip_files=[f for f in os.listdir(".") if f.endswith(".zip")]
if not zip_files:
    print(" No ZIP file found in project folder. Please add your dataset ZIP here.")
    exit()

if len(zip_files) > 1:
    print(f" Multiple ZIPs found, using: {zip_files[0]}")

zip_path=zip_files[0]
print(f"Found ZIP: {zip_path}")

extract_folder="temp_dataset"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    os.makedirs(extract_folder, exist_ok=True)
    zip_ref.extractall(extract_folder)


all_files=os.listdir(extract_folder)
train_csv=next((f for f in all_files if "training" in f.lower() and f.endswith(".csv")), None)
test_csv=next((f for f in all_files if "testing" in f.lower() and f.endswith(".csv")), None)

if train_csv and test_csv:
    print(f" Using pre-split files: {train_csv} + {test_csv}")
    train_df=pd.read_csv(os.path.join(extract_folder, train_csv)).fillna(0)
    test_df=pd.read_csv(os.path.join(extract_folder, test_csv)).fillna(0)

    X_train=train_df.drop("prognosis", axis=1)
    y_train=train_df["prognosis"]
    X_test=test_df.drop("prognosis", axis=1)
    y_test=test_df["prognosis"]
else:
    from sklearn.model_selection import train_test_split
    csv_files=[f for f in all_files if f.endswith(".csv")]
    if not csv_files:
        print("No CSV file found in ZIP!")
        shutil.rmtree(extract_folder)
        exit()
    df=pd.read_csv(os.path.join(extract_folder, csv_files[0])).fillna(0)
    X=df.drop("prognosis", axis=1)
    y=df["prognosis"]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


model=DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")

# ── Generate Confusion Matrix ─────────────────────────────────────────────────
print("\n📊 Generating confusion matrix...")
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(20, 18))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Count', fontsize=12)

# Add text annotations
thresh = cm.max() / 2.0
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black', fontsize=6)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted Disease', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Disease', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix — Overall Accuracy: {accuracy*100:.2f}%',
             fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("✅ Saved: confusion_matrix.png")

# ── Generate Per-Disease Accuracy Bar Chart ───────────────────────────────────
print("📊 Generating per-disease accuracy chart...")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Extract per-disease f1-scores (skip avg rows)
disease_names = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
f1_scores     = [report[d]['f1-score'] for d in disease_names]
precisions    = [report[d]['precision'] for d in disease_names]
recalls       = [report[d]['recall'] for d in disease_names]

# Sort by f1-score
sorted_idx    = np.argsort(f1_scores)
disease_names = [disease_names[i] for i in sorted_idx]
f1_scores     = [f1_scores[i] for i in sorted_idx]
precisions    = [precisions[i] for i in sorted_idx]
recalls       = [recalls[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(14, max(8, len(disease_names) * 0.3)))
y_pos = np.arange(len(disease_names))
bar_height = 0.25

bars1 = ax.barh(y_pos - bar_height, precisions, bar_height, label='Precision', color='#3498db', alpha=0.85)
bars2 = ax.barh(y_pos, f1_scores, bar_height, label='F1-Score', color='#2ecc71', alpha=0.85)
bars3 = ax.barh(y_pos + bar_height, recalls, bar_height, label='Recall', color='#e74c3c', alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(disease_names, fontsize=7)
ax.set_xlabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Per-Disease Classification Metrics\nOverall Accuracy: {accuracy*100:.2f}%',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 1.05)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('classification_report.png', dpi=150)
plt.close()
print("✅ Saved: classification_report.png")

# Print classification report to console
print("\n" + "═" * 60)
print("📋 CLASSIFICATION REPORT")
print("═" * 60)
print(classification_report(y_test, y_pred, zero_division=0))
print("═" * 60)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("symptoms.pkl", "wb") as f:
    pickle.dump(list(X_train.columns), f)


desc_file=next((f for f in all_files if "description" in f.lower()), None)
if desc_file:
    desc_df=pd.read_csv(os.path.join(extract_folder, desc_file))
    desc_dict=dict(zip(desc_df["Disease"], desc_df["Symptom_Description"]))
    with open("disease_description.pkl", "wb") as f:
        pickle.dump(desc_dict, f)
    print(f" Saved descriptions for {len(desc_dict)} diseases")


prec_file=next((f for f in all_files if "precaution" in f.lower()), None)
if prec_file:
    prec_df=pd.read_csv(os.path.join(extract_folder, prec_file))
    prec_dict={}
    for _, row in prec_df.iterrows():
        prec_dict[row["Disease"]] = [
            str(row[c]) for c in prec_df.columns[1:]
            if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
        ]
    with open("disease_precaution.pkl", "wb") as f:
        pickle.dump(prec_dict, f)
    print(f"Saved precautions for {len(prec_dict)} diseases")


sev_file=next((f for f in all_files if "severity" in f.lower()), None)
if sev_file:
    sev_df=pd.read_csv(os.path.join(extract_folder, sev_file))
    sev_dict=dict(zip(sev_df["Symptom"], sev_df["Symptom_severity"]))
    with open("symptom_severity.pkl", "wb") as f:
        pickle.dump(sev_dict, f)
    print(f" Saved severity weights for {len(sev_dict)} symptoms")


shutil.rmtree(extract_folder)
print(" Cleaned up temp files")
print(" Model trained and saved successfully!")
