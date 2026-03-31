import pickle
import os
import sys
import warnings
import pandas as pd
from thefuzz import process, fuzz

# Suppress all warnings
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

# ── Load required files ───────────────────────────────────────────────────────
for fname in ["model.pkl", "symptoms.pkl"]:
    if not os.path.exists(fname):
        print(f"❌ {fname} not found. Please run train.py first!")
        exit()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("symptoms.pkl", "rb") as f:
    symptoms_list = pickle.load(f)

# Optional enrichment files
desc_dict = pickle.load(open("disease_description.pkl", "rb")) if os.path.exists("disease_description.pkl") else {}
prec_dict = pickle.load(open("disease_precaution.pkl", "rb")) if os.path.exists("disease_precaution.pkl") else {}
sev_dict  = pickle.load(open("symptom_severity.pkl", "rb")) if os.path.exists("symptom_severity.pkl") else {}

symptoms_lower = [s.lower() for s in symptoms_list]

# ── Show available symptoms ───────────────────────────────────────────────────
print("\n📋 Available symptoms:")
for s in symptoms_list:
    print("  -", s)


# ── Fuzzy symptom resolver ────────────────────────────────────────────────────
def resolve_symptom(raw: str) -> str | None:
    raw = raw.strip().lower().replace(" ", "_")

    if raw in symptoms_lower:
        return symptoms_list[symptoms_lower.index(raw)]

    results = process.extract(raw, symptoms_lower, scorer=fuzz.token_sort_ratio, limit=3)
    best_match, best_score = results[0][0], results[0][1]

    if best_score >= 90:
        print(f"  🔍 Auto-corrected '{raw}' → '{best_match}'")
        return symptoms_list[symptoms_lower.index(best_match)]

    elif best_score >= 70:
        print(f"  ✏️  Fuzzy matched '{raw}' → '{best_match}' (score: {best_score})")
        return symptoms_list[symptoms_lower.index(best_match)]

    elif best_score >= 50:
        print(f"\n  ❓ '{raw}' not found. Did you mean:")
        for i, result in enumerate(results, 1):
            match, score = result[0], result[1]
            print(f"     {i}. {match}  (score: {score})")
        print(f"     0. Skip")
        choice = input("  Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            chosen = results[int(choice) - 1][0]
            return symptoms_list[symptoms_lower.index(chosen)]
        return None

    else:
        return None


# ── User input ────────────────────────────────────────────────────────────────
user_input = input("\nEnter symptoms (comma separated): ")
user_symptoms = [s.strip() for s in user_input.split(",")]

# ── Build input vector ────────────────────────────────────────────────────────
input_data = [0] * len(symptoms_list)
matched = 0
total_severity = 0
matched_symptoms = []

for raw_symptom in user_symptoms:
    resolved = resolve_symptom(raw_symptom)
    if resolved:
        idx = symptoms_list.index(resolved)
        input_data[idx] = 1  # Binary input — matches training data
        weight = sev_dict.get(resolved.lower(), 1)
        total_severity += weight
        matched += 1
        matched_symptoms.append(resolved)
    else:
        print(f"  ⚠️ '{raw_symptom}' — no match found, skipping")

if matched == 0:
    print("\n❌ No valid symptoms entered. Cannot make a prediction.")
    exit()

if matched <= 2:
    print(f"\n⚠️  Only {matched} symptom(s) matched. Predictions based on very few")
    print("   symptoms are less reliable. Try entering more symptoms for better results.")

# ── Severity Summary ─────────────────────────────────────────────────────────
if sev_dict:
    avg_severity = total_severity / matched
    severity_label = "🔴 High" if avg_severity >= 5 else ("🟡 Medium" if avg_severity >= 3 else "🟢 Low")
    print(f"\n⚡ Symptom Severity: {severity_label} (avg score: {avg_severity:.1f}/7)")

    print("\n┌─────────────────────────────────────────────────┐")
    print("│  MATCHED SYMPTOMS                    SEVERITY   │")
    print("├─────────────────────────────────────────────────┤")
    for symptom in matched_symptoms:
        name = symptom.replace("_", " ").title()
        sev_val = sev_dict.get(symptom.lower(), 0)
        sev_bar = "█" * sev_val + "░" * (7 - sev_val)
        sev_color = "🔴" if sev_val >= 5 else ("🟡" if sev_val >= 3 else "🟢")
        print(f"│  {sev_color} {name:<32} {sev_bar} {sev_val}/7 │")
    print("└─────────────────────────────────────────────────┘")

# ── Predict ───────────────────────────────────────────────────────────────────
input_df = pd.DataFrame([input_data], columns=symptoms_list)
proba = model.predict_proba(input_df)[0]
classes = model.classes_

top3_idx = proba.argsort()[-3:][::-1]

# Collect top predictions
top_predictions = []
for rank, idx in enumerate(top3_idx, 1):
    disease = classes[idx]
    confidence = proba[idx] * 100
    if confidence < 1:
        break
    top_predictions.append((rank, disease, confidence))

# Add reliability note for sparse input
if matched <= 2 and top_predictions:
    print(f"\n  ℹ️  Low confidence warning: only {matched} symptom(s) used.")
    print("     Results below may not be accurate.")

# ── Print Results ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("🩺  TOP PREDICTIONS")
print("═" * 55)

for rank, disease, confidence in top_predictions:
    medal = ["🥇", "🥈", "🥉"][rank - 1]

    # Confidence bar
    bar_len = int(confidence / 2)
    bar = "█" * bar_len + "░" * (50 - bar_len)
    print(f"\n{medal}  {disease}")
    print(f"   [{bar}] {confidence:.1f}%")

    if rank == 1:
        print()
        if disease in desc_dict:
            print(f"   📖 Description:")
            # Word wrap the description
            words = desc_dict[disease].split()
            line = "      "
            for word in words:
                if len(line + " " + word) > 55:
                    print(line)
                    line = "      " + word
                else:
                    line = line + " " + word
            print(line)

        if disease in prec_dict and prec_dict[disease]:
            print(f"\n   🛡️  Precautions:")
            for i, p in enumerate(prec_dict[disease], 1):
                print(f"      {i}. {p.capitalize()}")

print("\n" + "═" * 55)
print("⚠️  This tool is for educational purposes only.")
print("   Always consult a qualified doctor for medical advice.")
print("═" * 55)