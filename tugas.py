import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
from flask import Flask, request, jsonify

# === 1. Load Dataset ===
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

print("Distribusi label:\n", y.value_counts())

# === 2. Split Data ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(f"✅ Split selesai: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# === 3. Preprocessing ===
num_cols = X_train.select_dtypes(include="number").columns
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

# === 4. Logistic Regression Baseline ===
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])
pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)
print("\n=== Logistic Regression ===")
print("F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# === 5. Random Forest Baseline ===
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])
pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("\n=== Random Forest Baseline ===")
print("F1(val):", f1_score(y_val, y_val_rf, average="macro"))

# === 6. Hyperparameter Tuning ===
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param = {
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("\n=== Grid Search ===")
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))

# === 7. Evaluate on Test ===
y_test_pred = best_rf.predict(X_test)
print("\n=== TEST RESULT ===")
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# === 8. ROC Curve ===
if hasattr(best_rf, "predict_proba"):
    y_test_proba = best_rf.predict_proba(X_test)[:, 1]
    try:
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print(f"ROC-AUC(test): {roc_auc:.3f}")
    except Exception as e:
        print("ROC-AUC error:", e)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
    print("📈 ROC curve saved as 'roc_test.png'")

# === 9. Save Model ===
joblib.dump(best_rf, "model.pkl")
print("✅ Model tersimpan ke 'model.pkl'")

# === 10. Flask API ===
app = Flask(__name__)
MODEL = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X_input = pd.DataFrame([data])
    yhat = MODEL.predict(X_input)[0]
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(X_input)[:, 1][0])
    return jsonify({"prediction": int(yhat), "proba": proba})

if __name__ == "__main__":
    app.run(port=5000)
