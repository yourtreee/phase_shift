import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

FEATURES = ["disease_category_enc", "method_enc", "num_countries", "start_year", "trial_age", "enrollment"]

def train_phaseshift(feat_df):
    model_df = feat_df[FEATURES + ["advanced"]].dropna().reset_index(drop=True)
    model_df_sampled = model_df.sample(n=min(250, len(model_df)), random_state=42)

    X = model_df_sampled[FEATURES]
    y = model_df_sampled["advanced"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=2,
        max_features="sqrt", random_state=42, class_weight="balanced")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_precision = cross_val_score(model, X, y, cv=cv, scoring="precision")
    cv_recall    = cross_val_score(model, X, y, cv=cv, scoring="recall")
    cv_f1        = cross_val_score(model, X, y, cv=cv, scoring="f1")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 50)
    print("📊 PHASESHIFT — 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
    print(f"  Precision: {cv_precision.mean():.3f} ± {cv_precision.std():.3f}")
    print(f"  Recall:    {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")
    print(f"  F1 Score:  {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(classification_report(y_test, y_pred, target_names=["Phase 1 Only", "Advanced (P2/3)"]))

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, importance_df, (X_test, y_test, y_pred), (cv_accuracy, cv_precision, cv_recall, cv_f1)
