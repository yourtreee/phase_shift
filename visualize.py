import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

READABLE = {
    "disease_category_enc": "Disease Category",
    "method_enc": "Gene-Editing Method",
    "num_countries": "Number of Countries",
    "start_year": "Start Year",
    "trial_age": "Trial Age (Years Running)",
    "enrollment": "Enrollment Size"
}

def plot_feature_importance(importance_df):
    importance_df["Feature Label"] = importance_df["Feature"].map(READABLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2563eb" if i == 0 else "#93c5fd" for i in range(len(importance_df))]
    ax.barh(importance_df["Feature Label"][::-1], importance_df["Importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("PhaseShift — Feature Importance\n(What Predicts CRISPR Trial Phase Advancement?)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("phaseshift_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Phase 1 Only", "Advanced (P2/3)"],
                yticklabels=["Phase 1 Only", "Advanced (P2/3)"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("PhaseShift — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("phaseshift_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
