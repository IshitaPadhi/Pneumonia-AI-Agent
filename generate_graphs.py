"""
generate_graphs.py — XPneumoNet Report Graph Generator
=======================================================
Run this from your project root AFTER training:
    python generate_graphs.py

Outputs all graphs to: graphs/
Requires: best_pneumonia_model.keras + dataset/chest_xray/test/

Graphs produced:
  1. Confusion Matrix (normalised)
  2. Per-Class Classification Report (bar chart)
  3. ROC Curves (one per class, OvR)
  4. Precision-Recall Curves
  5. Training History — Accuracy & Loss (load from saved CSV or history dict)
  6. Class Distribution (dataset balance)
  7. Confidence Score Distribution (per class)
  8. Grad-CAM Sample Grid (best-confidence predictions, one per class)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ── Output directory ──────────────────────────────────────────────
os.makedirs("graphs", exist_ok=True)

# ── Global style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e2e8f0",
    "grid.linewidth":     0.7,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

CLASSES   = ["BACTERIA", "NORMAL", "VIRUS"]
PALETTE   = {"BACTERIA": "#ef4444", "NORMAL": "#10b981", "VIRUS": "#f59e0b"}
COLORS    = [PALETTE[c] for c in CLASSES]
TEST_DIR  = "dataset/chest_xray/test"

# ─────────────────────────────────────────────────────────────────
# 1. LOAD MODEL + GENERATE PREDICTIONS
# ─────────────────────────────────────────────────────────────────
print("Loading model…")
model = tf.keras.models.load_model("model/best_pneumonia_model.keras")

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

print("Running predictions on test set…")
probs  = model.predict(test_gen, verbose=1)          # shape (N, 3)
y_pred = np.argmax(probs, axis=1)
y_true = test_gen.classes
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])  # (N, 3) one-hot

# ─────────────────────────────────────────────────────────────────
# GRAPH 1 — Confusion Matrix (raw + normalised side by side)
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 1: Confusion Matrix…")

cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrix — DenseNet-121 on Test Set", fontsize=13, fontweight="bold", y=1.02)

for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ["Raw Counts", "Normalised (row %)"],
    ["d", ".2f"]
):
    sns.heatmap(
        data, annot=True, fmt=fmt,
        cmap="Blues", ax=ax,
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5, linecolor="#e2e8f0",
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("graphs/01_confusion_matrix.png")
plt.close()
print("  ✓ graphs/01_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 2 — Per-Class Metrics Bar Chart (Precision / Recall / F1)
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 2: Classification Metrics…")

report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)

metrics   = ["precision", "recall", "f1-score"]
x         = np.arange(len(CLASSES))
bar_width  = 0.25
metric_colors = ["#1d4ed8", "#059669", "#d97706"]

fig, ax = plt.subplots(figsize=(10, 5))
for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    vals = [report[cls][metric] for cls in CLASSES]
    bars = ax.bar(x + i * bar_width, vals, bar_width,
                  label=metric.capitalize(), color=color, alpha=0.88,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#1e293b")

ax.set_xticks(x + bar_width)
ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("Per-Class Precision, Recall & F1-Score", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.axhline(y=report["macro avg"]["f1-score"], color="#64748b",
           linestyle="--", linewidth=1, alpha=0.7)
ax.text(len(CLASSES) - 0.15, report["macro avg"]["f1-score"] + 0.01,
        f"Macro F1 = {report['macro avg']['f1-score']:.2f}",
        fontsize=8, color="#64748b")

plt.tight_layout()
plt.savefig("graphs/02_classification_metrics.png")
plt.close()
print("  ✓ graphs/02_classification_metrics.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 3 — ROC Curves (One-vs-Rest, per class)
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 3: ROC Curves…")

fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, color) in enumerate(zip(CLASSES, COLORS)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.2,
            label=f"{cls}  (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.05, color=color)

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("ROC Curves — One-vs-Rest (OvR)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9.5, loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig("graphs/03_roc_curves.png")
plt.close()
print("  ✓ graphs/03_roc_curves.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 4 — Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 4: Precision-Recall Curves…")

fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, color) in enumerate(zip(CLASSES, COLORS)):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
    ap = average_precision_score(y_true_bin[:, i], probs[:, i])
    ax.plot(recall, precision, color=color, lw=2.2,
            label=f"{cls}  (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.05, color=color)

ax.set_xlabel("Recall", fontsize=10)
ax.set_ylabel("Precision", fontsize=10)
ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9.5, loc="upper right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig("graphs/04_precision_recall_curves.png")
plt.close()
print("  ✓ graphs/04_precision_recall_curves.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 5 — Training History
#   Option A: Load from history CSV saved during training (preferred)
#   Option B: Paste your values below as fallback
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 5: Training History…")

HISTORY_CSV = "model/training_history.csv"   # set to None if not saved

if HISTORY_CSV and os.path.exists(HISTORY_CSV):
    import csv
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    with open(HISTORY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in history:
                history[k].append(float(row[k]))
else:
    # ── FALLBACK: paste your actual values from Colab training output ──
    # Example values — REPLACE with your real numbers
    history = {
        "accuracy":     [0.62, 0.71, 0.78, 0.83, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93],
        "val_accuracy": [0.60, 0.68, 0.74, 0.79, 0.82, 0.85, 0.86, 0.87, 0.88, 0.88],
        "loss":         [0.85, 0.72, 0.61, 0.52, 0.45, 0.39, 0.34, 0.30, 0.27, 0.25],
        "val_loss":     [0.90, 0.76, 0.65, 0.56, 0.50, 0.44, 0.40, 0.37, 0.35, 0.34],
    }
    print("  ⚠  Using fallback history values — replace with your real Colab numbers.")

epochs = range(1, len(history["accuracy"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("DenseNet-121 Training History", fontsize=13, fontweight="bold", y=1.02)

# Accuracy
ax1.plot(epochs, history["accuracy"],     color="#1d4ed8", lw=2, marker="o", ms=4, label="Train Accuracy")
ax1.plot(epochs, history["val_accuracy"], color="#1d4ed8", lw=2, marker="o", ms=4, ls="--", alpha=0.7, label="Val Accuracy")
ax1.fill_between(epochs, history["accuracy"], history["val_accuracy"], alpha=0.07, color="#1d4ed8")
ax1.set_xlabel("Epoch", fontsize=10); ax1.set_ylabel("Accuracy", fontsize=10)
ax1.set_title("Accuracy over Epochs", fontsize=11, pad=8)
ax1.legend(fontsize=9); ax1.set_ylim([0.5, 1.02])
best_val_acc = max(history["val_accuracy"])
best_ep      = history["val_accuracy"].index(best_val_acc) + 1
ax1.axvline(best_ep, color="#64748b", ls=":", lw=1)
ax1.text(best_ep + 0.1, 0.52, f"Best val acc\n{best_val_acc:.3f}", fontsize=7.5, color="#64748b")

# Loss
ax2.plot(epochs, history["loss"],     color="#dc2626", lw=2, marker="o", ms=4, label="Train Loss")
ax2.plot(epochs, history["val_loss"], color="#dc2626", lw=2, marker="o", ms=4, ls="--", alpha=0.7, label="Val Loss")
ax2.fill_between(epochs, history["loss"], history["val_loss"], alpha=0.07, color="#dc2626")
ax2.set_xlabel("Epoch", fontsize=10); ax2.set_ylabel("Loss", fontsize=10)
ax2.set_title("Loss over Epochs", fontsize=11, pad=8)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("graphs/05_training_history.png")
plt.close()
print("  ✓ graphs/05_training_history.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 6 — Dataset Class Distribution
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 6: Dataset Distribution…")

splits = ["train", "val", "test"]
counts = {}
for split in splits:
    split_dir = os.path.join("dataset/chest_xray", split)
    split_counts = {}
    if os.path.exists(split_dir):
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                split_counts[cls] = len(os.listdir(cls_dir))
            else:
                split_counts[cls] = 0
    else:
        # Fallback values from the Mooney dataset (approximate)
        fallback = {"train": {"BACTERIA": 2530, "NORMAL": 1341, "VIRUS": 1345},
                    "val":   {"BACTERIA":   8,  "NORMAL":   8,   "VIRUS":   8},
                    "test":  {"BACTERIA": 242,  "NORMAL": 234,   "VIRUS": 148}}
        split_counts = fallback.get(split, {c: 0 for c in CLASSES})
    counts[split] = split_counts

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Dataset Class Distribution", fontsize=13, fontweight="bold", y=1.02)

# Grouped bar — counts per split per class
x        = np.arange(len(CLASSES))
bar_w    = 0.26
split_colors = ["#1d4ed8", "#059669", "#d97706"]
ax = axes[0]
for i, (split, sc) in enumerate(zip(splits, split_colors)):
    vals = [counts[split].get(cls, 0) for cls in CLASSES]
    bars = ax.bar(x + i * bar_w, vals, bar_w, label=split.capitalize(),
                  color=sc, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(v), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_xticks(x + bar_w)
ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_ylabel("Image Count", fontsize=10)
ax.set_title("Images per Class per Split", fontsize=11)
ax.legend(fontsize=9)

# Pie chart — overall class totals
total_per_class = {cls: sum(counts[s].get(cls, 0) for s in splits) for cls in CLASSES}
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    total_per_class.values(), labels=total_per_class.keys(),
    autopct="%1.1f%%", colors=COLORS,
    startangle=120, wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=9)
)
for at in autotexts:
    at.set_fontweight("bold"); at.set_fontsize(9)
ax2.set_title("Overall Class Balance", fontsize=11)

plt.tight_layout()
plt.savefig("graphs/06_dataset_distribution.png")
plt.close()
print("  ✓ graphs/06_dataset_distribution.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 7 — Confidence Score Distribution (violin + box)
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 7: Confidence Distribution…")

# Collect max confidence per sample, grouped by true class
conf_by_class = {cls: [] for cls in CLASSES}
for i, (true_idx, pred_probs) in enumerate(zip(y_true, probs)):
    conf_by_class[CLASSES[true_idx]].append(float(np.max(pred_probs)))

fig, ax = plt.subplots(figsize=(9, 5))
data_for_plot = [conf_by_class[cls] for cls in CLASSES]

# Violin
parts = ax.violinplot(data_for_plot, positions=[1, 2, 3],
                      showmedians=True, showextrema=True, widths=0.6)
for i, (pc, color) in enumerate(zip(parts["bodies"], COLORS)):
    pc.set_facecolor(color); pc.set_alpha(0.45)
parts["cmedians"].set_color("#0f172a"); parts["cmedians"].set_linewidth(2)

# Overlay scatter (strip)
for i, (cls, color) in enumerate(zip(CLASSES, COLORS)):
    jitter = np.random.normal(0, 0.04, size=len(conf_by_class[cls]))
    ax.scatter(
        np.full(len(conf_by_class[cls]), i + 1) + jitter,
        conf_by_class[cls],
        alpha=0.18, color=color, s=8, zorder=2
    )

ax.set_xticks([1, 2, 3]); ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_ylabel("Max Predicted Confidence", fontsize=10)
ax.set_ylim([0, 1.05])
ax.set_title("Model Confidence Distribution per True Class", fontsize=13, fontweight="bold", pad=12)
ax.axhline(y=0.85, color="#64748b", ls="--", lw=1, alpha=0.6)
ax.text(3.35, 0.86, "High-conf\nthreshold", fontsize=7.5, color="#64748b")

plt.tight_layout()
plt.savefig("graphs/07_confidence_distribution.png")
plt.close()
print("  ✓ graphs/07_confidence_distribution.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 8 — Grad-CAM Sample Grid (1 correct high-conf per class)
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 8: Grad-CAM Sample Grid…")

# Build Grad-CAM model
last_conv = None
for layer in reversed(model.layers):
    if "conv" in layer.name:
        last_conv = layer.name
        break

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv).output, model.output]
)

def get_gradcam(img_array, class_idx):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    grads    = tape.gradient(loss, conv_out)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm       = conv_out[0] @ pooled[..., tf.newaxis]
    hm       = tf.squeeze(hm).numpy()
    hm       = np.maximum(hm, 0)
    hm      /= hm.max() + 1e-8
    return hm

# Find best samples (correct + highest confidence, one per class)
best = {cls: {"conf": 0, "path": None, "pred_idx": None} for cls in CLASSES}
filenames  = test_gen.filenames
class_idx_map = {v: k for k, v in test_gen.class_indices.items()}  # idx -> name

for i, (true_idx, pred_idx, prob_row) in enumerate(zip(y_true, y_pred, probs)):
    cls_name = CLASSES[true_idx]
    if pred_idx == true_idx and prob_row[pred_idx] > best[cls_name]["conf"]:
        best[cls_name]["conf"]     = prob_row[pred_idx]
        best[cls_name]["path"]     = os.path.join(TEST_DIR, filenames[i])
        best[cls_name]["pred_idx"] = pred_idx

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Grad-CAM Explainability — Best Predictions per Class",
             fontsize=13, fontweight="bold", y=1.01)

col_titles = ["Original X-Ray", "Grad-CAM Heatmap", "Overlay"]

for row, cls in enumerate(CLASSES):
    info = best[cls]
    if info["path"] is None or not os.path.exists(info["path"]):
        for col in range(3):
            axes[row][col].axis("off")
            axes[row][col].text(0.5, 0.5, "No image found", ha="center", transform=axes[row][col].transAxes)
        continue

    img_bgr = cv2.imread(info["path"])
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    img_arr = np.expand_dims(img_rgb / 255.0, axis=0).astype(np.float32)

    hm      = get_gradcam(img_arr, info["pred_idx"])
    hm_r    = cv2.resize(hm, (224, 224))
    hm_col  = cv2.applyColorMap(np.uint8(255 * hm_r), cv2.COLORMAP_JET)
    hm_rgb  = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
    overlay = (hm_rgb * 0.4 + img_rgb * 0.6).astype(np.uint8)

    for col, img_data in enumerate([img_rgb, hm_rgb, overlay]):
        ax = axes[row][col]
        ax.imshow(img_data)
        ax.axis("off")
        if row == 0:
            ax.set_title(col_titles[col], fontsize=10, fontweight="bold", pad=6)
        if col == 0:
            ax.set_ylabel(f"{cls}\nConf: {info['conf']:.2%}",
                          fontsize=9, fontweight="bold",
                          color=PALETTE[cls], rotation=0,
                          labelpad=70, va="center")

plt.tight_layout()
plt.savefig("graphs/08_gradcam_samples.png")
plt.close()
print("  ✓ graphs/08_gradcam_samples.png")


# ─────────────────────────────────────────────────────────────────
# GRAPH 9 — Summary Table (text figure for report)
# ─────────────────────────────────────────────────────────────────
print("Generating Graph 9: Summary Results Table…")

report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
overall_acc = np.sum(y_pred == y_true) / len(y_true)

col_labels = ["Class", "Precision", "Recall", "F1-Score", "Support"]
rows = []
for cls in CLASSES:
    r = report_dict[cls]
    rows.append([cls, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                 f"{r['f1-score']:.3f}", str(int(r['support']))])
rows.append(["─" * 9] * 5)
rows.append(["Macro Avg",
             f"{report_dict['macro avg']['precision']:.3f}",
             f"{report_dict['macro avg']['recall']:.3f}",
             f"{report_dict['macro avg']['f1-score']:.3f}",
             str(int(report_dict['macro avg']['support']))])
rows.append(["Overall Acc", "", "", f"{overall_acc:.3f}", ""])

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis("off")
table = ax.table(
    cellText=rows, colLabels=col_labels,
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#1d4ed8")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Style class rows
for i, cls in enumerate(CLASSES):
    table[i + 1, 0].set_text_props(color=PALETTE[cls], fontweight="bold")

# Style macro row
for j in range(len(col_labels)):
    table[len(CLASSES) + 2, j].set_facecolor("#f8fafc")
    table[len(CLASSES) + 2, j].set_text_props(fontweight="bold")

ax.set_title("XPneumoNet — Classification Results Summary",
             fontsize=12, fontweight="bold", pad=10, y=1.02)
plt.tight_layout()
plt.savefig("graphs/09_results_table.png")
plt.close()
print("  ✓ graphs/09_results_table.png")


# ─────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  All graphs saved to: graphs/")
print("="*55)
print("""
  01_confusion_matrix.png       ← raw + normalised
  02_classification_metrics.png ← precision / recall / F1 bars
  03_roc_curves.png             ← AUC per class
  04_precision_recall_curves.png
  05_training_history.png       ← accuracy + loss over epochs
  06_dataset_distribution.png   ← class balance
  07_confidence_distribution.png← violin plot per class
  08_gradcam_samples.png        ← XAI explainability grid
  09_results_table.png          ← summary table
""")
print("  ⚠  For Graph 5 (Training History):")
print("     Paste your real Colab epoch values into the 'history' dict")
print("     OR save history to CSV during training and set HISTORY_CSV.")
print("     See instructions at the top of the file.\n")