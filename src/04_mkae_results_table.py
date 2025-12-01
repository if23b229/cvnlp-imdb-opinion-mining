from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = (BASE_DIR / "../outputs").resolve()

# Pfade zu den Predictions
pred_nb_path   = OUT_DIR / "preds_nb.csv"
pred_lstm_path = OUT_DIR / "preds_lstm.csv"

def load_preds(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fehlt: {path} (bitte Skript für {model_name} zuerst laufen lassen)")
    df = pd.read_csv(path)
    # Erwartet Spalten: text, y_true, y_pred (und bei LSTM zusätzlich proba)
    needed = {"y_true", "y_pred"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Datei {path} hat nicht die erwarteten Spalten (benötigt: {needed})")
    df["model"] = model_name
    return df

def metrics_from(df: pd.DataFrame):
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

def md_row(name, acc, prec, rec, f1):
    return f"| {name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |"

def main():
    nb   = load_preds(pred_nb_path,   "TF-IDF + Naive Bayes")
    lstm = load_preds(pred_lstm_path, "BiLSTM")

    acc_nb,   prec_nb,   rec_nb,   f1_nb,   cm_nb   = metrics_from(nb)
    acc_lstm, prec_lstm, rec_lstm, f1_lstm, cm_lstm = metrics_from(lstm)

    # Markdown-Tabelle
    table = "\n".join([
        "| Modell                | Accuracy | Precision | Recall | F1   |",
        "|-----------------------|----------|-----------|--------|------|",
        md_row("TF-IDF + Naive Bayes", acc_nb,   prec_nb,   rec_nb,   f1_nb),
        md_row("BiLSTM",               acc_lstm, prec_lstm, rec_lstm, f1_lstm),
        ""
    ])

    # Confusion-Matrizen (Text)
    cm_txt = []
    for name, cm in [("TF-IDF + Naive Bayes", cm_nb), ("BiLSTM", cm_lstm)]:
        cm_txt.append(f"{name} – Confusion Matrix:\n{cm}\n")

    # Kurze Deutung (schlanker Template-Text)
    comment_lines = []
    if f1_lstm >= f1_nb + 0.005:
        comment_lines.append(
            "- **BiLSTM** erzielt die höhere F1-Score. Vermutlich hilft die Modellierung von Wortreihenfolge/Negation (z. B. „not good“), "
            "während TF-IDF+NB nur Beutel-von-Wörtern ohne Sequenz erfasst."
        )
    elif f1_nb >= f1_lstm + 0.005:
        comment_lines.append(
            "- **TF-IDF+NB** ist in diesem Lauf vorn. Das kann auf starke n-Gram-Signale und gutes Regularisieren hinweisen; "
            "gleichzeitig trainiert NB deutlich schneller und ist ressourcenschonend."
        )
    else:
        comment_lines.append(
            "- **Beide Modelle** sind leistungsmäßig sehr nah beieinander. LSTM erfasst Sequenzinformationen, NB ist deutlich schneller."
        )

    # Dateien schreiben
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results_table.md").write_text(table, encoding="utf-8")
    (OUT_DIR / "confusion_matrices.txt").write_text("\n".join(cm_txt), encoding="utf-8")
    (OUT_DIR / "results_comment.txt").write_text("\n".join(comment_lines), encoding="utf-8")

    print("Ergebnis-Tabelle -> outputs/results_table.md")
    print("Confusion-Matrizen -> outputs/confusion_matrices.txt")
    print("Kurzdeutung -> outputs/results_comment.txt")
    print("\nMarkdown-Tabelle:\n")
    print(table)

if __name__ == "__main__":
    main()