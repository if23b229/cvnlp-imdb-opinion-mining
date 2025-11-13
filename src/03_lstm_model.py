# src/03_lstm_model.py
import random, numpy as np
from pathlib import Path
import tensorflow as tf
from keras import Sequential
from keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from utils_imdb import read_imdb_split, basic_clean, SEED

# -------------------- Reproducibility --------------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- Paths --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = (BASE_DIR / "../data/aclImdb").resolve()
MODELS_DIR = (BASE_DIR / "../outputs/models").resolve()
PRED_PATH = (BASE_DIR / "../outputs/preds_lstm.csv").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Dev toggle (subset while iterating) --------------------
# Set to an int (e.g., 3000) for faster dev runs; set to None for full dataset
MAX_DOCS_PER_CLASS = None

# -------------------- Hyperparameters --------------------
MAX_VOCAB = 40_000
MAX_LEN   = 300
EMBED_DIM = 100
BATCH_SIZE = 64
EPOCHS = 6

# -------------------- Load & clean --------------------
train = read_imdb_split(DATA_ROOT / "train", max_docs_per_class=MAX_DOCS_PER_CLASS)
test  = read_imdb_split(DATA_ROOT / "test",  max_docs_per_class=MAX_DOCS_PER_CLASS)

Xtr = train["text"].apply(basic_clean).tolist()
Xte = test["text"].apply(basic_clean).tolist()
ytr = train["label"].to_numpy(dtype="int64")
yte = test["label"].to_numpy(dtype="int64")

# -------------------- Vectorizer (tokenize + pad in-graph) --------------------
vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=MAX_LEN
)  # default standardization is fine; we already do light cleaning

ds_text = tf.data.Dataset.from_tensor_slices(Xtr).batch(256)
vectorizer.adapt(ds_text)

# -------------------- Model --------------------
model = Sequential([
    vectorizer,
    Embedding(input_dim=MAX_VOCAB, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(128)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# -------------------- Train --------------------
es = EarlyStopping(monitor="val_accuracy", patience=2, mode="max", restore_best_weights=True)
history = model.fit(
    x=np.array(Xtr, dtype=object),  # strings in; vectorizer handles tokenization + padding
    y=ytr,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose="auto"
)

# -------------------- Evaluate --------------------
proba = model.predict(np.array(Xte, dtype=object), batch_size=BATCH_SIZE).ravel()
pred  = (proba >= 0.5).astype(int)

print("Test Accuracy:", accuracy_score(yte, pred))
print(classification_report(yte, pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(yte, pred))

# -------------------- Save artifacts --------------------
model.save(MODELS_DIR / "lstm_imdb.keras")
print(f"Saved: {MODELS_DIR / 'lstm_imdb.keras'}")

pd.DataFrame({"text": Xte, "y_true": yte, "y_pred": pred, "proba": proba}).to_csv(PRED_PATH, index=False)
print(f"Saved: {PRED_PATH}")
