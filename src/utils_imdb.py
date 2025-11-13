import os, glob, re, random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def _read_many(files, label, max_docs=None):
    texts, labels = [], []
    it = files if max_docs is None else files[:max_docs]
    for f in tqdm(it, desc=f"Reading {'pos' if label==1 else 'neg'}", ncols=80):
        # Some IMDB files have odd chars; ignore undecodable bytes
        with open(f, encoding="utf-8", errors="ignore") as fh:
            texts.append(fh.read())
            labels.append(label)
    return texts, labels

def read_imdb_split(split_dir: Path, max_docs_per_class: int | None = None) -> pd.DataFrame:
    """Read IMDB split (train or test) with optional cap per class for quick runs."""
    split_dir = Path(split_dir)
    pos_files = sorted(glob.glob(os.path.join(split_dir, "pos", "*.txt")))
    neg_files = sorted(glob.glob(os.path.join(split_dir, "neg", "*.txt")))

    pos_texts, pos_labels = _read_many(pos_files, 1, max_docs_per_class)
    neg_texts, neg_labels = _read_many(neg_files, 0, max_docs_per_class)

    texts = pos_texts + neg_texts
    labels = pos_labels + neg_labels
    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    return text

#note to self: der Name eines Files darf nicht mit nem Zahl anfangen ncncnc