from pathlib import Path
from utils_imdb import read_imdb_split, basic_clean

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = (BASE_DIR / "../data/aclImdb").resolve()

# Read a subset first to confirm everything works fast
train_df = read_imdb_split(DATA_ROOT / "train", max_docs_per_class=500)
test_df  = read_imdb_split(DATA_ROOT / "test",  max_docs_per_class=500)

print("Train size:", len(train_df), "Test size:", len(test_df))
print("\nLabel balance (train):")
print(train_df["label"].value_counts(normalize=True))

train_df["clean"] = train_df["text"].apply(basic_clean)
avg_len = train_df["clean"].str.split().map(len).mean()
print("\nAvg tokens per review (train):", round(avg_len, 1))