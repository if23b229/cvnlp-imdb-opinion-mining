# CV/NLP – IMDB Opinion Mining

Dieses Projekt gehört zur Lehrveranstaltung *Computer Vision & Natural Language Processing* (CVNLP).  
Ziel ist es, Sentiment-Analyse auf dem **IMDB Large Movie Review Dataset (aclImdb)** zu machen und verschiedene Modelle zu vergleichen:

- Naive Bayes mit TF-IDF Features
- LSTM-Modell auf Wortebene

---

## Projektstruktur

```text
Prjkt_NLP/
├─ src/
│  ├─ 01_explore_dataset.py      # Explorative Analyse des Datasets
│  ├─ 02_nb_tfidf_baseline.py    # Naive Bayes + TF-IDF Baseline
│  ├─ 03_lstm_model.py           # LSTM-Modell mit Keras
│  ├─ utils_imdb.py              # Hilfsfunktionen (Daten laden, Preprocessing, etc.)
│  └─ __init__.py
│
├─ outputs/
│  ├─ preds_nb.csv               # Vorhersagen des NB-Modells (Beispieloutput)
│  └─ (optional: figures/, models/ – werden lokal erzeugt)
│
├─ data/
│  └─ aclImdb/                   # IMDB-Dataset (liegt NICHT im Repo, siehe unten)
│
├─ env311/                       # Lokales venv (liegt NICHT im Repo)
├─ requirements.txt
├─ Project -NLP.pdf              # Projektbeschreibung / Bericht
└─ .gitignore
