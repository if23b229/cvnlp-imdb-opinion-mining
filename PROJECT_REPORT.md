# Projektbericht: IMDB Opinion Mining

**Projekt:** Natural Language Processing - Sentiment-Analyse auf IMDB Movie Reviews  


---

## 1. Dataset-Beschreibung

### 1.1 Dataset-Übersicht
Das **IMDB Large Movie Review Dataset (aclImdb)** enthält 50.000 Film-Reviews von der Internet Movie Database (IMDB). Das Dataset ist in Train- und Test-Sets aufgeteilt, jeweils mit 25.000 Reviews.

### 1.2 Dataset-Charakteristika

**Balance:**
- Das Dataset ist **balanciert**: 50% positive Reviews (Label: 1) und 50% negative Reviews (Label: 0)
- Train-Set: 25.000 Reviews (12.500 positive, 12.500 negative)
- Test-Set: 25.000 Reviews (12.500 positive, 12.500 negative)

**Text-Charakteristika:**
- Durchschnittliche Wortanzahl pro Review: ~230-250 Wörter (variiert je nach Subset)
- Median Wortanzahl: ~180-200 Wörter
- Textlänge variiert stark (von kurzen Reviews mit <50 Wörtern bis zu sehr langen Texten mit >1000 Wörtern)
- Reviews enthalten HTML-Tags (z.B. `<br/>`), die im Preprocessing entfernt werden
- Durchschnittliche Anzahl von Sätzen pro Review: ~10-15 Sätze

**Preprocessing:**
- Lowercasing aller Texte
- Entfernung von HTML-Tags (`<br/>`)
- Entfernung von URLs
- Weitere Normalisierung

### 1.3 Beispiel-Reviews

**Beispiel 1: Positives Review (Label: 1)**
```
When I was a kid, I loved "Tiny Toons". I especially loved the character of Buster Bunny. 
This movie captures that same spirit perfectly. The animation is top-notch, the humor is 
witty and the story is engaging. It's a great film for both kids and adults who grew up 
watching the show. Highly recommended!
```
*Charakteristika: Positive Wörter (loved, great, perfect, recommended), klare positive Bewertung*

**Beispiel 2: Negatives Review (Label: 0)**
```
Extremely formulaic with cosmic-sized logic holes. The characters are one-dimensional, 
the plot makes no sense, and the dialogue is cringe-worthy. I can't believe I wasted 
two hours of my life on this. Save your money and watch something else instead.
```
*Charakteristika: Negative Wörter (formulaic, wasted, cringe-worthy), klare negative Bewertung*

**Beobachtungen:**
- Positive Reviews enthalten typischerweise Wörter wie "great", "excellent", "amazing", "loved", "recommended"
- Negative Reviews enthalten Wörter wie "terrible", "awful", "waste", "boring", "disappointing"
- Die Reviews variieren stark in Länge und Detailgrad
- Viele Reviews enthalten persönliche Meinungen und emotionale Ausdrücke

---

## 2. Methodik

### 2.1 Klassischer Algorithmus: Naive Bayes + TF-IDF

**Warum Naive Bayes?**
- Schnelles Training und Evaluation
- Gute Baseline für Textklassifikation
- Interpretierbar
- Geringer Ressourcenbedarf

**Implementierung:**
- **TF-IDF Vectorization**: 
  - Max Features: 50.000
  - N-gram Range: (1, 2) - Unigrams und Bigrams
  - Min Document Frequency: 2
- **Naive Bayes**: MultinomialNB mit Laplace Smoothing (alpha=0.5)

**Vorteile:**
- Sehr schnelles Training
- Geringer Speicherbedarf
- Kann n-Gram-Muster erfassen (Bigrams helfen bei Phrasen)

**Nachteile:**
- Ignoriert Wortreihenfolge (Bag-of-Words Ansatz)
- Kann Negationen nicht gut modellieren (z.B. "not good" wird als "not" und "good" getrennt behandelt)

### 2.2 Deep Learning Algorithmus: Bidirectional LSTM

**Warum LSTM?**
- Kann Sequenzinformationen erfassen
- Modelliert langfristige Abhängigkeiten im Text
- Bidirektional erfasst Kontext in beide Richtungen

**Implementierung:**
- **Text Vectorization**: 
  - Max Vocabulary: 40.000 Tokens
  - Max Sequence Length: 300
- **Embedding Layer**: 100 Dimensionen
- **Bidirectional LSTM**: 128 Units
- **Dropout**: 0.4 nach LSTM, 0.3 vor finaler Dense Layer
- **Dense Layers**: 64 Units (ReLU) + 1 Unit (Sigmoid)
- **Training**: 
  - Optimizer: Adam
  - Loss: Binary Crossentropy
  - Early Stopping: Patience=2, Monitor=val_accuracy
  - Validation Split: 15%

**Vorteile:**
- Erfasst Wortreihenfolge und Sequenzmuster
- Kann komplexe linguistische Muster lernen
- Bidirektional erfasst Kontext in beide Richtungen

**Nachteile:**
- Längeres Training (mehrere Minuten/Stunden)
- Höherer Ressourcenbedarf (GPU empfohlen)
- Weniger interpretierbar

---

## 3. Ergebnisse

### 3.1 Performance-Metriken

| Modell | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Naive Bayes + TF-IDF | 0.8705 | 0.8709 | 0.8705 | 0.8704 |
| BiLSTM | 0.8380 | 0.8416 | 0.8380 | 0.8376 |

### 3.2 Confusion Matrices
Die Confusion Matrices für beide Modelle sind in den folgenden Dateien gespeichert:
- `outputs/confusion_matrix_nb.png` - Naive Bayes Confusion Matrix
- `outputs/confusion_matrix_lstm.png` - BiLSTM Confusion Matrix
- `outputs/confusion_matrices_comparison.png` - Direkter Vergleich beider Matrices

### 3.3 Vergleich und Interpretation

**Ergebnisse:**
- **Naive Bayes + TF-IDF** erzielt in diesem Lauf die bessere Performance mit einer Accuracy von 87.05% gegenüber 83.80% beim BiLSTM
- Die Metriken sind bei beiden Modellen relativ ausgewogen (Precision, Recall und F1-Score liegen nahe beieinander)
- Beide Modelle zeigen gute Performance für eine binäre Klassifikationsaufgabe

**Interpretation:**
- **Warum Naive Bayes besser performt:** 
  - TF-IDF mit Bigrams erfasst bereits wichtige Phrasen-Muster (z.B. "not good", "very bad")
  - Das IMDB-Dataset enthält viele eindeutige Sentiment-Wörter, die gut mit Bag-of-Words erfasst werden können
  - Naive Bayes profitiert von der großen Trainingsdatenmenge (25.000 Trainingsbeispiele)
  
- **Warum LSTM etwas schlechter abschneidet:**
  - Möglicherweise benötigt das LSTM mehr Trainingsepochen oder Hyperparameter-Tuning
  - Das Modell könnte von einer größeren Embedding-Dimension oder mehr LSTM-Units profitieren
  - LSTM-Modelle benötigen oft mehr Daten oder längeres Training für optimale Performance

- **Fehlertypen:**
  - Beide Modelle zeigen ähnliche Fehlertypen (False Positives und False Negatives sind relativ ausgewogen)
  - Schwierige Fälle sind vermutlich Reviews mit gemischten Sentiments oder Ironie
  - Kurze Reviews mit wenig Kontext können für beide Modelle problematisch sein

- **Stärken/Schwächen:**
  - **Naive Bayes:** Schnell, ressourcenschonend, aber ignoriert Wortreihenfolge
  - **LSTM:** Erfasst Sequenzinformationen, aber langsamer und ressourcenintensiver

---

## 4. Real-World Anwendungen

**Sentiment-Analyse** hat viele praktische Anwendungen:

1. **E-Commerce**: 
   - Automatische Bewertung von Produktreviews
   - Priorisierung von negativen Reviews für Customer Service

2. **Social Media Monitoring**: 
   - Tracking von Marken-Sentiment in sozialen Medien
   - Erkennung von Krisen oder negativen Trends

3. **Customer Service**: 
   - Priorisierung von Beschwerden basierend auf Sentiment
   - Automatische Kategorisierung von Support-Anfragen

4. **Market Research**: 
   - Analyse von Kundenfeedback und Meinungen
   - Trendanalyse in verschiedenen Branchen

5. **Content Moderation**: 
   - Erkennung von negativen/hassfülligen Inhalten
   - Automatische Filterung von Kommentaren

**IMDB Reviews** sind besonders relevant für:
- **Filmstudios**: Analyse von Publikumsreaktionen auf neue Filme
- **Streaming-Plattformen**: Verbesserung von Empfehlungssystemen
- **Filmkritiker**: Trendanalyse und Verständnis von Publikumspräferenzen

---

## 5. Probleme und Lösungen

### 5.1 Probleme während der Implementierung

**Problem 1: Encoding-Fehler**
- **Beschreibung**: Beim Laden der IMDB-Dateien traten Encoding-Fehler auf
- **Lösung**: Verwendung von `encoding="utf-8", errors="ignore"` beim Datei-Lesen

**Problem 2: Speicherprobleme beim Training**
- **Beschreibung**: Vollständiges Dataset war zu groß für verfügbaren RAM
- **Lösung**: Optionale Begrenzung der Dokumente pro Klasse für schnelle Entwicklung (`max_docs_per_class` Parameter)

**Problem 3: LSTM Training-Zeit**
- **Beschreibung**: LSTM Training dauerte sehr lange
- **Lösung**: 
  - Early Stopping implementiert
  - Validation Split für bessere Generalisierung
  - Batch Processing optimiert

**Problem 4: TensorFlow/Keras Import-Fehler**
- **Beschreibung**: Beim Import von Keras traten Kompatibilitätsprobleme zwischen TensorFlow und Keras auf
- **Lösung**: Verwendung von `tensorflow.keras` statt separatem `keras` Package, oder Aktualisierung der TensorFlow-Version

### 5.2 Design-Entscheidungen

**Warum TF-IDF statt Count Vectorization?**
- TF-IDF gewichtet seltene Wörter höher, was für Sentiment-Analyse wichtig ist
- Reduziert den Einfluss von sehr häufigen Wörtern

**Warum Bidirectional LSTM statt unidirektional?**
- Bidirektional erfasst Kontext in beide Richtungen
- Wichtig für Verständnis von Negationen und komplexen Phrasen

**Warum Dropout?**
- Verhindert Overfitting
- Verbessert Generalisierung auf Test-Set

---

## 6. Was wurde gemacht und warum?

### 6.1 Daten-Preprocessing
- **Lowercasing**: Normalisiert Texte, reduziert Vokabular-Größe
- **HTML-Tag Entfernung**: Entfernt Formatierungs-Tags, die keine semantische Information enthalten
- **URL Entfernung**: URLs enthalten keine Sentiment-Information

### 6.2 Feature Engineering
- **TF-IDF mit Bigrams**: Erfasst Phrasen wie "not good", "very bad"
- **Text Vectorization für LSTM**: Konvertiert Texte in Sequenzen von Integer-IDs

### 6.3 Modell-Auswahl
- **Naive Bayes**: Schnelle Baseline, interpretierbar
- **LSTM**: State-of-the-art für Sequenz-Klassifikation, erfasst komplexe Muster

### 6.4 Evaluation
- **Accuracy**: Gesamt-Performance
- **Precision/Recall/F1**: Detailliertere Metriken für beide Klassen
- **Confusion Matrix**: Zeigt Fehlertypen (False Positives/Negatives)

---

## 7. Zusammenfassung

Dieses Projekt implementierte zwei verschiedene Ansätze für Sentiment-Analyse auf IMDB Movie Reviews:
1. **Klassischer Ansatz**: Naive Bayes mit TF-IDF Features
2. **Deep Learning Ansatz**: Bidirectional LSTM

Beide Modelle wurden trainiert, evaluiert und verglichen. Die Ergebnisse zeigen, dass **Naive Bayes + TF-IDF** mit einer Accuracy von 87.05% leicht besser abschneidet als das **BiLSTM** Modell (83.80% Accuracy).

**Haupt-Erkenntnisse:**
- Klassische Methoden (Naive Bayes + TF-IDF) können bei Sentiment-Analyse sehr konkurrenzfähig sein, besonders bei großen, balancierten Datasets
- Bigrams in TF-IDF helfen bereits, wichtige Phrasen-Muster zu erfassen, auch ohne explizite Sequenzmodellierung
- LSTM-Modelle bieten Potenzial für komplexere Muster, benötigen aber möglicherweise mehr Tuning oder Training
- Beide Ansätze haben ihre Berechtigung: NB für schnelle, ressourcenschonende Anwendungen, LSTM für komplexere linguistische Muster

**Zukünftige Verbesserungen:**
- **Transformer-Modelle (BERT, RoBERTa)**: Pre-trained Language Models könnten bessere Performance erzielen
- **Attention Mechanisms**: Könnten helfen, wichtige Wörter/Phrasen zu identifizieren
- **Hyperparameter-Tuning**: Systematische Suche nach optimalen Parametern für beide Modelle
- **Ensemble-Methoden**: Kombination beider Modelle könnte die Performance weiter verbessern
- **Data Augmentation**: Erhöhung der Trainingsdaten durch Synonym-Ersetzung oder Back-Translation

---

## 8. Referenzen

- **IMDB Dataset**: http://ai.stanford.edu/~amaas/data/sentiment/
- **Scikit-learn Dokumentation**: https://scikit-learn.org/stable/
- **Keras/TensorFlow Dokumentation**: https://www.tensorflow.org/api_docs/python/tf/keras
- **Maas, A. L., et al. (2011)**: Learning Word Vectors for Sentiment Analysis. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics
- **Pang, B., & Lee, L. (2008)**: Opinion Mining and Sentiment Analysis. Foundations and Trends in Information Retrieval


