import pytest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from langdetect import detect
import spacy
import sqlite3
import os

# 1. Test KMeans clustering works
def test_kmeans_clustering():
    X = np.random.rand(20, 5)
    model = KMeans(n_clusters=3, random_state=42).fit(X)
    assert len(model.labels_) == 20

# 2. Test PCA dimensionality reduction
def test_pca_reduction():
    X = np.random.rand(50, 10)
    pca = PCA(n_components=2).fit_transform(X)
    assert pca.shape[1] == 2

# 3. Test RandomForest classifier accuracy
def test_randomforest_classifier():
    X = np.random.rand(40, 4)
    y = np.random.randint(0, 2, 40)
    clf = RandomForestClassifier().fit(X, y)
    preds = clf.predict(X)
    assert accuracy_score(y, preds) >= 0.5

# 4. Test TF-IDF vectorization
def test_tfidf_vectorizer():
    docs = ["HJKAI learns AI", "AI builds better worlds", "Data is knowledge"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    assert X.shape[0] == 3

# 5. Test language detection
def test_language_detection():
    text = "Hola, cómo estás"
    lang = detect(text)
    assert lang == "es"

# 6. Test spaCy tokenization
def test_spacy_tokenization():
    nlp = spacy.blank("en")
    doc = nlp("HJKAI will change everything")
    tokens = [token.text for token in doc]
    assert "HJKAI" in tokens

# 7. Test database creation (SQLite)
def test_sqlite_database():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER, name TEXT)")
    cursor.execute("INSERT INTO test VALUES (1, 'HJKAI')")
    conn.commit()
    cursor.execute("SELECT name FROM test WHERE id=1")
    result = cursor.fetchone()
    assert result[0] == "HJKAI"
    conn.close()

# 8. Test random number reproducibility
def test_random_reproducibility():
    np.random.seed(42)
    a = np.random.rand(5)
    np.random.seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)

# 9. Test file writing/reading
def test_file_io(tmp_path):
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("HJKAI Test")
    with open(file_path, "r") as f:
        content = f.read()
    assert content == "HJKAI Test"

# 10. Test system environment variable handling
def test_env_variable(monkeypatch):
    monkeypatch.setenv("HJKAI_KEY", "12345")
    assert os.getenv("HJKAI_KEY") == "12345"
