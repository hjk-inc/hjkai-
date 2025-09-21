# HJKAI Test Suite

This document explains the **10 test cases** included in the `tests/test_hjkai.py` file.

## ✅ Tests Included
1. **Clustering with KMeans** → Verifies AI can cluster data into groups.  
   🔗 *Image Prompt:* "A swarm of glowing AI clusters grouping into 3 categories in cyberspace."
2. **PCA Reduction** → Ensures dimensionality reduction works.  
   🔗 *Image Prompt:* "Data points shrinking from high dimensions to 2D glowing plane."
3. **RandomForest Classifier** → Checks classification accuracy.  
   🔗 *Image Prompt:* "Digital forest of decision trees making predictions."
4. **TF-IDF Vectorization** → Confirms text vectorization works.  
   🔗 *Image Prompt:* "AI reading documents and converting words into numbers and vectors."
5. **Language Detection** → Detects Spanish text correctly.  
   🔗 *Image Prompt:* "Flags of different countries connected with AI brain."
6. **spaCy Tokenization** → Splits text into tokens.  
   🔗 *Image Prompt:* "Sentence breaking into glowing word tokens inside AI brain."
7. **SQLite Database Test** → Creates a test database.  
   🔗 *Image Prompt:* "Digital database vault with AI inserting knowledge."
8. **Random Reproducibility** → Ensures results are reproducible.  
   🔗 *Image Prompt:* "Two identical holograms proving consistency in AI world."
9. **File I/O** → Saves and reads files correctly.  
   🔗 *Image Prompt:* "An AI hand writing and reading digital notes."
10. **Environment Variables** → Confirms system env vars work.  
   🔗 *Image Prompt:* "AI controlling secret keys floating in cyberspace."

## 📌 How to Run
```bash
pytest tests/test_hjkai.py --maxfail=1 --disable-warnings -q

Outputs will be logged cleanly to console.
To save them into a file:

pytest tests/test_hjkai.py > test_results.txt

You can print/share test_results.txt without messy console reactions.
