# 📈 Stock Price Direction Predictor
### Beginner ML Project | Student Starter File

> **Your goal:** Build a model that predicts whether a stock's price will go **UP** or **DOWN** tomorrow.

---

## 🗺️ Project Roadmap

Work through each step **in order**. Every step builds on the last.

| Step | What You'll Do | Key Concept |
|------|---------------|-------------|
| 1 | Set up imports & config | Libraries, variables |
| 2 | Download stock data | DataFrames, Yahoo Finance |
| 3 | Engineer features | Technical indicators |
| 4 | Prepare train/test split | Avoiding data leakage |
| 5 | Train the model | Logistic Regression |
| 6 | Evaluate performance | Accuracy, confusion matrix |
| 7 ⭐ | Predict on live data | Putting it all together |

---

## 🚀 Getting Started in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → New Notebook**
3. In the **first cell**, run:
   ```python
   !pip install yfinance scikit-learn pandas matplotlib seaborn
   ```
4. In the **second cell**, paste the contents of `stock_prediction_student.py`
5. Work through the `TODO` comments from top to bottom

---

## 🧠 Key Concepts to Understand Before You Start

### What is a feature?
A feature is any input you give to the model. For example, "how much did the stock move yesterday?" is a feature. Better features → smarter model.

### What is a label / target?
The thing you're trying to predict. In our case: did the price go **UP (1)** or **DOWN (0)** the next day?

### What is train/test split?
We hide some data from the model during training, then test on it later — like studying without looking at the exam answers first.

### Why do we scale features?
Logistic Regression is sensitive to the size of numbers. Scaling puts all features on the same scale so no single feature dominates unfairly.

---

## 💡 Hints (read only if you're stuck!)

<details>
<summary>Step 1 — How do I import a library?</summary>

```python
import yfinance as yf
import pandas as pd
```
</details>

<details>
<summary>Step 2 — How do I download stock data?</summary>

```python
df = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
```
</details>

<details>
<summary>Step 3 — How do I calculate a moving average?</summary>

```python
df["MA5"] = df["Close"].rolling(window=5).mean()
```
</details>

<details>
<summary>Step 3 — How do I create the Target column?</summary>

```python
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
```
</details>

<details>
<summary>Step 4 — How do I split my data?</summary>

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
```
</details>

<details>
<summary>Step 4 — How do I scale my data?</summary>

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```
</details>

<details>
<summary>Step 5 — How do I train the model?</summary>

```python
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```
</details>

<details>
<summary>Step 6 — How do I evaluate the model?</summary>

```python
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
</details>

---

## 🔧 Try Different Stocks

Once your model works, try swapping the ticker and compare results:

```python
TICKER = "TSLA"   # Tesla
TICKER = "MSFT"   # Microsoft
TICKER = "AMZN"   # Amazon
TICKER = "GOOG"   # Google
TICKER = "NVDA"   # NVIDIA
```

Does accuracy change between stocks? Why might that be?

---

## 🌟 Bonus Challenges

Finished early? Try these:

1. **Add a new feature** — can you find another indicator that improves accuracy?
2. **Visualize the stock price** — plot the closing price over time using matplotlib
3. **Try a different model** — swap `LogisticRegression` for `RandomForestClassifier` and compare
4. **Live prediction** — complete Step 7 to predict what tomorrow might look like

---

## ❓ Questions to Think About

- What does an accuracy of 55% mean for a stock predictor? Is that good or bad?
- Why do we use `shuffle=False` when splitting the data?
- Why do we `fit_transform` on training data but only `transform` on test data?
- What would happen if we used future data to train our model? (Hint: this is called **data leakage**)

---

## ⚠️ Disclaimer

This project is **for learning purposes only**.  
Do **not** use this model to make real investment decisions.
