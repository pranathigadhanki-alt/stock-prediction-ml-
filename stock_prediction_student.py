# ============================================================
#  Stock Price Direction Predictor
#  Beginner ML Project | Logistic Regression
#  Compatible with Google Colab
# ============================================================
#
# YOUR NAME:
# DATE:
#
# GOAL: Predict whether a stock's price will go UP or DOWN
#       the next trading day using Logistic Regression.
#
# HOW TO USE THIS FILE:
#   Work through each step in order (Step 1 → Step 7).
#   Every TODO is something you need to fill in.
#   Ask your instructor if you get stuck!
# ============================================================


# ── Step 0: Install dependencies (run this first in Colab) ──
# !pip install yfinance scikit-learn pandas matplotlib seaborn


# ============================================================
# STEP 1: Imports & Configuration
# ============================================================
# We start by importing all the libraries we'll need.
# Think of imports like grabbing your tools before starting a job.

# TODO: Import the following libraries:
#   - yfinance         (for downloading stock data)
#   - pandas           (for working with data tables)
#   - numpy            (for math operations)
#   - matplotlib.pyplot (for charts)
#   - seaborn          (for prettier charts)
#
#   From sklearn, import:
#   - LogisticRegression      (our ML model)
#   - train_test_split        (to split data into train/test)
#   - StandardScaler          (to normalize our features)
#   - accuracy_score, classification_report, confusion_matrix
#     (to measure model performance)

# YOUR CODE HERE:


# ── Settings ─────────────────────────────────────────────────
# TODO: Set these four variables:
#   TICKER      = the stock symbol you want to predict (e.g. "AAPL")
#   START_DATE  = the start of your historical data (e.g. "2018-01-01")
#   END_DATE    = the end of your historical data   (e.g. "2024-01-01")
#   TEST_SIZE   = fraction of data used for testing (e.g. 0.2 means 20%)
#   RANDOM_STATE = set to 42 (makes results reproducible)

# YOUR CODE HERE:


print(f"Configuration loaded. Predicting price direction for: {TICKER}")


# ============================================================
# STEP 2: Download & Explore the Data
# ============================================================
# We use the yfinance library to pull free historical stock data
# directly from Yahoo Finance.
#
# 💡 Key concept: yf.download() returns a DataFrame with columns:
#    Open, High, Low, Close, Volume, Adj Close
#    Each row = one trading day

# TODO: Use yf.download() to download stock data for your TICKER
#       between START_DATE and END_DATE. Save it to a variable called df.

# YOUR CODE HERE:


# TODO: Print the following to understand your data:
#   1. How many rows (trading days) are in the dataset?
#   2. The first 5 rows using df.head()
#   3. Basic statistics using df.describe()

# YOUR CODE HERE:


# ============================================================
# STEP 3: Feature Engineering
# ============================================================
# Raw stock prices alone aren't great features for ML.
# Instead, we calculate "technical indicators" — signals that
# traders use to analyze whether a stock is likely to go up or down.
#
# 💡 Key concept: A "feature" is any input variable we give the model.
#    The better the features, the smarter the model can be.

# NOTE: yfinance sometimes returns MultiIndex columns in Colab.
#       Run this fix before creating features:
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ── Feature 1: Daily Return ───────────────────────────────────
# How much did the stock price change today, as a percentage?
# Formula: (today's close - yesterday's close) / yesterday's close
#
# 💡 Hint: pandas has a built-in method called .pct_change()

# TODO: Calculate the daily return and store it in df["Daily_Return"]

# YOUR CODE HERE:


# ── Feature 2: Moving Averages ────────────────────────────────
# A moving average smooths out price noise over N days.
# We'll compute a 5-day MA and a 20-day MA.
#
# 💡 Hint: use df["Close"].rolling(window=N).mean()

# TODO: Calculate df["MA5"]  — 5-day moving average of Close price
# TODO: Calculate df["MA20"] — 20-day moving average of Close price
# TODO: Calculate df["MA_Ratio"] = df["MA5"] / df["MA20"]
#       (This tells us: is the short-term trend above the long-term trend?)

# YOUR CODE HERE:


# ── Feature 3: Volatility ─────────────────────────────────────
# How much has the daily return been bouncing around lately?
# Higher volatility = more uncertainty.
#
# 💡 Hint: use .rolling(window=5).std() on Daily_Return

# TODO: Calculate df["Volatility"] — 5-day rolling std of Daily_Return

# YOUR CODE HERE:


# ── Feature 4: Volume Ratio ───────────────────────────────────
# Is today's trading volume unusually high or low?
# Ratio > 1 means more activity than usual.
#
# 💡 Hint: divide today's Volume by its 20-day rolling mean

# TODO: Calculate df["Volume_Ratio"]

# YOUR CODE HERE:


# ── Feature 5: RSI (Relative Strength Index) ──────────────────
# RSI is a momentum indicator that ranges from 0 to 100.
#   RSI > 70 → stock may be overbought (price might drop)
#   RSI < 30 → stock may be oversold  (price might rise)
#
# Formula (don't worry if this looks complex — just fill it in!):
#   Step 1: delta    = df["Close"].diff()
#   Step 2: gain     = delta.clip(lower=0)
#   Step 3: loss     = -delta.clip(upper=0)
#   Step 4: avg_gain = gain.rolling(window=14).mean()
#   Step 5: avg_loss = loss.rolling(window=14).mean()
#   Step 6: rs       = avg_gain / avg_loss
#   Step 7: RSI      = 100 - (100 / (1 + rs))

# TODO: Calculate df["RSI"] using the steps above

# YOUR CODE HERE:


# ── Target Variable ───────────────────────────────────────────
# This is what we're trying to predict: did the price go UP tomorrow?
#   1 = price went UP
#   0 = price went DOWN (or stayed the same)
#
# 💡 Hint: compare df["Close"].shift(-1) with df["Close"]
#          .shift(-1) moves tomorrow's price into today's row
#          Wrap with .astype(int) to convert True/False → 1/0

# TODO: Create df["Target"]

# YOUR CODE HERE:


# ── Drop missing values ───────────────────────────────────────
# Rolling calculations leave NaN at the start. Drop them.

# TODO: Use df.dropna(inplace=True)

# YOUR CODE HERE:


print(f"Features created! Dataset has {len(df)} usable rows.")
print(f"Class balance — UP: {df['Target'].sum()} | DOWN: {(df['Target']==0).sum()}")


# ============================================================
# STEP 4: Prepare Data for the Model
# ============================================================
# We split the data into:
#   X = the features (inputs to the model)
#   y = the target   (what we want to predict)
#
# Then we split THOSE into training and test sets, and scale them.

FEATURES = ["Daily_Return", "MA_Ratio", "Volatility", "Volume_Ratio", "RSI"]

# TODO: Create X = df[FEATURES]  and  y = df["Target"]

# YOUR CODE HERE:


# TODO: Use train_test_split() to split X and y into:
#       X_train, X_test, y_train, y_test
#
# ⚠️ Important: set shuffle=False so time order is preserved!
#    (We don't want to "peek into the future" during training)

# YOUR CODE HERE:


# TODO: Create a StandardScaler and use it to scale your data.
#       - fit_transform on X_train (learns the scale FROM training data)
#       - transform on X_test      (applies the SAME scale — don't refit!)
#
# 💡 Why scale? Logistic Regression is sensitive to feature magnitude.
#    Scaling puts all features on the same playing field.

# YOUR CODE HERE:


print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")


# ============================================================
# STEP 5: Train the Model
# ============================================================
# Now for the fun part — actually training the model!
#
# 💡 Key concept: model.fit(X_train, y_train) is where the
#    "learning" happens. The model finds the best coefficients
#    to separate UP days from DOWN days.

# TODO: Create a LogisticRegression model with:
#       random_state=RANDOM_STATE, max_iter=1000

# YOUR CODE HERE:


# TODO: Train (fit) the model on your training data

# YOUR CODE HERE:


print("Model trained!")

# TODO (BONUS): Print the model's coefficients alongside feature names.
#               Positive coefficient = bullish signal
#               Negative coefficient = bearish signal

# YOUR CODE HERE:


# ============================================================
# STEP 6: Evaluate the Model
# ============================================================
# A model is only useful if we know how well it performs.
# We test it on data it has NEVER seen (the test set).
#
# 💡 Key metrics:
#    Accuracy  = % of predictions that were correct overall
#    Precision = when we predicted UP, how often were we right?
#    Recall    = of all actual UP days, how many did we catch?

# TODO: Use model.predict(X_test) to generate predictions → y_pred

# YOUR CODE HERE:


# TODO: Print the accuracy score using accuracy_score(y_test, y_pred)

# YOUR CODE HERE:


# TODO: Print the full classification report using classification_report()
#       Set target_names=["DOWN", "UP"] for readable labels

# YOUR CODE HERE:


# ── Confusion Matrix ──────────────────────────────────────────
# A confusion matrix shows:
#   True Positives  (predicted UP,   actually UP)   ✅
#   True Negatives  (predicted DOWN, actually DOWN) ✅
#   False Positives (predicted UP,   actually DOWN) ❌
#   False Negatives (predicted DOWN, actually UP)   ❌

# TODO: Calculate the confusion matrix → cm = confusion_matrix(y_test, y_pred)
# TODO: Plot it as a heatmap using seaborn's sns.heatmap()
#       Label the axes with ["Predicted DOWN", "Predicted UP"]
#                       and ["Actual DOWN",    "Actual UP"]
# TODO: Add a title and save with plt.savefig("confusion_matrix.png")

# YOUR CODE HERE:


# ── Feature Importance Chart ──────────────────────────────────
# TODO (BONUS): Plot a horizontal bar chart of model.coef_[0]
#               Color bars green if positive, red if negative.
#               Save as "feature_importance.png"

# YOUR CODE HERE:


# ============================================================
# STEP 7 (BONUS): Predict on the Most Recent Trading Day
# ============================================================
# Can you make the model predict what might happen TOMORROW?
#
# Steps:
#   1. Download the last 60 days of data for your TICKER
#   2. Re-calculate all 5 features on this fresh data
#   3. Take only the LAST row (most recent day)
#   4. Scale it using your already-fitted scaler
#   5. Call model.predict() and model.predict_proba()
#   6. Print the direction (UP/DOWN) and confidence %

# YOUR CODE HERE:


print("\n⚠️  Disclaimer: This is for educational purposes only.")
print("    Do NOT use this model to make real investment decisions.")
