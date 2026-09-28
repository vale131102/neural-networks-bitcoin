[README_neural network.md](https://github.com/user-attachments/files/32767790/README_neural.network.md)
# neural-networks-bitcoin
Forecasts Bitcoin log returns with CNN and LSTM models, combining hourly market data, CryptoBERT tweet sentiment, and the Fear &amp; Greed Index (2019). 📄 README in English, remaining materials in Italian. 📩 For access to the large datasets and knime workflow contact me via email (vale131102@gmail.com) or LinkedIn (link below).
# Bitcoin Log Return Analysis with CNN and LSTM

A Machine Learning project that attempts to analyze **Bitcoin log returns** using Deep Learning architectures, combining market data with **social sentiment and engagement metrics**.

## Objective

The cryptocurrency market is characterized by high volatility, non-stationarity and strong non-linearity. Unlike traditional assets, Bitcoin's price is heavily influenced by social narrative and collective sentiment, and classical statistical models struggle to capture these dependencies and filter out the noise.

The goal of this project is to develop, evaluate and compare two architectures:

- **CNN** (Convolutional Neural Network)
- **LSTM** (Long Short-Term Memory)

Both models process a heterogeneous set of inputs:

- historical log returns
- logarithm of trading volume
- Garman-Klass volatility
- user sentiment and engagement metrics (tweets)
- Crypto Fear & Greed Index

Predictive performance is evaluated over three time horizons: **1 hour**, **1 day** and **1 week**.

## Repository Structure

| File | Description |
|---|---|
| `script_ML_completo.ipynb` | Python notebook with the full preprocessing: price aggregation, tweet filtering, sentiment analysis with CryptoBERT, dataset merging and volatility computation |
| `Workflow.knwf` | KNIME workflow with the final preprocessing, lag construction, neural network training (CNN and LSTM) and evaluation (not included, too large: **ASK FOR IT**) |
| `dataset_knime.csv` | Final dataset produced by the notebook, used as input for the KNIME workflow |
| `fear_and_greed_index.csv` | Daily Crypto Fear & Greed Index |
| `dataset_sin_spam.csv` | Tweet dataset (**not included**, too large: see Datasets section) |
| `btcusd_1-min_data.csv` | 1-minute Bitcoin historical data (**not included**, too large: see Datasets section) |

## Datasets

The data comes from three public Kaggle datasets.

### 1. Bitcoin historical market data (`btcusd_1-min_data.csv`)
High-frequency time series with open, high, low, close (OHLC) prices and trading volumes for BTC/USD.
Link: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data

### 2. Crypto Fear & Greed Index (`fear_and_greed_index.csv`)
Daily index built from several factors (volatility, momentum/volume, social media, dominance, trends) that measures public interest and perception. Data comes from [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/).
Link: https://www.kaggle.com/datasets/adelsondias/crypto-fear-and-greed-index

> **Note:** in the original file the `date` and `fng_value` column names are swapped, dates are in `DD-MM-YYYY` format, and rows are sorted from most recent to oldest. The notebook handles all of this.

### 3. Bitcoin Twitter Sentiment Dataset 2013–2023 (`dataset_sin_spam.csv`)
Over 39 million spam-free tweets. Five columns are used: `date`, `n_replies`, `n_likes`, `n_retweets`, `text`.
Link: https://www.kaggle.com/datasets/andreapenasmartinez/bitcoin-twitter-sentiment-dataset-20132023?select=dataset_sin_spam.csv

### Datasets not included

`btcusd_1-min_data.csv` and `dataset_sin_spam.csv` are not included in the repository because of their size. To reproduce the project, download them from the Kaggle links above and place them in the same folder as the notebook.

### Final dataset (`dataset_knime.csv`)

Hourly dataset with 7,840 rows and 8 columns:

| Column | Description |
|---|---|
| `date` | Date and time (hourly frequency) |
| `avg_weighted_sentiment` | Average weighted sentiment for the hour |
| `tweet_volume` | Number of tweets in the hour |
| `total_engagement` | Sum of likes, retweets and replies in the hour |
| `Close` | Hourly closing price |
| `Volume` | Hourly trading volume |
| `F&G_value` | Fear & Greed Index value for the day |
| `vol_garman_klass` | Garman-Klass volatility |

## Pipeline

### Part 1: Python preprocessing (`script_ML_completo.ipynb`)

**Bitcoin prices**
1. Conversion of the timestamp to datetime format.
2. Aggregation from 1-minute to **hourly** data (Open: first, High: max, Low: min, Close: last, Volume: sum).
3. Selection of the period from **01/01/2019** to **23/11/2019 (15:00)**, the limit beyond which social data shows a gap of 920 consecutive hours (38 days).

**Tweets and sentiment**
1. Chunked reading (100,000 rows per chunk) of `dataset_sin_spam.csv`, excluding tweets before 02/01/2018.
2. Engagement filter: tweets with zero `n_likes`, `n_replies` or `n_retweets` are discarded (510,073 tweets remain).
3. Classification with **CryptoBERT** (`ElKulako/cryptobert`), a Transformer model pre-trained on crypto-related Twitter language, which returns a class (Bullish, Neutral, Bearish) and a confidence score. Inference runs on GPU (if available) with batches of 64 tweets.
4. Weighted sentiment computation:
   - sign: Bullish = 1, Neutral = 0, Bearish = -1
   - engagement weight: `Weight = ln(n_likes + n_retweets + n_replies)`
   - `weighted_sentiment = Sign × Confidence × Weight`
5. Hourly aggregation: average weighted sentiment (`avg_weighted_sentiment`), tweet count (`tweet_volume`) and total engagement (`total_engagement`).
6. Missing values: linear interpolation for sentiment (maximum gap of 3 consecutive hours), zero-filling for volume and engagement.

**Dataset merging**
1. Merge of hourly prices and sentiment on the time key.
2. The daily Fear & Greed Index is replicated across the 24 hours of each day and merged with the hourly data.
3. Computation of **Garman-Klass volatility** from OHLC to capture intra-hour dynamics.
4. Removal of redundant columns (`Open`, `High`, `Low`, `F&G_classification`) and export to `dataset_knime.csv`.

### Part 2: KNIME workflow (`Workflow.knwf`)

1. **Series transformation**: computation of price log returns to ensure stationarity.
2. **Regularization**: `ln(x + 1)` transformation on `Volume`, `tweet_volume` and `total_engagement`.
3. **Lags**: for each horizon (1 hour, 1 day, 1 week), lagged variables are created for all input features.
4. **Cleaning**: removal of null values generated by lags and selection of relevant columns.
5. **Temporal split** into training and test sets (*Table Partitioner* node).
6. **Min-Max normalization** [0, 1] fitted on the training set only and then applied to the test set, to avoid data leakage. Predictions are then denormalized.
7. **Models**:
   - **CNN**: two Conv1D layers, one Flatten layer and two Dense layers.
   - **LSTM**: one Reshape layer, one LSTM layer and two Dense layers.
8. **Evaluation**: comparison of predicted and actual values with the *Numeric Scorer* and *Line Plot* nodes.

## Requirements

**Python** (notebook)
- `pandas`, `numpy`, `tqdm`
- `transformers`, `torch` (a CUDA GPU is recommended for sentiment analysis)

```bash
pip install pandas numpy tqdm transformers torch
```

**KNIME**
- KNIME Analytics Platform with the **KNIME Deep Learning Integration** (Keras) extension, required for the *Keras Network Learner/Executor* nodes.

## How to Reproduce

1. Download `btcusd_1-min_data.csv` and `dataset_sin_spam.csv` (and `fear_and_greed_index.csv` if needed) from Kaggle and place them in the same folder as the notebook.
2. Run `script_ML_completo.ipynb` to obtain `dataset_knime.csv`. The notebook also generates intermediate files: `btcusd_2019_orario.csv`, `dataset_filtrato.csv`, `dataset_con_sentiment.csv`, `sentiment_2019_orario.csv`.
3. Import `Workflow.knwf` into KNIME, update the *CSV Reader* node path to point to `dataset_knime.csv`, and run the workflow.

> To skip the sentiment analysis step (the most computationally expensive one), you can start directly from `dataset_knime.csv` and go to step 3.
