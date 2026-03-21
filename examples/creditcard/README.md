# Credit Card Fraud Detection

[Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) with 284k transactions, 30 PCA-derived continuous features. Highly imbalanced: only 492 fraud cases. The script keeps all fraud samples and subsamples 5,000 normal transactions for a 20x20 hexagonal SOM.

## Run

```bash
uv run python examples/creditcard/run.py
```

Data is downloaded automatically on first run via `prepare.py`.
