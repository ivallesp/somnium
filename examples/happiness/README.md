# World Happiness Report

[Kaggle dataset](https://www.kaggle.com/datasets/mathurinache/world-happiness-report) combining reports from 2015 to 2022. Each country-year observation has 6 features: GDP, social support, health, freedom, generosity, and corruption. Column names are harmonized across years, yielding ~1,200 samples from 195 countries. Trains a 15x10 hexagonal SOM.

## Run

```bash
uv run python examples/happiness/run.py
```

Data is downloaded automatically on first run via `prepare.py`.
