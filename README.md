# PhaseShift

A machine learning classifier that predicts whether a CRISPR clinical trial will advance beyond Phase 1. Built using a random forest model trained on a structured dataset of ~250 CRISPR clinical trials scraped from public sources.

## About

PhaseShift was developed to identify which features of a CRISPR clinical trial — such as disease category, gene-editing method, enrollment size, and trial age — are most predictive of phase advancement. The model uses 5-fold stratified cross-validation and achieves strong performance across accuracy, precision, recall, and F1 score.

This project was submitted to the Golden Gate STEM Fair and is being written up for the Journal of Emerging Investigators (JEI).

## How to Read This Repo (Start Here)

Go through the files in this order:

1. **`data_collection.py`** — Scrapes CRISPR trial pages and uses GPT-4o-mini to extract structured data (trial phase, disease category, enrollment, countries, etc.) into a dataset
2. **`preprocessing.py`** — Cleans the raw data, engineers features (trial age, number of countries, label encoding), and builds the target variable (did the trial advance past Phase 1?)
3. **`model.py`** — Trains the PhaseShift random forest classifier with 5-fold cross-validation and prints performance metrics
4. **`visualize.py`** — Generates the feature importance bar chart and confusion matrix

## Files

| File | Description |
|------|-------------|
| `data_collection.py` | Web scraping + GPT-4o-mini extraction pipeline |
| `preprocessing.py` | Data cleaning and feature engineering |
| `model.py` | Random forest model training and evaluation |
| `visualize.py` | Matplotlib/Seaborn charts |
| `CRISPR_Data.csv` | Final structured dataset of ~250 CRISPR clinical trials |
