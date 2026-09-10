# Nova Academy - Course Cancellation Prediction

End-to-end machine learning pipeline for predicting the probability that a registered course order will be cancelled before it begins. Final project for Introduction to Machine Learning at Tel Aviv University.

**Authors:** Amit Masel, Esti Binyamin

## Business Problem

Nova Academy is a fictional B2B education company. Course cancellations before start date result in lost revenue and operational overhead. The goal: build a model that predicts cancellation probability from registration data, enabling proactive retention actions.

**Success criterion:** AUC ≥ 0.70 on held-out test set.

## Methodology

Full CRISP-DM process:
1. Business Understanding
2. Data Understanding & EDA
3. Data Preparation & Feature Engineering
4. Modeling
5. Evaluation
6. Deployment considerations

## Models Trained

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- Multi-Layer Perceptron (MLP)
- **Stacking Ensemble** (final model)

## Results

- **Out-of-fold AUC: ~0.955** - significantly exceeding the 0.70 success criterion
- Extensive feature engineering contributed materially to performance
- Stacking ensemble outperformed all individual base learners

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Google Colab

## How to Run

Open the notebook directly in Colab using the badge at the top of the file, or download and run locally with Jupyter.
