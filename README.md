# 🌾 Rice Variety Classification — Cammeo vs Osmancik

## IT3091 Machine Learning Group Assignment

This project develops and evaluates machine learning models to classify individual rice grains as either **Cammeo** or **Osmancik** using morphological measurements extracted from rice grain images.

The project follows the **Guided Data Track — Agriculture** and focuses primarily on **rice variety classification**, with feature analysis used as a secondary lens to support quality-control decision making.

---

## 📌 1. Business Problem

### Business Scenario

An agricultural organisation wants to improve crop/product classification and support quality-related decisions using measured rice grain characteristics.

Manual identification of rice varieties may be time-consuming and subject to human inconsistency. A machine learning model can support the classification process by using measurable morphological characteristics of individual rice grains.

### Stakeholder

The primary stakeholder is an **agricultural quality-control organisation or rice processing facility** that needs to distinguish between Cammeo and Osmancik rice varieties.

### Primary Decision Lens

**Rice Variety Classification**

The main objective is to determine whether an individual rice grain belongs to:

- **Cammeo**
- **Osmancik**

### Secondary Decision Lens

**Feature Importance / Quality-Control Decision Support**

Feature analysis is used to investigate which measured morphological characteristics contribute most strongly to model predictions. This secondary analysis supports interpretation of the classification task and may provide useful information for quality-control decisions.

### Unit of Analysis

One **individual rice grain**.

### Machine Learning Task

This is a **supervised binary classification problem**.

### Target / Output

The target variable is:

`Class`

with two possible classes:

- `Cammeo`
- `Osmancik`

The machine learning models predict the rice variety of an individual grain from its measured morphological features.

---

## 📊 2. Dataset

The dataset contains morphological measurements of **3,810 rice grains** belonging to the Cammeo and Osmancik varieties.

### Dataset Features

| Feature | Description |
|---|---|
| `Area` | Number of pixels within the boundary of the rice grain |
| `Perimeter` | Circumference of the rice grain |
| `Major_Axis_Length` | Length of the longest axis of the grain |
| `Minor_Axis_Length` | Length of the shortest axis of the grain |
| `Eccentricity` | Measure describing the shape/elongation of the grain |
| `Convex_Area` | Area of the smallest convex boundary containing the grain |
| `Extent` | Ratio of grain area to its bounding-box area |
| `Class` | Rice variety: Cammeo or Osmancik |

### Class Distribution

The dataset contains approximately:

- **57.2% Osmancik**
- **42.8% Cammeo**

The imbalance is moderate rather than extreme. Therefore, a **stratified train/test split** was used to preserve the original class proportions.

---

## 👥 3. Group Responsibilities

| Member | Main Responsibility | Tasks |
|---|---|---|
| Member 1 | Data Understanding & EDA | Dataset understanding, data-quality investigation, distributions, class balance, correlations and exploratory visualisations |
| Member 2 | Preprocessing & Pipeline Design | Missing-value and duplicate verification, stratified train/test split, IQR-based outlier capping, StandardScaler transformation and target encoding |
| Member 3 | Model Building | Logistic Regression baseline and alternative models including Decision Tree, Random Forest, SVM and KNN |
| Member 4 | Evaluation, Comparison & Recommendation | Evaluation metrics, confusion matrices, ROC-AUC, train/test comparison, model comparison, limitations and recommendation |

---

## 🔄 4. Machine Learning Workflow

The project follows the workflow below:

```text
Business Problem
       ↓
Dataset Understanding
       ↓
Exploratory Data Analysis
       ↓
Data Quality Assessment
       ↓
Stratified 80/20 Train/Test Split
       ↓
Training-Only Preprocessing
       ↓
IQR-Based Outlier Capping
       ↓
StandardScaler
       ↓
Model Building
       ↓
Logistic Regression Baseline
       ↓
Decision Tree / Random Forest / SVM / KNN
       ↓
Model Evaluation & Comparison
       ↓
Critical Interpretation
       ↓
Recommendation + Limitations
```

---

## 📓 5. Notebook Execution Order

The project is divided into multiple notebooks so that each stage of the machine learning workflow is clearly separated.

Run the notebooks in the following order:

1. **Data Understanding & EDA**
2. **Train/Test Split & Preprocessing**
3. **Model Building**
4. **Evaluation, Comparison & Recommendation**

The notebooks together form one complete machine learning workflow.

---

## 🔍 6. Exploratory Data Analysis

Exploratory Data Analysis was performed before model development to understand:

- dataset structure;
- variable types;
- missing values;
- duplicate observations;
- class distribution;
- feature distributions;
- possible outliers;
- relationships between numerical features;
- relationships between features and rice variety;
- correlations among morphological measurements.

The EDA showed that several size-related measurements are correlated. This is considered when interpreting model-based feature importance.

The dataset contained no missing values or duplicate records requiring additional treatment.

---

## 🧹 7. Preprocessing & Data Leakage Prevention

Preprocessing was designed carefully to reduce the risk of **data leakage**.

### 7.1 Train/Test Split

The dataset was first divided using an:

**80% training / 20% testing stratified split**

with:

```python
random_state = 42
```

Stratification preserves approximately the same Cammeo/Osmancik class distribution in both partitions.

### 7.2 Missing Values

Missing values were checked during data exploration.

No missing values requiring imputation were identified.

Therefore, unnecessary imputation was avoided.

### 7.3 Duplicate Records

The dataset was checked for duplicate observations.

No duplicate records requiring removal were identified.

### 7.4 Outlier Handling

Potential outliers were investigated using the **Interquartile Range (IQR)**.

For each numerical feature:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Values outside these boundaries were capped at the corresponding lower or upper boundary.

Importantly, the outlier boundaries were calculated **using the training data only**.

The same training-derived boundaries were subsequently applied to the test data.

This prevents information from the test set influencing preprocessing decisions.

Outlier capping was selected instead of simply deleting extreme observations because extreme measurements may represent genuine variation in rice grain morphology. Capping limits their influence while retaining observations.

A limitation of this approach is that capping modifies genuine extreme measurements and may therefore reduce some information contained in those observations.

### 7.5 Feature Scaling

The project uses:

**`StandardScaler`**

The scaler was fitted only on the training features:

```python
scaler.fit_transform(X_train)
```

The already fitted scaler was then used to transform the test features:

```python
scaler.transform(X_test)
```

This is important because several models used in this project, particularly Logistic Regression, SVM and KNN, can be affected by differences in feature scale.

Fitting the scaler exclusively on the training data prevents information from the test set from leaking into model training.

### 7.6 Target Encoding

The rice variety target was encoded numerically for model training.

The encoding used is:

```text
Cammeo   → 0
Osmancik → 1
```

---

## 💾 8. Processed Data

The preprocessing stage produces separate processed training and testing datasets:

```text
train_processed.csv
test_processed.csv
```

These datasets are used by the subsequent model-building and evaluation stages.

The original dataset is retained separately so that the transformation from raw data to processed data remains reproducible and transparent.

---

## 🤖 9. Machine Learning Models

Five classification algorithms were investigated.

### 9.1 Logistic Regression

**Role:** Baseline model

Logistic Regression provides a relatively simple and interpretable baseline for the binary classification task.

It allows more complex models to be compared against a straightforward linear classifier.

### 9.2 Decision Tree

Decision Tree was included because it can represent nonlinear decision rules and provides an interpretable tree-based approach.

### 9.3 Random Forest

Random Forest was included as an ensemble tree-based method capable of modelling nonlinear relationships and interactions among features.

### 9.4 Support Vector Machine — RBF Kernel

SVM with an RBF kernel was included because the standardized feature space allows the RBF kernel to model potentially nonlinear class boundaries while reducing the effect of differences in feature scale.

### 9.5 K-Nearest Neighbours

KNN was included as an instance-based classification method.

Because KNN depends strongly on distances between observations, feature scaling is particularly important.

---

## 📏 10. Evaluation Metrics

Model performance is evaluated using multiple metrics rather than relying on accuracy alone.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Training Accuracy
- Test Accuracy
- Train/Test Generalisation Gap

Using several metrics provides a more complete assessment of classification performance.

### F1-Score Interpretation

The current binary F1-score uses Osmancik (`1`) as the positive class.

Because both rice varieties are important to the classification task, **Macro-F1 is also being considered as an additional evaluation measure** so that both classes receive equal weight.

---

## 📈 11. Current Holdout Results

The current results obtained from the stratified holdout test set are:

| Model | Test Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 0.9173 | 0.9287 |
| Random Forest | 0.9160 | 0.9281 |
| Decision Tree | 0.9121 | 0.9229 |
| SVM | 0.9108 | 0.9233 |
| KNN | 0.9094 | 0.9217 |

These results show that the candidate models perform relatively similarly on the current holdout partition.

Therefore, small differences in test performance should not by themselves be interpreted as strong evidence that one model is universally superior.

---

## ⚖️ 12. Generalisation Analysis

Training and test performance were compared to identify possible overfitting.

For example:

### Logistic Regression

```text
Training Accuracy ≈ 0.932
Test Accuracy     ≈ 0.917
Generalisation Gap ≈ 0.015
```

### Random Forest

```text
Training Accuracy = 1.000
Test Accuracy     ≈ 0.916
Generalisation Gap ≈ 0.084
```

Random Forest therefore shows stronger evidence of overfitting on the current split than Logistic Regression.

This does **not** mean that Random Forest is inherently unsuitable. Instead, it demonstrates the importance of considering generalisation, model complexity and validation evidence rather than training accuracy alone.

---

## 🔁 13. Cross-Validation — Planned Improvement

The current results are based primarily on a single stratified train/test partition.

To strengthen model-selection reliability, the next evaluation stage will incorporate:

**Stratified 5-Fold Cross-Validation**

on the training data.

The intended validation workflow is:

```text
Original Dataset
       ↓
80/20 Stratified Split
       ↓
Training Data
       ↓
Stratified 5-Fold Cross-Validation
       ↓
Compare Candidate Models
       ↓
Select / Finalise Model
       ↓
Fit on Full Training Data
       ↓
Evaluate on Untouched Test Set
       ↓
Final Recommendation
```

Cross-validation results will be used to assess both average performance and variability across folds.

The final test set will remain separate from model-selection decisions and will be used for final evaluation.

> **Note:** Cross-validation is a planned improvement and should not be interpreted as completed until the corresponding implementation and results are added to the repository.

---

## 🌾 14. Feature Importance & Interpretation

Feature-importance analysis is used as a secondary decision lens to investigate which morphological characteristics contribute strongly to model predictions.

Current Random Forest feature-importance analysis indicates that features such as **Major_Axis_Length** and **Eccentricity** contribute strongly to predictions.

However, several size-related variables in the dataset are correlated.

Therefore, Random Forest importance values should be interpreted as **model-based associations**, rather than evidence that an individual feature independently causes a rice grain to belong to a particular variety.

Feature-importance conclusions are consequently used as supporting information rather than causal claims.

---

## 🏆 15. Current Model Recommendation

Based on the current single holdout test split, **Logistic Regression is the provisional candidate model**.

It currently provides:

- competitive test accuracy;
- competitive F1-score;
- strong ROC-AUC;
- a relatively small train/test generalisation gap;
- comparatively straightforward interpretation;
- lower model complexity than some alternatives.

However, the difference between Logistic Regression and Random Forest on the current holdout F1-score is very small.

Therefore, Logistic Regression should **not yet be considered the final model solely because it achieved the highest score on this single test partition**.

The final model recommendation will be reconsidered after the planned cross-validation analysis.

---

## ⚠️ 16. Limitations

Several limitations should be considered when interpreting the results.

### Single Train/Test Partition

Current model comparison is based primarily on one stratified holdout split.

Although the split preserves class proportions, performance can vary depending on the selected observations.

Stratified cross-validation is therefore planned to provide stronger evidence for model comparison.

### Dataset Environment

The dataset contains morphological measurements obtained from images of individual rice grains under controlled conditions.

Real agricultural or industrial environments may include:

- overlapping grains;
- damaged grains;
- dust or other objects;
- different lighting conditions;
- different cameras or sensors;
- additional rice varieties.

Therefore, performance on this dataset does not automatically guarantee equivalent performance in a real production environment.

### Outlier Capping

IQR-based capping reduces the influence of extreme observations but modifies their original measurements.

Some extreme values may represent genuine morphological variation.

### Feature Importance

Correlated features can affect model-based feature-importance estimates.

Feature importance should therefore not be interpreted as causal evidence.

### Model Tuning

Further hyperparameter optimisation may improve some candidate models.

However, tuning should be performed using appropriate validation procedures rather than repeatedly evaluating configurations on the final test set.

---

## 💡 17. Stakeholder Value

A reliable classification model could support agricultural quality-control processes by providing a consistent data-driven method for distinguishing Cammeo and Osmancik rice grains.

Potential value includes:

- supporting automated variety identification;
- reducing dependence on subjective manual classification;
- improving consistency;
- supporting quality-control decisions;
- identifying useful morphological patterns.

However, additional real-world validation would be necessary before deploying the model in an operational environment.

---

## 🧠 18. Responsible AI Considerations

This project treats machine learning as **decision support** rather than assuming that model predictions are automatically correct.

Important responsible-use considerations include:

- model predictions can be incorrect;
- dataset conditions may not represent all real-world environments;
- model performance should be monitored on new data;
- predictions should not be generalized to rice varieties not represented in the training data;
- feature importance should not be interpreted as causal evidence;
- preprocessing and evaluation decisions should be documented transparently;
- deployment should only occur after appropriate real-world validation.

---

## 📝 19. Decision Logging

Important project decisions are documented with consideration of:

- alternatives considered;
- selected approach;
- reason for the decision;
- supporting evidence;
- limitations where applicable.

Major decisions include:

- selection of variety classification as the primary lens;
- definition of the individual grain as the unit of analysis;
- use of supervised binary classification;
- use of stratified train/test splitting;
- training-only calculation of preprocessing parameters;
- IQR-based outlier capping;
- use of StandardScaler;
- selection of Logistic Regression as the baseline;
- comparison with four alternative models;
- use of multiple evaluation metrics;
- assessment of generalisation gaps;
- planned use of stratified cross-validation;
- cautious interpretation of feature importance.

A consolidated decision log can be maintained separately from the notebooks so that the reasoning behind the technical workflow remains transparent.

---

## 🤖 20. AI Use Transparency

AI tools may be used to support activities such as:

- clarification of machine learning concepts;
- debugging assistance;
- review of code or documentation;
- improvement of written explanations;
- identification of possible methodological issues.

All final project decisions, code, outputs and interpretations should be reviewed and understood by the group members.

AI-generated suggestions should not be treated as experimental evidence. Final claims and recommendations must be supported by the project's dataset, implemented code and observed results.

---

## ♻️ 21. Reproducibility

The repository is organised so that the machine learning workflow can be followed from the original dataset through preprocessing, modelling and evaluation.

For reproducibility:

1. Use the original rice dataset included/referenced by the project.
2. Run the EDA notebook.
3. Run the preprocessing notebook.
4. Generate the processed training and test datasets.
5. Run the model-building notebook.
6. Run the evaluation/comparison notebook.
7. Confirm that generated results match those documented in the final report.

The project uses a fixed:

```python
random_state = 42
```

where applicable to improve reproducibility.

---

## 📂 22. Repository Structure

The repository is organised broadly as follows:

```text
2026-DS-04K/
│
├── Rice_Cammeo_Osmancik.csv
│
├── Data_Understanding_and_EDA.ipynb
│
├── Evaluation_+_Comparison_+_Recommendation.ipynb
│
├── README.md
│
├── preprocessing/
│   ├── Train_Test_Split_+_Preprocessing.ipynb
│   ├── train_processed.csv
│   ├── test_processed.csv
│   └── label_encoder.pkl
│
├── model_building/
│   └── Modeling.ipynb
│
└── visualizations/
    └── ...
```

> Repository contents may be extended as the project progresses, including the consolidated decision log and cross-validation results.

---

## 📌 23. Current Project Status

### Completed

- ✅ Business problem and primary lens identified
- ✅ Dataset understanding
- ✅ Exploratory Data Analysis
- ✅ Missing-value verification
- ✅ Duplicate verification
- ✅ Class-distribution analysis
- ✅ Stratified train/test split
- ✅ Leakage-aware preprocessing
- ✅ IQR-based outlier capping
- ✅ StandardScaler feature scaling
- ✅ Target encoding
- ✅ Logistic Regression baseline
- ✅ Decision Tree
- ✅ Random Forest
- ✅ SVM
- ✅ KNN
- ✅ Holdout evaluation
- ✅ Confusion-matrix analysis
- ✅ ROC-AUC evaluation
- ✅ Train/test generalisation comparison
- ✅ Preliminary feature-importance analysis
- ✅ Preliminary recommendation and limitations

### To Be Completed / Strengthened

- ⏳ Stratified 5-Fold Cross-Validation
- ⏳ Macro-F1 comparison
- ⏳ Final model selection after cross-validation
- ⏳ Consolidated Decision Log
- ⏳ Final workflow diagram
- ⏳ Dataset source/fingerprint documentation
- ⏳ Final report
- ⏳ 3-minute demonstration video
- ⏳ Individual Personal Learning Journey reports

---

## 🎯 24. Final Goal

The goal of this project is not simply to obtain the highest possible accuracy.

The project aims to make a **defensible machine learning decision** by combining:

- clear business problem framing;
- appropriate data understanding;
- leakage-aware preprocessing;
- justified model selection;
- multiple evaluation measures;
- validation;
- interpretability;
- critical judgement;
- responsible AI considerations;
- practical stakeholder value.

The final recommendation will therefore consider not only predictive performance, but also **generalisation, model complexity, interpretability, limitations and practical usefulness**.
