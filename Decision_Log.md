# 📋 Decision Log

## Rice Variety Classification — Cammeo vs Osmancik

**Module:** IT3091 Machine Learning  
**Track:** Guided Data Track  
**Domain:** Agriculture  
**Primary Decision Lens:** Rice Variety Classification  
**Secondary Decision Lens:** Feature Importance / Quality-Control Decision Support  

---

## 1. Purpose

This decision log records the major decisions made throughout the machine learning project.

For each decision, the alternatives considered, selected approach, reasoning, supporting evidence, and current status are documented. The purpose of maintaining this log is to ensure that the machine learning workflow remains transparent, reproducible, and evidence-based.

This document will continue to be updated as the project progresses and additional validation evidence becomes available.

---

## 2. Project Decision Log

| ID | Decision Area | Alternatives Considered | Selected Decision | Reason / Evidence | Status |
|---|---|---|---|---|---|
| D01 | Primary Business Lens | Variety classification, feature-importance analysis, quality-control decision support | **Variety classification** | The main business problem requires distinguishing between Cammeo and Osmancik rice grains using measured morphological characteristics. Feature analysis supports this primary classification objective. | ✅ Completed |
| D02 | Secondary Decision Lens | No secondary lens, feature-importance analysis, broader quality analysis | **Feature importance / quality-control decision support** | Feature analysis can help identify morphological measurements that contribute strongly to model predictions and can support interpretation of the primary classification task. | ✅ Completed |
| D03 | Unit of Analysis | Rice batch, individual rice grain | **Individual rice grain** | Each row of the dataset represents measurements obtained from an individual rice grain. Therefore, the individual grain is the appropriate unit for prediction. | ✅ Completed |
| D04 | Machine Learning Task | Regression, clustering, classification | **Supervised binary classification** | The target variable contains two known rice varieties, Cammeo and Osmancik. Therefore, supervised binary classification is appropriate. | ✅ Completed |
| D05 | Train/Test Strategy | Random split, stratified split | **80/20 stratified train/test split with `random_state=42`** | The dataset has an approximately 57.2% Osmancik and 42.8% Cammeo class distribution. Stratification preserves approximately the same class proportions in the training and test sets. | ✅ Completed |
| D06 | Missing-Value Handling | Imputation, row deletion, no treatment | **No treatment required** | Data-quality analysis identified no missing values requiring imputation or deletion. Applying unnecessary imputation could alter the original data without providing a benefit. | ✅ Completed |
| D07 | Duplicate Handling | Remove duplicates, retain observations | **No duplicate treatment required** | Duplicate checking identified no duplicate records requiring removal. | ✅ Completed |
| D08 | Outlier Handling | Delete outliers, retain unchanged, IQR-based capping | **IQR-based outlier capping** | Extreme observations may represent genuine variation in rice grain morphology. Capping reduces the influence of extreme measurements without deleting observations. The lower and upper bounds are calculated using the training data only and subsequently applied to the test data. | ✅ Completed |
| D09 | Feature Scaling | No scaling, MinMaxScaler, StandardScaler | **StandardScaler** | The numerical features have different measurement ranges. Scaling is particularly relevant for models such as Logistic Regression, SVM and KNN. `StandardScaler` is fitted only on the training data and then applied to the test data to prevent leakage. | ✅ Completed |
| D10 | Target Encoding | Retain text labels, manual numerical mapping, LabelEncoder | **Numerical target encoding** | Machine learning models require the categorical target to be represented numerically. Cammeo is encoded as `0` and Osmancik as `1`. | ✅ Completed |
| D11 | Baseline Model | Logistic Regression, Decision Tree, other classifiers | **Logistic Regression** | Logistic Regression provides a relatively simple and interpretable baseline for the binary classification task. It allows the performance of more complex models to be compared against a straightforward classifier. | ✅ Completed |
| D12 | Alternative Models | Decision Tree, Random Forest, SVM, KNN and other classifiers | **Decision Tree, Random Forest, SVM (RBF) and KNN** | These alternatives represent different modelling approaches: tree-based, ensemble, kernel-based and instance-based classification. This provides a broader comparison than relying on a single algorithm. | ✅ Completed |
| D13 | Evaluation Metrics | Accuracy only, precision, recall, F1-score, ROC-AUC | **Multiple evaluation metrics** | Accuracy alone does not provide a complete assessment of classification performance. Precision, recall, F1-score, ROC-AUC and confusion matrices provide a broader evaluation of model behaviour. | ✅ Completed |
| D14 | Overfitting Assessment | Test performance only, training/test comparison | **Train/test generalisation gap** | Comparing training and test performance helps identify models that fit the training data substantially better than unseen data and therefore may be overfitting. | ✅ Completed |
| D15 | Feature-Importance Analysis | No feature analysis, model-based feature importance | **Random Forest feature importance** | Model-based importance is used to investigate which morphological features contribute strongly to model predictions. Because several size-related features are correlated, the importance values are interpreted as model-based associations rather than independent causal effects. | ✅ Completed |
| D16 | Validation Strategy | Single holdout split, stratified cross-validation | **Stratified 5-Fold Cross-Validation on training data** | Current model comparison is primarily based on a single holdout partition. Stratified cross-validation is planned to provide stronger evidence about average model performance and variability across folds while preserving class proportions. | ⏳ Planned |
| D17 | Final Model Selection | Logistic Regression, Decision Tree, Random Forest, SVM, KNN | **Pending cross-validation results** | Logistic Regression currently performs strongly on the holdout test set, but the difference between some candidate models is small. The final model should therefore not be selected solely from one test partition. | ⏳ Pending |

---

## 3. Detailed Preprocessing Decisions

### 3.1 Train/Test Split

An **80/20 stratified train/test split** was selected.

The split uses:

```python
random_state = 42
```

Stratification was chosen because the target classes are not perfectly balanced. The original dataset contains approximately:

- **57.2% Osmancik**
- **42.8% Cammeo**

Using a stratified split helps preserve approximately the same class proportions in both the training and test datasets.

The train/test split is performed **before learned preprocessing operations** to reduce the risk of data leakage.

---

### 3.2 Missing Values

The dataset was checked for missing values during the data-understanding and preprocessing stages.

No missing values requiring treatment were identified.

Therefore, no imputation or row deletion was performed.

**Decision:** No missing-value treatment required.

---

### 3.3 Duplicate Records

The dataset was checked for duplicate observations.

No duplicate records requiring removal were identified.

**Decision:** No duplicate-removal operation required.

---

### 3.4 Outlier Handling

Potential outliers were investigated using the **Interquartile Range (IQR)** method.

For each numerical feature:

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Values outside these boundaries are capped at the corresponding lower or upper boundary.

The important leakage-prevention decision is that these boundaries are calculated using **training data only**.

The training-derived boundaries are then applied to both the training and test datasets.

**Alternatives considered:**

1. Delete outliers
2. Leave all extreme values unchanged
3. Cap extreme values using IQR-derived boundaries

**Selected approach:** IQR-based capping.

**Reasoning:** Extreme measurements may represent genuine rice grain morphology rather than incorrect observations. Deleting these records could unnecessarily remove useful data. Capping limits the influence of extreme measurements while retaining the observations.

**Limitation:** Capping changes genuine extreme measurements and may reduce some information contained in those observations.

---

### 3.5 Feature Scaling

The project uses:

```python
StandardScaler
```

The scaler is fitted on the training features only:

```python
scaler.fit_transform(X_train)
```

The fitted scaler is subsequently used to transform the test features:

```python
scaler.transform(X_test)
```

**Alternatives considered:**

- No scaling
- MinMaxScaler
- StandardScaler

**Selected approach:** StandardScaler.

**Reasoning:** The morphological features have different numerical ranges. Algorithms such as Logistic Regression, SVM and KNN can be affected by differences in feature scale.

Fitting the scaler exclusively on the training data also prevents information from the test data from influencing the learned scaling parameters.

---

## 4. Model Selection Decisions

### 4.1 Baseline Model

**Selected baseline:** Logistic Regression.

Logistic Regression was selected as the baseline because the project involves binary classification and Logistic Regression provides a relatively simple and interpretable reference model.

More complex models can therefore be evaluated against this baseline.

---

### 4.2 Alternative Models

Four alternative models were selected:

1. **Decision Tree**
2. **Random Forest**
3. **Support Vector Machine (RBF Kernel)**
4. **K-Nearest Neighbours**

These models represent different machine learning approaches.

### Decision Tree

Included as an interpretable tree-based model capable of representing nonlinear decision boundaries.

### Random Forest

Included as an ensemble tree-based method capable of modelling nonlinear relationships and interactions among features.

### SVM — RBF Kernel

Included because the RBF kernel can model potentially nonlinear class boundaries in the standardized feature space.

Because the RBF kernel depends on distances between observations, feature scaling is important.

### K-Nearest Neighbours

Included as an instance-based classification approach.

Because KNN relies directly on distances between observations, standardized features are important for preventing features with larger numerical ranges from dominating the distance calculation.

---

## 5. Evaluation Decisions

### 5.1 Multiple Evaluation Metrics

The models are evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Training Accuracy
- Test Accuracy
- Train/Test Generalisation Gap

**Decision:** Do not rely on accuracy alone.

**Reasoning:** Different metrics provide different information about model behaviour. Using several measures allows a more complete and defensible comparison of the candidate models.

---

### 5.2 F1-Score

The current binary target encoding is:

```text
Cammeo   → 0
Osmancik → 1
```

Therefore, the default binary F1-score treats **Osmancik as the positive class**.

Because correct classification of both varieties is relevant to the business problem, **Macro-F1 is being considered as an additional evaluation measure**.

Macro-F1 would give equal weight to the performance of both rice varieties.

**Status:** To be strengthened during the validation stage.

---

### 5.3 Generalisation and Overfitting

Training and test performance are compared to investigate possible overfitting.

Current results show that some models have larger train/test performance differences than others.

For example, the current results indicate approximately:

```text
Logistic Regression
Training Accuracy ≈ 0.932
Test Accuracy     ≈ 0.917
Gap               ≈ 0.015
```

while:

```text
Random Forest
Training Accuracy = 1.000
Test Accuracy     ≈ 0.916
Gap               ≈ 0.084
```

The larger Random Forest gap provides evidence of stronger overfitting behaviour on the current train/test split.

This does not mean that Random Forest is inherently unsuitable. Instead, it demonstrates why training performance alone should not determine the final model.

---

## 6. Current Holdout Model Comparison

The current holdout test results are approximately:

| Model | Test Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 0.9173 | 0.9287 |
| Random Forest | 0.9160 | 0.9281 |
| Decision Tree | 0.9121 | 0.9229 |
| SVM | 0.9108 | 0.9233 |
| KNN | 0.9094 | 0.9217 |

The current results show relatively similar performance among several candidate models.

In particular, the F1-score difference between Logistic Regression and Random Forest on this test split is very small.

Therefore, the project will **not treat this single holdout comparison as sufficient evidence for final model selection**.

---

## 7. Feature-Importance Decision

Feature importance is included as a secondary analysis to support interpretation of the classification task.

Current Random Forest feature-importance analysis indicates that features such as:

- `Major_Axis_Length`
- `Eccentricity`

contribute strongly to model predictions.

However, several size-related morphological features are correlated.

Therefore, these feature-importance values are interpreted as **model-based associations**.

They are **not interpreted as evidence that an individual feature independently causes a grain to belong to a particular rice variety**.

This distinction is important for responsible interpretation of the model.

---

## 8. Cross-Validation Decision

### Status: ⏳ Planned — Not Yet Completed

The current model comparison primarily uses a single stratified holdout test partition.

To improve the reliability of model comparison, the project plans to add:

**Stratified 5-Fold Cross-Validation**

The intended workflow is:

```text
Original Dataset
       ↓
80/20 Stratified Train/Test Split
       ↓
Training Set
       ↓
Stratified 5-Fold Cross-Validation
       ↓
Compare Candidate Models
       ↓
Select / Finalise Model
       ↓
Train on Full Training Set
       ↓
Final Evaluation on Untouched Test Set
       ↓
Final Recommendation
```

Cross-validation will be performed using the **training data**, while the final test set will remain separate from model-selection decisions.

The cross-validation comparison should consider:

- Mean performance across folds
- Standard deviation across folds
- F1-score / Macro-F1
- Other relevant evaluation measures
- Model stability

### Important

Cross-validation has **not yet been completed**.

No cross-validation results should be reported until the implementation has been completed and verified.

---

## 9. Final Model Decision

### Status: ⏳ Pending

The final model has **not yet been selected**.

Based on the current single holdout split, Logistic Regression is a strong provisional candidate because it provides competitive predictive performance with a relatively small train/test generalisation gap.

However, the difference between Logistic Regression and some alternative models is small.

Therefore, the final model decision will be made after cross-validation.

The final decision will consider:

1. Cross-validation performance
2. Performance variability across folds
3. Final holdout test performance
4. Generalisation
5. Model complexity
6. Interpretability
7. Practical stakeholder value

This section will be updated once cross-validation has been completed.

---

## 10. Responsible Interpretation Decisions

The project treats machine learning as a form of **decision support** rather than assuming that model predictions are always correct.

The following interpretation decisions are applied:

- Model predictions may contain errors.
- Performance on the current dataset does not guarantee equivalent performance in real agricultural environments.
- Feature importance is interpreted as model-based association rather than causation.
- Models should not automatically be generalized to rice varieties that were not represented in the training data.
- Real-world deployment would require additional validation.
- Model complexity alone is not considered evidence of better performance.
- Final recommendations should be supported by validation evidence rather than a single performance metric.

---

## 11. Current Decision Status Summary

| Area | Current Status |
|---|---|
| Business lens | ✅ Completed |
| Unit of analysis | ✅ Completed |
| ML task formulation | ✅ Completed |
| Train/test strategy | ✅ Completed |
| Missing-value decision | ✅ Completed |
| Duplicate decision | ✅ Completed |
| Outlier strategy | ✅ Completed |
| Feature scaling | ✅ Completed |
| Target encoding | ✅ Completed |
| Baseline model | ✅ Completed |
| Alternative models | ✅ Completed |
| Evaluation metrics | ✅ Completed |
| Generalisation analysis | ✅ Completed |
| Feature-importance approach | ✅ Completed |
| Stratified 5-Fold CV | ⏳ Planned |
| Macro-F1 extension | ⏳ Planned / To be strengthened |
| Final model selection | ⏳ Pending |
| Final recommendation | ⏳ Pending |

---

## 12. Pending Decision Log Updates

The following updates will be made as the project progresses:

- [ ] Implement Stratified 5-Fold Cross-Validation.
- [ ] Record cross-validation mean performance for each model.
- [ ] Record cross-validation standard deviation for each model.
- [ ] Add Macro-F1 results if implemented.
- [ ] Compare model stability across folds.
- [ ] Finalise model selection using validation evidence.
- [ ] Evaluate the selected/finalised model on the untouched test set.
- [ ] Update the final recommendation.
- [ ] Record final limitations and stakeholder implications.

---

## 13. Decision Log Update Policy

This decision log represents the decisions supported by the project's current implementation and evidence.

When new evidence becomes available, particularly from cross-validation, the relevant decision should be **updated rather than rewriting the project's previous reasoning as though the later evidence had always been available**.

This allows the decision log to demonstrate how the group's machine learning reasoning developed throughout the project.

---

## Current Project Decision

> **The project does not yet declare a final winning model.**
>
> Logistic Regression is currently a strong provisional candidate based on the existing holdout evaluation. However, final model selection and recommendation will be completed only after the planned cross-validation analysis provides additional evidence.

---

*This decision log will be updated as the IT3091 Machine Learning group assignment progresses.*
