# Classification_NN_RF_Income
## Project Overview
This project compares the performance of three classification models applied to the Census Income dataset from the UCI Machine Learning Repository. The goal is to predict whether a person earns more or less than $50K per year, using personal and demographic attributes.

The models analyzed include:

- Decision Tree: Rule-based classification.

- Random Forest: Ensemble learning through multiple trees.

- Neural Network (Multilayer Perceptron): Deep learning approach using fully connected layers.

## Dataset Description
Records: 32,560 individuals

Target variable: Income (1: ≥$50K, 0: <$50K)

Features (14):

Nominal: Marital Status, Occupation, Relationship, Work Class, Race, Sex, Country

Ordinal: Education, Education Number

Ratio: Age, Hours per Week, Capital Gain, Capital Loss, Fnlwgt

## Data Preprocessing
- Removed 24 duplicate records

- Removed outliers beyond 3 standard deviations (≈6%)

- Imputed missing values (mode for most, proportional fill for distributed categories)

- Discretized and aggregated ordinal/categorical variables

- Binarized nominal values (e.g., Sex, Country)

- Handled multicollinearity (dropped one highly correlated feature)

- One-hot encoded categorical features

- Split into 70% training and 30% testing sets

## Models
### Neural Network (Multilayer Perceptron)
Architecture: 2 hidden layers (128 and 64 neurons), ReLU activation, dropout (20%)

Output: 1 neuron with sigmoid (binary classification)

Optimizer: Adam

Loss Function: Binary Crossentropy

Epochs: 50

Results:

Accuracy: 85%

Precision: 0.88 (class 1), 0.52 (class 0)

Recall: 0.94 (class 1), 0.52 (class 0)

ROC: Good separation but slightly lower than tree-based models

### Decision Tree
Hyperparameters tuned for best split (max leaves: 23)

Based on entropy minimization

Results:

Accuracy: 86%

Better precision on class 0 (compared to NN)

ROC: Slightly better than NN

### Random Forest
200 trees, random variable selection optimized (best: 26 variables)

Averaged predictions to improve generalization

Results:

Accuracy: 86%

Similar ROC and classification quality as Decision Tree

More robust and less prone to overfitting

## Evaluation Metrics
Accuracy: General classification correctness

Precision & Recall: Especially analyzed due to class imbalance (78% of records are class 1)

ROC Curves: All models perform above random baseline; tree models slightly outperform NN

## Conclusion
All models show comparable accuracy, but tree-based models slightly outperform in ROC and class 0 detection.

Neural Network offers competitive performance but is more computationally expensive.

Random Forest proves the most balanced in performance and robustness.

Future Work: Apply class balancing techniques (e.g., oversampling, SMOTE) to improve minority class (class 0) recall.
