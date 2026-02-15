# Telco Customer Churn Prediction

This project focuses on predicting customer churn for a Telco company using Python, machine learning, and deep learning. The dataset is sourced from Kaggle: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn).

## Contents
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Customer data including churn information  
- Python notebook: Data preprocessing, model training, and evaluation  

## Technologies Used
- Python 
- Pandas, NumPy  
- Scikit-learn (Logistic Regression, Random Forest)  
- TensorFlow / Keras (Deep Learning)  
- Matplotlib, Seaborn (Visualization)  

## Data Preprocessing
1. Dropped the `customerID` column.  
2. Converted `TotalCharges` to numeric type and removed missing values.  
3. Applied one-hot encoding to categorical variables.  
4. Split the data into training and test sets using `train_test_split`.  
5. Scaled the features using `StandardScaler`.  

## Models
### 1. Logistic Regression
- Trained on scaled training data.  
- Accuracy: ~0.79  
- F1-score and recall evaluated, especially for churned customers.  

### 2. Random Forest Classifier
- 300 trees, `class_weight='balanced'` for imbalanced data.  
- Accuracy: ~0.79  
- Confusion matrix visualized using Seaborn.  

### 3. Deep Learning (Keras)
- 3 hidden layers with dropout and L2 regularization.  
- Activation functions: ReLU (hidden layers), Sigmoid (output layer)  
- Loss function: Binary Crossentropy  
- Optimizer: Adam  
- Epochs: 50, Batch size: 10  

## Results
| Model                  | Accuracy |
|------------------------|---------|
| Logistic Regression    | 0.7901  |
| Random Forest          | 0.7928  |
| Deep Learning (Keras)  | ~0.78   |

## Visualization
- Confusion matrix of the Random Forest model visualized using Seaborn.  

