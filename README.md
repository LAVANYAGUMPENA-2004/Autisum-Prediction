##  Autism Prediction using Machine Learning

This project implements a supervised machine learning pipeline to predict whether an individual may be on the autism spectrum based on a series of behavioral and demographic features. The model uses a structured dataset and trains multiple classifiers to optimize prediction accuracy.

---

### 🎯 Objective

To develop a binary classification model that predicts the likelihood of autism in individuals using survey responses and demographic information. The prediction is based on features such as age, gender, ethnic group, family history, and responses to autism spectrum quotient (AQ-10) questions.

---

### 📁 Dataset

* **Source**: [Autism Screening Adult Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult)
* **Format**: CSV file
* **Target Column**: `Class/ASD`
* **Features**: Includes AQ-10 responses, age, gender, family history, and other relevant attributes.

---

### 🧪 Features Used

* Age
* Gender
* Ethnicity
* Jaundice at birth
* Family history of autism
* Screening score responses (A1 to A10)
* Relation to respondent
* Used app before?
* Result from AQ-10 Test

---

### Model Development

* **Preprocessing**:

  * Handled missing values
  * Encoded categorical variables
  * Normalized/standardized features
* **Algorithms Used**:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
* **Evaluation Metrics**:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1-Score

---

### 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | \~95%    |
| Decision Tree       | \~97%    |
| Random Forest       | \~98%    |
| SVM                 | \~96%    |

> Random Forest achieved the highest accuracy in predicting autism from the input features.

---

### 🛠️ Technologies Used

| Tool/Library        | Purpose                 |
| ------------------- | ----------------------- |
| Python              | Programming language    |
| Pandas, NumPy       | Data manipulation       |
| Scikit-learn        | Machine learning models |
| Matplotlib, Seaborn | Data visualization      |
| Google Colab        | Execution environment   |

---

### 🚀 How to Run

1. Clone this repository or open the notebook in Google Colab.
2. Upload the dataset (`autism_screening.csv`).
3. Run the notebook step-by-step to preprocess, train, and evaluate models.
4. View metrics and confusion matrices to select the best model.

---

### 📄 Folder Structure

```
autism-prediction/
├── Autism_Prediction.ipynb
├── autism_screening.csv
├── README.md
└── requirements.txt
```

---

### 🔮 Future Enhancements

* Use neural networks or deep learning models for comparison.
* Deploy the best-performing model as a web app using Streamlit.
* Incorporate user input forms for live predictions.

---


