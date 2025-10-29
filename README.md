# 🩺 Thyroid Nodule Malignancy Prediction using Machine Learning

This project applies **machine learning models** — K-Means Clustering, Random Forest, and a Neural Network — to predict **malignant thyroid nodules** from clinical ultrasound and lab data.  
The analysis explores **unsupervised**, **supervised**, and **deep learning** methods to identify malignancy indicators and compare predictive performance.

---

## 🚀 Overview

Thyroid nodules are common clinical findings, and distinguishing **benign vs. malignant** nodules is crucial for early diagnosis and treatment.  
This Jupyter Notebook demonstrates an end-to-end pipeline: from data preprocessing and feature selection to model training, evaluation, and visualization.

### 🧩 Objectives
- Explore **feature relevance** (shape, calcification, echo pattern, size, margin, and composition)
- Perform **clustering (K-Means)** to identify natural groupings in the data  
- Train **supervised models (Random Forest & Neural Network)** for malignancy prediction  
- Compare model performance using **ROC-AUC**, **F1**, **Precision**, and **Recall**

---

## ⚙️ Technical Approach

### 1. **Data Preprocessing**
- Imported, normalized, and standardized clinical features  
- Removed non-essential identifiers (`id`)  
- Selected relevant imaging and diagnostic attributes  
- Addressed feature scaling for proper model convergence

### 2. **Feature Selection**
Focused on clinically significant features:
| Feature | Type | Description |
|----------|------|-------------|
| Shape | Binary | Irregular shapes often indicate malignancy |
| Calcification | Binary | Microcalcifications are strong cancer indicators |
| Echo Pattern | Binary | Uneven texture (hypoechoic) correlates with malignancy |
| Size | Continuous | Larger nodules tend to be more suspicious |
| Composition | Categorical | Solid nodules carry higher malignancy risk |
| Margin | Binary | Unclear or irregular margins imply invasive growth |

---

## 🧠 Machine Learning Models

### 🔹 **K-Means Clustering**
- Unsupervised binary clustering (`k=2`)
- Used **Elbow Method** and **Silhouette Scores** for evaluation
- Compared clusters to true labels using **Adjusted Rand Index (ARI)** and confusion matrix
- Generated 3D scatter plots to visualize feature relationships

### 🔹 **Random Forest Classifier**
- Supervised ensemble learning model with feature importance ranking
- Evaluated via:
  - ROC-AUC curve
  - Precision-Recall curve
  - Cross-validation and Grid Search for hyperparameter tuning
- Visualized key contributing features and individual decision trees

### 🔹 **Neural Network (MLPClassifier)**
- Two hidden layers, 11 neurons each  
- **Logistic activation** and **Adam optimizer**  
- **L2 regularization (α = 0.05)** to prevent overfitting  
- Trained with an 80/20 split and evaluated using:
  - Accuracy
  - ROC-AUC
  - Precision, Recall, and F1-Score
- Visualized **loss curve** and **ROC curve**

---

## 📊 Model Comparison

| Model | Accuracy | ROC AUC | F1 Score | Precision | Recall |
|--------|-----------|----------|-----------|------------|---------|
| Neural Network | High | Excellent | Balanced | Strong | Strong |
| Random Forest | High | Excellent | High | High | High |
| K-Means | Moderate | N/A | Moderate | Moderate | Moderate |

> 🧪 Both Random Forest and Neural Network outperformed K-Means, showing strong predictive power for malignancy classification.

---

## 📈 Visualization Highlights
- **Elbow Plot & Silhouette Analysis** – Cluster validation  
- **Confusion Matrices** – Model accuracy and misclassification insight  
- **Feature Importance Graphs** – Random Forest interpretability  
- **ROC & Precision-Recall Curves** – Classification performance  
- **3D Interactive Plots (Plotly)** – Visual exploration of clustering behavior  

---

## 🧰 Technologies Used
- **Python 3**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **scikit-learn**, **Plotly**, **SHAP**
- **Jupyter Notebook / Google Colab**

---

## 🔍 Key Insights
- K-Means clustering aligned moderately with malignancy labels (Adjusted Rand ≈ 0.4).  
- Random Forest achieved high accuracy and interpretability through feature importance ranking.  
- Neural Network provided robust performance, with high ROC-AUC and balanced classification metrics.  
- Clinically relevant features (shape, calcification, margin) proved the strongest malignancy indicators.

---

## 📚 Future Work
- Incorporate **deep learning CNNs** for ultrasound image data  
- Use **ensemble methods** combining Random Forest and NN outputs  
- Expand dataset for improved generalization  
- Add **SHAP interpretability dashboard** for model explanations

---

## 👨‍💻 Author
**Matthew Abounassar**  
Georgia Institute of Technology  
📧 [mabounassar3@gatech.edu]  

---

⭐ *If you found this project interesting, please consider starring the repository!*
