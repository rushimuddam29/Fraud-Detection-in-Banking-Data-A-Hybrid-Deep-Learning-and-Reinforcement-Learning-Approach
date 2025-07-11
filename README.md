# Fraud-Detection-in-Banking-Data-A-Hybrid-Deep-Learning-and-Reinforcement-Learning-Approach
bankfraud
# Fraud Detection in Banking Data 🏦  
*A Hybrid Deep Learning and Reinforcement Learning Approach*

## 📌 Overview
This project presents a scalable and adaptive fraud detection system that integrates Deep Neural Networks (DNNs) for initial fraud classification and Deep Q-Networks (DQNs) for dynamic threshold tuning. It addresses the challenges of class imbalance, evolving fraud patterns, and high false positives in banking transaction datasets.

## 🎯 Objectives
- Build a hybrid model combining supervised learning and reinforcement learning.
- Use DNNs to classify fraudulent vs. non-fraudulent transactions.
- Apply DQN to dynamically adjust fraud detection thresholds in real-time.
- Handle class imbalance using SMOTE and class weight tuning.
- Evaluate model performance using robust metrics like F1-score and MCC.

## 🛠️ Technologies & Tools
- **Programming**: Python  
- **Libraries & Frameworks**:  
  - Deep Learning: TensorFlow, PyTorch  
  - Machine Learning: Scikit-learn, imbalanced-learn (SMOTE), Bayesian Optimization  
  - Data Handling: Pandas, NumPy  
  - Visualization: Matplotlib, Seaborn  
  - Reinforcement Learning: OpenAI Gym  
  - Evaluation: ROC-AUC, F1-score, MCC, Confusion Matrix

## 🧠 Algorithms Used
- **Deep Neural Networks (DNNs)** – for fraud classification  
- **Deep Q-Networks (DQN)** – for adaptive threshold learning  
- **SMOTE** – to address class imbalance  
- **Bayesian Optimization** – for hyperparameter tuning  
- **XGBoost / LightGBM / CatBoost** – used for benchmarking traditional approaches

## 📊 Evaluation Metrics
- Accuracy, Precision, Recall  
- F1-Score: **0.85**  
- Matthews Correlation Coefficient (MCC): **0.86**  
- ROC-AUC, PR-AUC

## 📈 Results
The hybrid DNN + DQN model outperformed traditional ML classifiers by dynamically adapting to changing fraud patterns and handling class imbalance effectively. Precision-Recall and ROC curves validated the reliability of the system on the Kaggle credit card fraud dataset.

## 🔍 Future Enhancements
- Integrate Explainable AI (e.g., SHAP or LIME) for model transparency  
- Deploy using cloud platforms (e.g., AWS Lambda, GCP Functions)  
- Explore Graph Neural Networks (GNNs) for detecting fraud rings  
- Apply Multi-Agent Reinforcement Learning (MARL) for distributed detection systems

## 📂 Dataset
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions with 492 frauds (highly imbalanced)

## 👨‍💻 Contributors
- MUDDAM SANJEEVA RUSHIJANA  
- GUTURU THIRUMALESHU SHETTY
- BATHINI SUDHARSHAN 
- Supervisor: Dr. M. Jahir Pasha  
- Co-supervisor: Mr. S. Md. Shakeer

## 📝 License
This project is for academic and research use only. Please cite appropriately if used in your work.

