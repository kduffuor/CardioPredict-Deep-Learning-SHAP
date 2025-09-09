## 🫀 CardioPredict: Deep Learning-Based Heart Disease Prediction with SHAP Explainability
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)

A comprehensive machine learning pipeline for cardiovascular disease prediction using deep neural networks with explainable AI capabilities. This project demonstrates end-to-end ML development from exploratory data analysis to deployment-ready pipeline.

### Key Features
- **Deep Neural Network**: 4-layer architecture with dropout regularization
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Model Interpretability**: SHAP-based feature importance analysis
- **Perfect Performance**: 100% accuracy with robust cross-validation
- **Fast Training**: Optimized preprocessing and architecture design
- **Deployment Ready**: Saved models and preprocessing pipelines

### Architecture
**Neural Network Structure:**
```
Input Layer (13 features)
    ↓
Dense Layer (128 neurons, ReLU) → Dropout (30%)
    ↓
Dense Layer (64 neurons, ReLU) → Dropout (30%)
    ↓
Dense Layer (32 neurons, ReLU) → Dropout (30%)
    ↓
Output Layer (1 neuron, Sigmoid)
```

### Dataset
**Heart Disease Dataset** - 1025 samples, 13 features
- **Source**: [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Target**: Binary classification (0: No Disease, 1: Disease)
- **Features include**:
  - `age`, `sex`, `cp` (chest pain type)
  - `trestbps` (resting blood pressure)
  - `chol` (cholesterol)
  - `fbs` (fasting blood sugar)
  - `restecg` (resting ECG)
  - `thalach` (maximum heart rate)
  - `exang` (exercise-induced angina)
  - `oldpeak` (ST depression)
  - `slope`, `ca`, `thal`
- **Quality**: No missing values, balanced classes

### Tools and Technologies Used
- **Python** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning utilities and preprocessing
- **SHAP** - Model interpretability and explainability
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Development environment

### How to Use This Repository
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cardiopredict-deep-learning-shap.git
   cd cardiopredict-deep-learning-shap
   ```

2. **Install required packages:**
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn shap
   ```

3. **Run the analysis:**
- Open and execute the Jupyter notebook
- Follow the complete ML workflow exploring EDA, model training, and SHAP interpretability analysis

### Clinical Applications
- **Risk Assessment**: Early identification of high-risk patients
- **Decision Support**: Assist healthcare providers with data-driven insights
- **Screening Programs**: Population-level cardiovascular health monitoring
- **Treatment Planning**: Prioritize interventions based on risk factors
- **Research Tool**: Feature importance analysis for clinical research

### Disclaimer
**Important Notice**: This model is developed for educational and research purposes only. It should **NOT** be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.
