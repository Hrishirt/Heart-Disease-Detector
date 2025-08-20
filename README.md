# 🫀 Heart Disease Detector  

This project explores different approaches to predicting the presence of **heart disease** using the UCI Heart Disease dataset. The goal was to not just run models, but to **implement them from scratch**, compare them to library methods, and analyze trade-offs in evaluation.  

---

## 🔧 Approaches Implemented  

1. **Manual Logistic Regression (NumPy)**  
   - Implemented logistic regression from scratch with gradient descent.  
   - Debugged weights, shapes, and loss function behavior.  
   - Validated performance against sklearn’s implementation.  

2. **Logistic Regression (scikit-learn)**  
   - Used sklearn’s `LogisticRegression` for efficiency and reproducibility.  
   - Achieved similar results to the manual version, confirming correctness.  

3. **🌲 Random Forest (scikit-learn)**  
   - Leveraged an ensemble of decision trees (`RandomForestClassifier`).  
   - Outperformed logistic regression on this dataset (higher ROC AUC).  
   - Showed how tree-based ensembles capture non-linear feature interactions.  

---

## 📊 Metrics  

### Logistic Regression (Manual & sklearn)  
- **Accuracy:** ~78%  
- **ROC AUC:** ~0.88  
- **Precision (class 1 / disease):** 0.74  
- **Recall (class 1 / disease):** 0.87  

### 🌲 Random Forest  
- **Accuracy:** ~99% (cross-validation)  
- **ROC AUC:** ~0.999 (cross-validation)  
- ⚠️ *Note: dataset is small/clean → ensemble methods can nearly memorize it. In real-world data, results won’t be this high.*  

---

## ⚖️ Precision vs Recall Trade-off  

By default, classifiers use a threshold of **0.5** to decide between “disease” and “no disease.”  

- Lowering the threshold (e.g. `0.3`) → **higher recall** (catch more true disease cases) but lower precision (more false alarms).  
- Raising the threshold (e.g. `0.9`) → **higher precision** (fewer false alarms) but lower recall (miss more true cases).  

NOTE: In medical applications, **recall is often prioritized** it’s usually safer to flag more patients for follow-up than to miss someone with a condition.  

---

## 🚀 How to Use  

```python
# Train logistic regression
log_reg.fit(X_train, y_train)

# Predict probability for a new patient
proba = log_reg.predict_proba(new_patient)[:,1]
pred  = int(proba >= 0.3)   # threshold can be tuned

