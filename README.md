# Interpretable Prediction of Post-transplant Lymphoproliferative Disorder (PTLD) in Thoracic Organ Recipients

Interpretable prediction of Post-transplant Lymphoproliferative Disorder (PTLD) based on FasterRisk algorithm. Other models (unregularized and regularized logistic regression, LightGBM, XGBoost, Random Forest) are developed for comparison. The 10-fold rolling cross-validation (CV) is used for model evaluation. Bayesian hyper-parameter optimization is performed using Tree-Structured Parzen Estimator (TPE).

- Original FasterRisk paper (by Liu et al.): https://proceedings.neurips.cc/paper_files/paper/2022/hash/7103444259031cc58051f8c9a4868533-Abstract-Conference.html 

- Paper on rolling cross-validation for model evaluation (by Miller et al.): https://www.jhltonline.org/article/S1053-2498(22)01882-4/fulltext

- srtr_thoracic_ptld_5y.py: models for predicting the 5-year risk of PTLD in thoracic organ recipients (heart, lung and heart-lung). 
- srtr_thoracic_ptld_5y_jupyter.ipynb: models for predicting the 5-year risk of PTLD in thoracic organ recipients (heart, lung and heart-lung). Jupyter notebook version.
