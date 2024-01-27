import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 本代码使用xgboost的sklearn接口进行训练和调参
feature_columns = ["gender", "onset_age", "onset_age_class", "education_level", "occupation", "marital_status",
                   "venereal_history", "infection_pathway", "last_CD4_result", "treat_status"]
label_columns = ["survival_status"]

# 特征值
data = pd.read_excel("../dataset/Datas.xlsx", sheet_name="Sheet1", usecols=feature_columns)
# 标签值，生存状态
target = pd.read_excel("../dataset/Datas.xlsx", sheet_name="Sheet1", usecols=label_columns)

# 训练集和测试集划分，test_size=0.3表示训练集和测试集划分比例为7:3
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=2024)

# XGBoost的Sklearn接口训练
best_model1 = XGBClassifier(
    objective='binary:logistic', # 目标函数，二分类
    booster='gbtree',
    nthread=4,
    seed=2024,
    n_estimators=4,
    max_depth=4,
    min_child_weight=3,
    colsample_bytree=0.6,
    subsample=1.0,
    reg_alpha=0.7,
    reg_lambda=0.2,
    learning_rate=0.3
)
best_model1.fit(X_train, y_train)

# ROC

# 训练集ROC
y_train_predict = best_model1.predict(X_train)  # 输出分类
y_train_predict_prob = best_model1.predict_proba(X_train)  # 输出为概率
y_train_predict_prob_positive = y_train_predict_prob[:, 1]  # 取为正例1的概率
fpr_train, tpr_train, thresholds_rain = roc_curve(y_train, y_train_predict_prob_positive, pos_label=1)
roc_auc_train = auc(fpr_train, tpr_train)

# 测试集ROC
y_test_predict = best_model1.predict(X_test)  # 输出分类
y_test_predict_prob = best_model1.predict_proba(X_test)  # 输出为概率
y_test_predict_prob_positive = y_test_predict_prob[:, 1]  # 取为正例1的概率
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_predict_prob_positive, pos_label=1)
roc_auc_test = auc(fpr_test, tpr_test)

# 绘制ROC曲线
plt.figure(num=1, figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title('Train Roc Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc_train:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.title('Test Roc Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc='lower right')
plt.show()

# PR
# 训练集
precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, y_train_predict_prob_positive)
# 测试集
precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, y_test_predict_prob_positive)

# 绘制PR曲线
plt.figure(num=2, figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title('Train Precission-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precission')
plt.plot(recall_train, precision_train, color='darkorange', lw=2, label='XGBoost')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.title('Test Precission-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precission')
plt.plot(recall_test, precision_test, color='darkorange', lw=2, label='XGBoost')
plt.legend(loc='upper right')
plt.show()

