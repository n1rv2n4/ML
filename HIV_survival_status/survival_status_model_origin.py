import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 本代码使用xgboost原生接口进行训练

feature_columns = ["gender", "onset_age", "onset_age_class", "education_level", "occupation", "marital_status",
                   "venereal_history", "infection_pathway", "last_CD4_result", "treat_status"]
label_columns = ["survival_status"]

data = pd.read_excel("../dataset/Datas.xlsx", sheet_name="Sheet1", usecols=feature_columns)
target = pd.read_excel("../dataset/Datas.xlsx", sheet_name="Sheet1", usecols=label_columns)

# 训练集和测试集划分，test_size=0.3表示训练集和测试集划分比例为7:3
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=2024)
dTrain = xgb.DMatrix(X_train, y_train)
dTest = xgb.DMatrix(X_test, y_test)

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
}
watchlist = [(dTrain, 'train')]

model = xgb.train(params, dTrain, num_boost_round=100, evals=watchlist)

xgb.plot_importance(model)
plt.show()

y_predict_prob = model.predict(dTest)
y_predict = (y_predict_prob >= 0.5) * 1

roc_auc = roc_auc_score(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
confusion_mat = confusion_matrix(y_test, y_predict)

print(f'auc: {roc_auc}')
print(f'accuracy score: {accuracy}')
print(f'precision score: {precision}')
print(f'recall score: {recall}')
print(f'f1 score: {f1}')
print(f'confusion matrix: {confusion_mat}')