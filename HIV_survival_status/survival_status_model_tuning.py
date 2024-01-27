import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# 本代码使用xgboost的sklearn接口进行训练和调参
feature_columns = ["gender", "onset_age", "onset_age_class", "education_level", "occupation", "marital_status", "venereal_history", "infection_pathway", "last_CD4_result", "treat_status"]
label_columns = ["survival_status"]

# 特征值
data = pd.read_excel("../dataset/Datas.xlsx", sheet_name="Sheet1", usecols=feature_columns)
# 标签值，生存状态
target = pd.read_excel("../dataset/Datas.xlsx", sheet_name="Sheet1", usecols=label_columns)

# 训练集和测试集划分，test_size=0.3表示训练集和测试集划分比例为7:3
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=2024)

# 初始模型，除下列learning task参数外其他参数暂为默认值
xgb_clf = XGBClassifier(
    objective='binary:logistic', # 目标函数，二分类
    nthread=4,
    seed=2024,
    booster='gbtree'
)

# 网格搜索调参
# 1. 调节n_estimators参数, n_estimator是最大生成树数目，也是最大迭代次数
param_grid1 = {'n_estimators': range(0, 10, 1)}
optimized_xgb_clf1 = GridSearchCV(estimator=xgb_clf, param_grid=param_grid1, scoring='roc_auc', cv=5)
optimized_xgb_clf1.fit(X_train, y_train)
print(f'最佳迭代结果: {optimized_xgb_clf1.best_score_}')
print(f'最佳n_estimators参数: {optimized_xgb_clf1.best_params_}')

# 2. 调节 max_depth和min_child_weight参数，max_depth 是最大树深度，用于控制过拟合；min_child_weight是最小叶子点样本权重和，同样用于避免过拟合
param_grid2 = {'max_depth': range(1, 10, 1),
               'min_child_weight': range(1, 10, 1)}
# 得到上一次调参的最佳参数后, 在下一次调参里在estimator中将上一个最佳参数设置
optimized_xgb_clf2 = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', nthread=4, seed=2024, n_estimators=4),
                                  param_grid=param_grid2, scoring='roc_auc', cv=5)
optimized_xgb_clf2.fit(X_train, y_train)
print(f'最佳迭代结果: {optimized_xgb_clf2.best_score_}')
print(f'最佳max_depth和min_child_weight参数: {optimized_xgb_clf2.best_params_}')

# 3. 调节gamma参数， gamma是树节点分裂所需最小损失函数下降值，gamma值越大算法越保守
param_grid3 = {'gamma': [i/10.0 for i in range(1, 10, 1)]}
optimized_xgb_clf3 = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', nthread=4, seed=2024, n_estimators=4,
                                                          max_depth=4, min_child_weight=3),
                                  param_grid=param_grid3, scoring='roc_auc', cv=5)
optimized_xgb_clf3.fit(X_train, y_train)
print(f'最佳迭代结果: {optimized_xgb_clf3.best_score_}')
print(f'最佳gamma参数: {optimized_xgb_clf3.best_params_}')

# 4. 调节subsample和colsample_bytree参数， subsample用于控制每棵树随机采样的比例，colsample_bytree控制每棵树随机采样的列数的占比，参数值越小算法越保守
param_grid4 = {'subsample': [i/10.0 for i in range(5, 11, 1)],
               'colsample_bytree': [i/10.0 for i in range(5, 11, 1)]}
optimized_xgb_clf4 = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', nthread=4, seed=2024, n_estimators=4,
                                                          max_depth=4, min_child_weight=3, gamma=0.2),
                                  param_grid=param_grid4, scoring='roc_auc', cv=5)
optimized_xgb_clf4.fit(X_train, y_train)
print(f'最佳迭代结果: {optimized_xgb_clf4.best_score_}')
print(f'最佳subsample和colsample_bytree参数: {optimized_xgb_clf4.best_params_}')

# 5. 调节reg_alpha和reg_lambda参数，reg_alpha是权重的L1正则化项，reg_lambda参数是权重的L2正则化项，用于减少过拟合
param_grid5 = {'reg_alpha': [i/10.0 for i in range(1, 11, 1)],
               'reg_lambda': [i/10.0 for i in range(1, 11, 1)]}
optimized_xgb_clf5 = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', nthread=4, seed=2024, n_estimators=4,
                                                          max_depth=4, min_child_weight=3, gamma=0.2, subsample=1, colsample_bytree=0.6),
                                  param_grid=param_grid5, scoring='roc_auc', cv=5)
optimized_xgb_clf5.fit(X_train, y_train)
print(f'最佳迭代结果: {optimized_xgb_clf5.best_score_}')
print(f'最佳reg_alpha和reg_lambda参数: {optimized_xgb_clf5.best_params_}')

# 6. 调节learning_rate参数，调小learning_rate，默认的learning_rate是0.3
param_grid6 = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]}
optimized_xgb_clf6 = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', nthread=4, seed=2024, n_estimators=4,
                                                          max_depth=4, min_child_weight=3, gamma=0.2, subsample=1, colsample_bytree=0.6,
                                                          reg_alpha=0.7, reg_lambda=0.2),
                                  param_grid=param_grid6, scoring='roc_auc', cv=5)
optimized_xgb_clf6.fit(X_train, y_train)
print(f'最佳迭代结果: {optimized_xgb_clf6.best_score_}')
print(f'最佳learning_rate参数: {optimized_xgb_clf6.best_params_}')

