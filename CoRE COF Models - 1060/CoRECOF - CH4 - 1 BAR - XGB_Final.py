import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np  # Plotlama için gerekli
from xgboost import XGBRegressor  # XGBoostRegressor'ı import ediyoruz
tpot_data = pd.read_csv('C:/ERC Project - COF Storage/ML Excels/OnlyCoRECOF - HighHenry/CoRECOF - CH4 - 1 BAR - XGB.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('CH4-1 bar (mol/kg)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['CH4-1 bar (mol/kg)'], train_size=0.80, test_size=0.20, random_state=42)

# Average CV score on the training set was: -0.005829691596959061
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=1.0, learning_rate=0.1, max_depth=4, n_estimators=100, subsample=1.0)),
    XGBRegressor(colsample_bytree=1.0, learning_rate=0.1, max_depth=8, n_estimators=100, subsample=0.7)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train = exported_pipeline.predict(training_features)
preds = exported_pipeline.predict(testing_features)

X_unseen = pd.read_csv('C:/Users/Lenovo/Desktop/KodSystems/TPOT/Unseen/Unseen - CH4 -  Henrys - 6872.csv', sep=',')
y_pred_unseen = exported_pipeline.predict(X_unseen)
df5 = pd.DataFrame(y_pred_unseen)
df5.to_csv('HypoCOF - CH4 - Unseen.csv', index=False)

# Metrikleri depolamak için bir liste oluştur
metrics = []

# Metrikleri hesapla
metrics.append({'Metric': 'R2', 'Train': r2_score(training_target, y_pred_train), 'Test': r2_score(testing_target, preds)})
metrics.append({'Metric': 'MSE', 'Train': mean_squared_error(training_target, y_pred_train), 'Test': mean_squared_error(testing_target, preds)})
metrics.append({'Metric': 'MAE', 'Train': mean_absolute_error(training_target, y_pred_train), 'Test': mean_absolute_error(testing_target, preds)})
metrics.append({'Metric': 'RMSE', 'Train': math.sqrt(mean_squared_error(training_target, y_pred_train)), 'Test': math.sqrt(mean_squared_error(testing_target, preds))})
metrics.append({'Metric': 'SRCC', 'Train': spearmanr(training_target, y_pred_train)[0], 'Test': spearmanr(testing_target, preds)[0]})

# Metrikleri içeren bir DataFrame oluştur
metrics_df = pd.DataFrame(metrics)

# DataFrame'i yazdır
print(metrics_df)

# Plot için figür ve eksen oluştur
fig, ax = plt.subplots()

# Eğitim verileri için scatter plot
training_scatter = ax.scatter(training_target, y_pred_train, color="blue", label='Training Data')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')

# Test verileri için scatter plot
testing_scatter = ax.scatter(testing_target, preds, color="red", label='Testing Data')
ax.set_xlabel('Simulated')
ax.set_ylabel('ML-predicted')

# x=y doğrusunu çiz
x = np.linspace(min(min(training_target), min(testing_target)), max(max(training_target), max(testing_target)), 100)
ax.plot(x, x, color='black', linestyle='--', label='_nolegend_')

# Sadece "Training Data" ve "Testing Data" için legend ekle
handles = [training_scatter, testing_scatter]
labels = [handle.get_label() for handle in handles]
ax.legend(handles=handles, labels=labels)

# Plot'u göster
plt.show()

results = exported_pipeline.predict(testing_features)
