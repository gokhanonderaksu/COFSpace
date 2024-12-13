import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np  # Needed for plotting
tpot_data = pd.read_csv('C:/ERC Project - COF Storage/ML Excels/HypoCoRE - HighHenry/HypoCoRE - O2 - 0.1 BAR.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('O2-0.1 bar (mol/kg)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['O2-0.1 bar (mol/kg)'], train_size=0.80, test_size=0.20, random_state=42)

# Average CV score on the training set was: -8.876197666266904e-07
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    FastICA(tol=0.15000000000000002),
    RidgeCV()
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train = exported_pipeline.predict(training_features)
preds = exported_pipeline.predict(testing_features)

X_unseen = pd.read_csv('C:/Users/Lenovo/Desktop/KodSystems/TPOT/Unseen/Unseen - O2 -  Henrys - 6309.csv', sep=',')
y_pred_unseen = exported_pipeline.predict(X_unseen)
df5 = pd.DataFrame(y_pred_unseen)
df5.to_csv('HypoCOF - O2 - Unseen.csv', index=False)

X_unseen2 = pd.read_csv('C:/Users/Lenovo/Desktop/KodSystems/TPOT/Unseen/Unseen - O2 -  Henrys - ALL.csv', sep=',')
y_pred_unseen2 = exported_pipeline.predict(X_unseen2)
df5 = pd.DataFrame(y_pred_unseen)
df5.to_csv('HypoCOF - O2 - Unseen.csv', index=False)

# Create a list to store the metrics
metrics = []

# Calculate metrics
metrics.append({'Metric': 'R2', 'Train': r2_score(training_target, y_pred_train), 'Test': r2_score(testing_target, preds)})
metrics.append({'Metric': 'MSE', 'Train': mean_squared_error(training_target, y_pred_train), 'Test': mean_squared_error(testing_target, preds)})
metrics.append({'Metric': 'MAE', 'Train': mean_absolute_error(training_target, y_pred_train), 'Test': mean_absolute_error(testing_target, preds)})
metrics.append({'Metric': 'RMSE', 'Train': math.sqrt(mean_squared_error(training_target, y_pred_train)), 'Test': math.sqrt(mean_squared_error(testing_target, preds))})
metrics.append({'Metric': 'SRCC', 'Train': spearmanr(training_target, y_pred_train)[0], 'Test': spearmanr(testing_target, preds)[0]})

# Create a DataFrame from the list of metrics
metrics_df = pd.DataFrame(metrics)

# Print the DataFrame
print(metrics_df)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Scatter plot for training data
training_scatter = ax.scatter(training_target, y_pred_train, color="blue", label='Training Data')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')

# Scatter plot for testing data
testing_scatter = ax.scatter(testing_target, preds, color="red", label='Testing Data')
ax.set_xlabel('Simulated')
ax.set_ylabel('ML-predicted')

# Plot the x=y line
x = np.linspace(min(min(training_target), min(testing_target)), max(max(training_target), max(testing_target)), 100)
ax.plot(x, x, color='black', linestyle='--', label='_nolegend_')

# Add legend with only "Training Data" and "Testing Data"
handles = [training_scatter, testing_scatter]
labels = [handle.get_label() for handle in handles]
ax.legend(handles=handles, labels=labels)

# Show the plot
plt.show()

results = exported_pipeline.predict(testing_features)
