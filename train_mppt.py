#!/usr/bin/env python
# coding: utf-8

# # MPPT Optimal Duty Cycle Prediction Using ANN
# 
# **Project:** Deep Learning-based MPPT for a solar panel
# 
# This notebook follows the structure of the sample ANN regression notebook, adapted to `mppt_4sensor_dataset.csv`. The model uses four input features and predicts `Optimal_Duty`.

# ## 1. Load the dataset
# Keep `mppt_4sensor_dataset.csv` in the same folder as this notebook.

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_file = "mppt_4sensor_dataset.csv" if os.path.exists("mppt_4sensor_dataset.csv") else "mppt_4sensor_dataset (1).csv"
df = pd.read_csv(dataset_file)
print(f"Loaded dataset from: {dataset_file}")
print("Dataset shape:", df.shape)
df.head()


# In[2]:


# Check column names, data types, missing values, and duplicate rows
print(df.columns.tolist())
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df.describe()


# ## 2. Exploratory Data Analysis (EDA)

# In[3]:


# Visualize the target variable
plt.figure(figsize=(7, 4))
plt.hist(df["Optimal_Duty"], bins=30)
plt.xlabel("Optimal Duty Cycle")
plt.ylabel("Frequency")
plt.title("Distribution of Optimal_Duty")
plt.tight_layout()
plt.show()


# In[4]:


# Correlation between the dataset columns
plt.figure(figsize=(7, 5))
plt.imshow(df.corr(numeric_only=True), aspect="auto")
plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar(label="Correlation")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# ## 3. Separate features (X) and target (y)
# 
# The four inputs are `Irradiance`, `Temperature`, `V_pv`, and `I_pv`. The target is `Optimal_Duty`. This dataset therefore trains the ANN to predict duty cycle, **not Vmpp voltage**.

# In[5]:


feature_columns = ["Irradiance", "Temperature", "V_pv", "I_pv"]
target_column = "Optimal_Duty"

X = df[feature_columns]
y = df[target_column]

print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Input features:", feature_columns)
print("Target:", target_column)


# ## 4. Split data into training, validation, and testing sets (80/10/10)

# In[6]:


from sklearn.model_selection import train_test_split

# First separate 10% for final testing
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# From the remaining 90%, take 1/9 (10% of the full dataset) for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=1/9, random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# ## 5. Normalize the input data
# 
# Fit the scaler only on training data to avoid data leakage, then apply it to validation and test data.

# In[7]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print("Normalized training shape:", X_train_norm.shape)


# ## 6. Build the 4-16-8-1 ANN model
# 
# - 4 input features
# - Hidden layer 1: 16 neurons
# - Hidden layer 2: 8 neurons
# - 1 output neuron for predicted duty cycle
# 
# This is a regression problem, so the output layer uses a linear activation.

# In[8]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)

model = Sequential([
    Input(shape=(4,), name="Input_Layer"),
    Dense(16, activation="relu", name="Hidden1"),
    Dense(8, activation="relu", name="Hidden2"),
    Dense(1, activation="linear", name="Output_Optimal_Duty")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ## 7. Train the ANN

# In[9]:


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# ## 8. Plot training and validation loss

# In[10]:


plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="Training Loss (MSE)")
plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()


# ## 9. Evaluate on unseen test data
# 
# MAE tells the average absolute prediction error; MSE penalizes larger errors more strongly.

# In[11]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test_norm).ravel()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE:  {mae:.6f}")
print(f"Test MSE:  {mse:.6f}")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test R²:   {r2:.6f}")


# ## 10. Compare actual and predicted duty cycle

# In[12]:


comparison = pd.DataFrame({
    "Actual_Optimal_Duty": y_test.to_numpy(),
    "Predicted_Optimal_Duty": y_pred
})
comparison["Absolute_Error"] = abs(
    comparison["Actual_Optimal_Duty"] - comparison["Predicted_Optimal_Duty"]
)
comparison.head(15)


# In[13]:


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
low = min(float(y_test.min()), float(y_pred.min()))
high = max(float(y_test.max()), float(y_pred.max()))
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual Optimal Duty")
plt.ylabel("Predicted Optimal Duty")
plt.title("Actual vs Predicted Duty Cycle")
plt.tight_layout()
plt.show()


# ## 11. Save the trained model and input scaler

# In[14]:


import joblib

model.save("mppt_ann_4_16_8_1.keras")
joblib.dump(scaler, "mppt_input_scaler.pkl")
print("Saved model: mppt_ann_4_16_8_1.keras")
print("Saved scaler: mppt_input_scaler.pkl")


# ## 12. Predict duty cycle for one new sample
# 
# Replace the example values with real sensor readings in the same column order.

# In[15]:


new_sample = pd.DataFrame([{
    "Irradiance": 800,
    "Temperature": 35,
    "V_pv": 17.0,
    "I_pv": 2.0
}])

new_sample_norm = scaler.transform(new_sample[feature_columns])
predicted_duty = model.predict(new_sample_norm, verbose=0)[0, 0]
print(f"Predicted Optimal Duty Cycle: {predicted_duty:.4f}")


# In[16]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict on normalized test features
y_pred = model.predict(X_test_norm).flatten()
y_actual = np.asarray(y_test).flatten()

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("Test MAE:", mae)
print("Test RMSE:", rmse)
print("Test R2 Score:", r2)


# ## Important notes
# 
# 1. The CSV currently contains **2,736 rows and 5 columns** (four inputs plus one target).
# 2. The provided project proposal specifies an 80/10/10 train-validation-test split, MSE loss, Adam optimizer, and MAE evaluation; this notebook follows that plan.
# 3. The target in this CSV is `Optimal_Duty`. Do not describe this model as predicting `Vmpp` unless you use a dataset whose target column is actually Vmpp.
# 4. A low test error on this dataset does not by itself prove faster real-world MPPT tracking. That requires comparison against a baseline such as P&O and, ideally, hardware/simulation validation.
# 5. Run the notebook from top to bottom. If TensorFlow is missing, install it in your notebook environment with `%pip install tensorflow scikit-learn joblib matplotlib pandas`.
