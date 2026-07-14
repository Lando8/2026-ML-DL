import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score  

kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Set the path to the file you'd like to load
file_path = "mental_health_prediction.csv"

# Load the latest version
mentalDF = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "harpartapsingh13/mental-health-prediction-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more informatSion:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

mentalDF = mentalDF.dropna(subset=["sleep_hours", "depression_score", "sleep_quality", "anxiety_score", "stress_level"])

X = mentalDF[["sleep_hours", "sleep_quality", "anxiety_score", "stress_level"]]
y = mentalDF["depression_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"기울기(계수): {model.coef_}")
print(f"절편: {model.intercept_}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 5-fold 방식으로 R-squared 구하기
lr_for_cv = LinearRegression()
scores = cross_val_score(lr_for_cv, X, y, cv=kf, scoring='r2')
print(f"5-fold 평균 R²: {scores.mean():.4f}")


n = len(y_test)
p = X_test.shape[1]  # 변수 개수
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R²: {adj_r2:.4f}")

# ### 최소제곱법 계산 코딩 - 시간 부족으로 정확한 이해 x


# # X_train, y_train을 numpy 배열로 변환
# X_arr = X_train.values          # (n, 3) 형태 그대로 유지, flatten 안 함!
# y_arr = y_train.values          # (n,) 형태

# # 절편을 위해 맨 앞에 1로 채워진 열 추가 → (n, 4) 형태
# X_with_bias = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

# # 정규방정식: beta = (X^T X)^-1 X^T y
# beta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y_arr

# intercept_manual = beta[0]
# coef_manual = beta[1:]   # 계수 3개

# print(f"직접 계산한 절편: {intercept_manual:.4f}")
# print(f"직접 계산한 계수: {coef_manual}")

# # 예측
# X_test_arr = X_test.values
# X_test_with_bias = np.column_stack([np.ones(X_test_arr.shape[0]), X_test_arr])
# y_pred_manual = X_test_with_bias @ beta

# # 평가
# mse_manual = np.mean((y_test.values - y_pred_manual) ** 2)
# ss_res = np.sum((y_test.values - y_pred_manual) ** 2)
# ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
# r2_manual = 1 - (ss_res / ss_tot)

# print(f"직접 계산한 MSE: {mse_manual:.4f}")
# print(f"직접 계산한 R²: {r2_manual:.4f}")
