import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


from scipy import stats

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


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


# 예측할 label : anxiety_score, stress_level, depression_score

# 결측치 제거
mentalDF = mentalDF.dropna(subset=["sleep_hours", "sleep_quality", "social_media_hours", "work_life_balance", "academic_work_pressure", "physical_activity_days", "depression_score", "stress_level", "anxiety_score"])

X = mentalDF[["sleep_hours", "sleep_quality", "social_media_hours", "work_life_balance", "academic_work_pressure", "physical_activity_days"]]
y = mentalDF["anxiety_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링 추가
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"기울기(계수): {model.coef_}")
print(f"절편: {model.intercept_}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")


# pipeline을 통해 5-fold에서 스케일링이 되도록 수정
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

scores = cross_val_score(lr_pipeline, X, y, cv=kf, scoring='r2')
print(f"5-fold 평균 R²: {scores.mean():.4f}")
