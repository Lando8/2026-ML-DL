import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

target_col = "mental_health_condition"

feature_cols = ["sleep_hours", "sleep_quality", "social_media_hours", 
                 "work_life_balance", "academic_work_pressure", 
                 "physical_activity_days", "stress_level", "anxiety_score"]

# 2. 결측치 제거
mentalDF = mentalDF.dropna(subset=feature_cols + [target_col])

# 3. X, y 정의
X = mentalDF[feature_cols]
y = mentalDF[target_col]

# 타겟 클래스 확인
print(y.value_counts())

# stratify : train/test에서도 positive / negative 비율을 동등하게 나눈다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. 예측
y_pred = model.predict(X_test_scaled)

# 8. 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도(Accuracy): {accuracy:.4f}")

print("\n분류 리포트:")
print(classification_report(y_test, y_pred))

print("\n혼동행렬(Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))



from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


nb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])

nb_pipeline.fit(X_train, y_train)
y_pred_nb = nb_pipeline.predict(X_test)

print(f"Naive Bayes 정확도: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\n분류 리포트 (Naive Bayes):")
print(classification_report(y_test, y_pred_nb))
print("\n혼동행렬 (Naive Bayes):")
print(confusion_matrix(y_test, y_pred_nb))



# ROC curve > 각 분류해야하는 class에 대해 곡선을 하나씩 그렸다.

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# 1. y를 One-vs-Rest 형태로 이진화 (4개 클래스 → 4개의 0/1 컬럼)
classes = model.classes_  # ['Anxiety', 'Burnout', 'Depression', 'Normal'] 순서
y_test_bin = label_binarize(y_test, classes=classes)  # (n_samples, 4) 형태

# 2. 각 클래스에 대한 예측 확률 얻기
y_score = model.predict_proba(X_test_scaled)  # (n_samples, 4) 형태

# 3. 클래스별로 ROC curve 계산
fpr = dict() # false positive rate
tpr = dict() # true poistiive rate
roc_auc = dict()

for i, cls in enumerate(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 4. 그리기
plt.figure(figsize=(8, 6))
colors = ['blue', 'orange', 'green', 'red']

for i, cls in enumerate(classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'{cls} (AUC = {roc_auc[i]:.3f})')

# 대각선 (완전 랜덤 예측 기준선)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

