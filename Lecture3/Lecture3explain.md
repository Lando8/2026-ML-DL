#### Mental Health Prediction Dataset

사용한 자료 : https://www.kaggle.com/datasets/harpartapsingh13/mental-health-prediction-dataset/data

불안, 우울증 & 번아웃 등을 잠, 스트레스, 생활 방식 등으로 예측하는 자료

feature : sleep_hours", "sleep_quality", "social_media_hours", "work_life_balance", "academic_work_pressure", "physical_activity_days"
label : anxiety_score, stress_level, depression_score

X를 통해 y를 예측하는 모델을 sklearn 라이브러리를 통해 구현

#### depression_score 예측

결과 - 수정된 값
기울기(계수): [ 0.55676035 -0.17247597 0.17230171 -1.06100758 0.28152491 -0.35005176]
절편: 4.836734693877551
MSE: 2.8736
R²: 0.5448
5-fold 평균 R²: 0.4692

#### stress_level 예측

기울기(계수): [-0.43454122 -0.17684576 0.16304466 -1.09667932 0.44227662 -0.39706047]
절편: 5.924731182795699
MSE: 2.3483
R²: 0.6078
5-fold 평균 R²: 0.6160

#### anxiety_score 예측

기울기(계수): [-0.8235981 -0.27122935 0.36098109 -0.66320673 0.2611944 -0.07463336]
절편: 4.701492537313433
MSE: 2.1953
R²: 0.5659
5-fold 평균 R²: 0.4981

절편과 기울기를 통해 y ≈ βTx 식을 예측하였고, 그에 따른 MSE와 R-squared 값을 구했다.
X의 변수의 개수가 많아질수록 R-squared의 값이 높아졌으며, linearRegression의 특징인 variable이 많아질수록 성능이 좋아진다는 것을 확인할 수 있었다.

또한 K-fold 방법을 사용해 R-squared를 각각 구한 뒤 평균을 내 더욱 자세한 R-squared을 구할 수 있었다.

### 해석

label인 anxiety_score, stress_level, depression_score을 각각 하나씩 예측하였고,
모두 R-squared 값이 0.5 이상이 나오는 것을 확인할 수 있었다.
데이터 분석 분야가 정신 / 건강 인 만큼 잠을 잔 시간, 잠의 quality를 통해 우울중, 불안, 스트레스의 정도를 유의미하게 예측할 수 있다는 것을 확인하였다.

---

수정 / 추가되어야 했던 점 (문제점)

1. target variable / feature를 명확히 나누지 않음

    > target variable : condtion : Normal, Anxiety, Depression, Burnout / severity : Mild, Moderate, Severe
    > 이 대상들의 분류를 목표로 분석을 진행하면 분류 문제이기 때문에, 회귀 문제로 풀이하기 위해
    > quantitive한 수치들로만 이루어진 feature를 잠 시간, 퀄리티 / 우울증 수치로 나눠서 분석을 진행할 예정이다.
    > (추후 5,6강에서 배운 분류 문제로도 분석할 예정)
    >
    > feature : Lifestyle Factors, Psychological Indicators, Age, Gender, Occupation type

2. 데이터의 스케일의 범위를 동일하게 조정해 linear regression을 진행했어야 함

3. 여러 개의 feature 중 소수의 feature로만 학습을 진행
   (variable를 많이 포함해보지 않음)

---

자세한 수정 내용

1. feature, label 수정
   사용할 x feature :

- Lifestyle Factors(6 features)

예측할 y feature :
Psychological Indicators

1. stress level (1-10 scale)
2. Anxiety scores
3. Depression scores

4. 스케일링 진행

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
