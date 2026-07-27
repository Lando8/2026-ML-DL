#### Mental Health Prediction Dataset

사용한 자료 : https://www.kaggle.com/datasets/harpartapsingh13/mental-health-prediction-dataset/data

불안, 우울증 & 번아웃 등을 잠, 스트레스, 생활 방식 등으로 예측하는 자료

사용한 자료 : "sleep_hours", "depression_score", "sleep_quality", "anxiety_score", "stress_level

X - sleep_hours", "sleep_quality", "anxiety_score", "stress_level
y - "depression_score"

X를 통해 y를 예측하는 모델을 sklearn 라이브러리를 통해 구현

결과
기울기(계수): [ 0.37015451 -0.30218797 0.19507073 0.39953194]
절편: 0.6156377666528856
MSE: 3.7604
R²: 0.3959
5-fold를 통한 평균 R²: 0.4680
Adjusted R²: 0.3641

절편과 기울기를 통해 y ≈ βTx 식을 예측하였고, 그에 따른 MSE와 R-squared 값을 구했다.
X의 변수의 개수가 많아질수록 R-squared의 값이 높아졌으며, linearRegression의 특징인 variable이 많아질수록 성능이 좋아진다는 것을 확인할 수 있었다.

또한 4강에서 나온 K-fold 방법을 사용해 R-squared를 구해보았으며,
test 케이스를 여러번 나눠 더욱 자세한 R-squared 값을 구할 수 있었다.

adjusted-R은 정말 변수가 추가되었을 때, 유의미한 변수인지 확인하는 계수로,
불필요한 x가 들어갔다면 score가 차감되는 식으로 구성되었다.
adjusted-R을 구할 때에는 k-fold가 적용되지 않았다.

하지만 sleep_hours의 값과 다른 값들이 다른 범위로 존재해 스케일링을 추후 진행할 필요가 있으며,
변수 간 상관관계가 있는지를 판단하지 않고 변수 X를 넣었기 때문에, 추후 변수 간의 상관관계를 확인할 필요가 있다.
또한 test set을 다수 관찰되어서 엄밀하게 모델 구성이 이루어지지 않았을 경우를 고려해 validation을 추가로 만들어서 엄밀하게 측정을 할 필요가 있다.

---

수정 / 추가되어야 했던 점

1. target variable / feature를 명확히 나누지 않음
    > target variable : condtion : Normal, Anxiety, Depression, Burnout / severity : Mild, Moderate, Severe
    > 이 대상들의 분류를 목표로 분석을 진행하면 분류 문제이기 때문에, 회귀 문제로 풀이하기 위해
    > quantitive한 수치들로만 이루어진 feature를 잠 시간, 퀄리티 / 우울증 수치로 나눠서 분석을 진행할 예정이다.
    > (추후 5,6강에서 배운 분류 문제로도 분석할 예정)

> feature : Lifestyle Factors, Psychological Indicators, Age, Gender, Occupation type

2. 데이터의 스케일의 범위를 동일하게 조정해 linear regression을 진행했어야 함

3. 여러 개의 feature 중 소수의 feature로만 학습을 진행

---

자세한 수정 내용

사용할 x feature :

- Lifestyle Factors(6 features)

예측할 y feature :
Psychological Indicators

1. stress level (1-10 scale)
2. Anxiety scores
3. Depression scores
