#### Mental Health Prediction Dataset

사용한 자료 : https://www.kaggle.com/datasets/harpartapsingh13/mental-health-prediction-dataset/data

불안, 우울증 & 번아웃 등을 잠, 스트레스, 생활 방식 데이터를 통해 분류하는 자료

분류 대상 : "mental_health_condition"

사용한 feature : "sleep_hours", "sleep_quality", "social_media_hours",
"work_life_balance", "academic_work_pressure",
"physical_activity_days", "stress_level", "anxiety_score"

## Result (Logstic Regression)

우선적으로 class의 분포에 불균형이 있을 시 정확도가 왜곡될 가능성이 있어, 클래스 간 개수가 같은지 확인한다.

    y.value_counts()

> mental_health_condition
> Burnout 90
> Normal 88
> Anxiety 87
> Depression 86
> Name: count, dtype: int64
> 정확도(Accuracy): 0.9014

결과적으로 4개의 class 모두 비슷한 값을 가지고 있는 것을 확인할 수 있었다.
또한 분류 정확도가 90%로 높은 것을 확인할 수 있다.

#### Confusion Matrix

condition의 class가 4개이므로 양성 / 음성 예측이 아닌 어떤 것을 예측했는지를 나타내는 표가 생성되었다.

                   예측:Anxiety  예측:Burnout  예측:Depression  예측:Normal

실제:Anxiety 15 2 1 0
실제:Burnout 1 17 0 0
실제:Depression 2 1 14 0
실제:Normal 0 0 0 18

### classification_report

import classification_report을 통해 정밀도, 재현율, f1-score값을 불러왔다.

macro avg : 단순평균
weighted avg : 가중평균

Classification Report:
precision recall f1-score support(test set에 들어있던 각 class의 개수)

     Anxiety       0.83      0.83      0.83        18
     Burnout       0.85      0.94      0.89        18

Depression 0.93 0.82 0.88 17
Normal 1.00 1.00 1.00 18

    accuracy                           0.90        71

macro avg 0.90 0.90 0.90 71
weighted avg 0.90 0.90 0.90 71

## Result (Naive Bayes)

1.  Naive Bayes 정확도: 0.9014

2.  분류 리포트 (Naive Bayes):
    precision recall f1-score support

         Anxiety       0.83      0.83      0.83        18
         Burnout       0.84      0.89      0.86        18

    Depression 0.94 0.88 0.91 17
    Normal 1.00 1.00 1.00 18

        accuracy                           0.90        71

    macro avg 0.90 0.90 0.90 71
    weighted avg 0.90 0.90 0.90 71

3.  Confusion Matrix
    예측:Anxiety 예측:Burnout 예측:Depression 예측:Normal
    실제:Anxiety 15 2 1 0
    실제:Burnout 2 16 0 0
    실제:Depression 1 1 15 0
    실제:Normal 0 0 0 18

## 해석

Logstic Regression과 Naive Bayes 결과가 거의 동등하게 나와

Naive Bayes의 기본 가정인 모든 x가 독립이라는 가정 하에도
분류가 잘 되었음을 확인할 수 있었으며,

또한 Logstic Regression으로도 분류가 잘 되었음을 확인할 수 있었다.

또한 ROC curve를 data x에 대해 각 label y 하나씩 나눠 그려보게 되면,
precision, recall, f1-score에서 모두 수치가 낮은 y인 anxiety가 class들 중 가장 넓이가 작게 그려진 것을 확인할 수 있었다.
(AUC 값을 통해 확인)

이를 위해 label_binarize를 통해 y를 이진화하였고,

y_test_bin[:, i] → 실제 정답
y_score[:, i] → 모델이 예측한 확률값 (X_test 사용)

roc_curve 함수를 통해 두 값을 비교해 ROC Curve를 그려냈다.
