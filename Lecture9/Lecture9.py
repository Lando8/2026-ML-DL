## 이해하지 못한 코드
## 추후 수정 예

import numpy as np
import pandas as pd

import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error, r2_score,
)

SEED = 42
DATASET = "harshadapatil31/student-performance-and-study-habits-dataset"

# dataload
def load():
    loader = getattr(kagglehub, "dataset_load", None) or kagglehub.load_dataset
    for file_path in ["student_performance_dataset.csv", ""]:
        try:
            return loader(KaggleDatasetAdapter.PANDAS, DATASET, file_path)
        except Exception as e:
            last = e
    # 그래도 안 되면 폴더를 통째로 받아서 안에 뭐가 있는지 보여준다
    import os
    d = kagglehub.dataset_download(DATASET)
    print("다운로드 폴더:", d, "\n안의 파일:", os.listdir(d))
    raise last


df = load()
print("shape:", df.shape)
print(df.head(), "\n")

# 결측치: 숫자는 중앙값, 문자는 최빈값
for c in df.columns:
    if df[c].dtype.kind in "ifu":
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna(df[c].mode()[0])

# ★ final_exam_score는 final_grade를 만든 원본 점수다.
#   이걸 feature로 넣고 등급을 맞추면 정확도 100%가 나온다 (= 정답을 그대로 봄).
#   반드시 제외한다. previous_grade(이전 성적)는 정당한 feature라 유지.
X = pd.get_dummies(
    df.drop(columns=["student_id", "final_exam_score", "final_grade"]),
    drop_first=True,
).astype(float)
y = df["final_grade"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

p = X.shape[1]
print(f"feature 개수 p = {p},  sqrt(p) ≈ {np.sqrt(p):.1f}")
print(f"등급 분포:\n{y.value_counts().to_string()}\n")


# decision tree
print("1. 단일 Decision Tree")

tree = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
print(f"정확도: {accuracy_score(y_te, tree.predict(X_te)):.4f}")



# 2. Bagging
#    복원추출로 B개 데이터셋 → 각각 학습 → 다수결
#    bias는 그대로, variance만 줄인다

print("2. Bagging")


bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),  # 어떤 모델이든 가능 (slide 13)
    n_estimators=200,                    # B: 만들 모델 개수
    bootstrap=True,                      # 복원 추출
    oob_score=True,                      # 안 뽑힌 약 37%로 자체 평가
    random_state=SEED,
).fit(X_tr, y_tr)

print(f"정확도    : {accuracy_score(y_te, bag.predict(X_te)):.4f}")
print(f"OOB score : {bag.oob_score_:.4f}   ← 각 트리가 못 본 데이터로 평가한 점수")

print("\nB(모델 개수)에 따른 변화:")
for B in [1, 5, 10, 50, 100, 200]:
    m = BaggingClassifier(DecisionTreeClassifier(), n_estimators=B,
                          random_state=SEED).fit(X_tr, y_tr)
    print(f"  B={B:>3} : {accuracy_score(y_te, m.predict(X_te)):.4f}")


# 3. Random Forest (slide 24~25)
#    Bagging + "각 split마다 m개 feature만 후보"
#    → 트리들이 서로 덜 닮게(decorrelate) 만든다

print("3. Random Forest — m 값 비교")

for label, m in [("m = p (Bagging과 동일)", None),
                 ("m = p/2", p // 2),
                 ("m = sqrt(p)", "sqrt")]:
    rf_ = RandomForestClassifier(n_estimators=300, max_features=m,
                                 random_state=SEED).fit(X_tr, y_tr)
    print(f"{label:>22} : {accuracy_score(y_te, rf_.predict(X_te)):.4f}")

rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                            random_state=SEED).fit(X_tr, y_tr)

print("\nfeature importance (어떤 요인이 등급을 가르는가):")
print(pd.Series(rf.feature_importances_, index=X.columns)
      .sort_values(ascending=False).to_string())


# 4. AdaBoost 
#    약한 모델을 "순차적으로" 학습.
#    앞 모델이 틀린 샘플의 가중치를 올려 다음 모델이 거기 집중하게 한다.

print("4. AdaBoost")

stump = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X_tr, y_tr)
print(f"Decision Stump 1개 (weak learner) : {accuracy_score(y_te, stump.predict(X_te)):.4f}")

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner
    n_estimators=200,                               # T: 순차 학습 횟수
    learning_rate=1.0,
    random_state=SEED,
).fit(X_tr, y_tr)
print(f"AdaBoost (stump, T=200)          : {accuracy_score(y_te, ada.predict(X_te)):.4f}")

ada2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                          n_estimators=200, random_state=SEED).fit(X_tr, y_tr)
print(f"AdaBoost (depth=2, T=200)        : {accuracy_score(y_te, ada2.predict(X_te)):.4f}")

# 각 base learner의 alpha = 성능에 비례하는 가중치 (slide 29)
print(f"\n앞 5개 base learner의 alpha: {np.round(ada.estimator_weights_[:5], 4)}")
print(f"앞 5개 base learner의 error: {np.round(ada.estimator_errors_[:5], 4)}")
K = y.nunique()
print(f"  ※ 슬라이드는 2-class 기준(error < 0.5 필요)이지만 지금은 {K}-class라")
print(f"    sklearn은 SAMME를 쓴다: alpha = log((1-err)/err) + log(K-1)")
print(f"    → 기준선이 0.5가 아니라 1 - 1/K = {1 - 1/K:.2f}")
print(f"    error가 0.5를 넘어도 정상. 기준선에 가까울수록 alpha가 0에 가까워짐.")

# T를 늘리면? (slide 30은 overfitting에 강하다고 하는데 실제로 그런지)
print("\nT에 따른 train/test 정확도:")
for T in [1, 10, 50, 100, 200, 400]:
    m = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                           n_estimators=T, random_state=SEED).fit(X_tr, y_tr)
    print(f"  T={T:>3} : train={accuracy_score(y_tr, m.predict(X_tr)):.4f}  "
          f"test={accuracy_score(y_te, m.predict(X_te)):.4f}")


# ─────────────────────────────────────────────────────────────
# 5. Boosting for Regression (slide 35~36)
#    타깃을 final_exam_score(연속값)로 바꾼다.
#    남은 오차(residual)를 다음 트리가 계속 메워 나가는 방식.
#    learning_rate = 슬라이드의 shrinkage λ
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("5. Boosting for Regression (타깃: final_exam_score)")
print("=" * 55)

y_reg = df["final_exam_score"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=SEED)

base = DecisionTreeRegressor(max_depth=2, random_state=SEED).fit(Xr_tr, yr_tr)
print(f"단일 트리(depth=2) : MSE={mean_squared_error(yr_te, base.predict(Xr_te)):7.3f}  "
      f"R²={r2_score(yr_te, base.predict(Xr_te)):.4f}")

print("\nshrinkage λ(learning_rate) 비교 (T=200 고정):")
for lam in [1.0, 0.3, 0.1, 0.05, 0.01]:
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=lam,
                                    max_depth=2, random_state=SEED).fit(Xr_tr, yr_tr)
    pred = gbr.predict(Xr_te)
    print(f"  λ={lam:<5} : MSE={mean_squared_error(yr_te, pred):7.3f}  R²={r2_score(yr_te, pred):.4f}")
print("  → λ가 작으면 천천히 배우므로 T를 더 키워야 하고, 크면 빠르지만 거칠어짐")


# ─────────────────────────────────────────────────────────────
# 6. 종합 비교 (slide 37)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("6. 종합 비교")
print("=" * 55)

models = {
    "Decision Tree": tree,
    "Bagging": bag,
    "Random Forest": rf,
    "AdaBoost": ada2,
}

rows = []
for name, m in models.items():
    cv = cross_val_score(m, X_tr, y_tr, cv=5)
    rows.append({
        "model": name,
        "test_acc": accuracy_score(y_te, m.predict(X_te)),
        "cv_mean": cv.mean(),
        "cv_std": cv.std(),
    })
print(pd.DataFrame(rows).sort_values("test_acc", ascending=False).to_string(index=False))

print("\nRandom Forest 상세:")
print(classification_report(y_te, rf.predict(X_te), zero_division=0))