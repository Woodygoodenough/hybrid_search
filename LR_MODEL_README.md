# Logistic Regression-Based Adaptive Search

## 개요

이 시스템은 **Logistic Regression 모델**을 사용하여 PRE-search와 POS-search 중 어느 것이 더 빠를지 자동으로 예측하고 선택합니다.

**핵심 최적화**: pkl 파일 로딩 오버헤드를 제거하기 위해 **모델 파라미터를 search.py에 직접 하드코딩**합니다!

---

## 📁 파일 구조

### 1. **model_evaluation.py**
- 4개 모델 학습 및 비교 (Simple Rule-Based, Advanced Rule-Based, Logistic Regression, Decision Tree)
- 각 모델의 inference time 측정
- 최적 모델 선택 (Composite Score: 70% F1 + 20% Accuracy + 10% Speed)
- **LR 모델 파라미터를 search.py에 자동으로 주입**

### 2. **search.py**
- 하드코딩된 LR 모델 파라미터 (`_SCALER_MEAN`, `_SCALER_SCALE`, `_LR_COEF`, `_LR_INTERCEPT`)
- `_predict_pos_faster_hardcoded()`: 빠른 inference 함수 (pkl 로딩 없음!)
- `lr_based_adap_search()`: LR 기반 adaptive search 메인 함수

### 3. **train_and_save_model.py** (선택 사항)
- pkl 파일로 모델을 저장하는 방식 (하지만 우리는 하드코딩 방식 사용)

### 4. **example_lr_search.py**
- lr_based_adap_search 사용 예시

---

## 🚀 사용 방법

### Step 1: 모델 학습 및 파라미터 하드코딩

```bash
python model_evaluation.py
```

**이 스크립트가 자동으로:**
1. 4개 모델 학습 및 비교
2. Logistic Regression이 최적 모델로 선택됨
3. **LR 모델의 파라미터를 search.py에 직접 주입**
4. 시각화 파일 생성 (model_comparison.png, confusion_matrices.png, decision_tree.png)
5. 참고용 lr_model_hardcoded.py 파일 생성

**실행 후 search.py는 다음과 같이 업데이트됩니다:**
```python
# Before
_SCALER_MEAN = None
_SCALER_SCALE = None
_LR_COEF = None
_LR_INTERCEPT = None

# After (자동으로 채워짐)
_SCALER_MEAN = np.array([2697.0833333333, 82298.1379310345, 0.5486542529])
_SCALER_SCALE = np.array([3297.0814064447, 35386.0087430147, 0.2359073916])
_LR_COEF = np.array([-0.1234567890, 2.3456789012, 1.2345678901])
_LR_INTERCEPT = 0.5678901234
```

### Step 2: 검색에 사용

```python
from search import Search
from shared_dataclasses import Predicate

# Search 인스턴스 생성 (pkl 로딩 없음!)
search = Search()

# 쿼리 임베딩
query = "machine learning algorithms"
query_embedding = search.embedder.encode_query(query)

# Predicates
predicates = [
    Predicate(key="token_count", value=400, operator=">="),
]

# LR 기반 adaptive search (자동으로 PRE/POS 선택)
results = search.lr_based_adap_search(query_embedding, predicates, k=10)

# 결과 확인
print(f"Found {len(results.results)} results")
print(results.to_df(show_cols=['item_id', 'title']))
```

---

## ⚡ 성능 비교

### 기존 방식 (pkl 로딩)
```python
# 매번 검색할 때마다 pkl 파일 로딩
with open('lr_model.pkl', 'rb') as f:
    model_package = pickle.load(f)  # ⏱️ 오버헤드!

model = model_package['model']
scaler = model_package['scaler']
prediction = model.predict(scaler.transform(features))
```

### 하드코딩 방식 (현재)
```python
# 파라미터가 이미 메모리에 있음 (파일 I/O 없음!)
features_scaled = (features - _SCALER_MEAN) / _SCALER_SCALE
logit = np.dot(features_scaled, _LR_COEF) + _LR_INTERCEPT
probability_pos = 1 / (1 + np.exp(-logit))  # ⚡ 초고속!
```

**장점:**
- ✅ pkl 파일 로딩 오버헤드 제거
- ✅ 의존성 감소 (pickle 모듈 불필요)
- ✅ 코드 한 곳에 모든 로직 집중
- ✅ 초기화 시간 단축

---

## 🧠 작동 원리

### 1. 예측 프로세스

```
Query + Predicates
        ↓
Estimate survivors (histogram)
        ↓
Features: [k, num_survivors, selectivity]
        ↓
Scale features
        ↓
Logistic Regression inference (hardcoded)
        ↓
Prediction: PRE or POS?
        ↓
Execute chosen method
```

### 2. 하드코딩된 예측 함수

```python
def _predict_pos_faster_hardcoded(k, num_survivors, total_docs=150000):
    # Calculate selectivity
    selectivity = num_survivors / total_docs

    # Create and scale features
    features = np.array([k, num_survivors, selectivity])
    features_scaled = (features - _SCALER_MEAN) / _SCALER_SCALE

    # Logistic regression
    logit = np.dot(features_scaled, _LR_COEF) + _LR_INTERCEPT
    probability_pos = 1 / (1 + np.exp(-logit))

    return probability_pos > 0.5
```

---

## 📊 모델 성능

Logistic Regression이 최적 모델로 선택된 이유:

| Model | Accuracy | F1 Score | Inference Time | Composite Score |
|-------|----------|----------|----------------|-----------------|
| **Logistic Regression** | **~0.95** | **~0.96** | **~0.5 ms** | **~0.95** |
| Decision Tree | ~0.94 | ~0.95 | ~0.8 ms | ~0.94 |
| Advanced Rule-Based | ~0.90 | ~0.91 | ~2.0 ms | ~0.88 |
| Simple Rule-Based | ~0.85 | ~0.86 | ~0.1 ms | ~0.83 |

---

## 🔧 재학습 방법

모델을 재학습하고 싶다면:

```bash
# 1. 새로운 데이터로 timed_results.csv 업데이트
# 2. model_evaluation.py 실행
python model_evaluation.py

# 3. 자동으로 search.py가 업데이트됨!
```

---

## 📝 예시 실행

```bash
# 전체 워크플로우 테스트
python example_lr_search.py
```

**출력 예시:**
```
================================================================================
LR-BASED ADAPTIVE SEARCH EXAMPLE
================================================================================
Query: machine learning algorithms
Predicates: [Predicate(key='token_count', value=400, operator='>=')]
k: 10

Prediction: Using POS search (estimated survivors: 5000)

Found 10 results
Is k satisfied: True

   item_id                                    title  similarity
0    12345                Machine Learning Basics    0.92
1    67890    Deep Learning for Natural Language    0.89
...
```

---

## 🎯 핵심 포인트

1. **pkl 로딩 제로**: 모델 파라미터가 search.py에 하드코딩되어 있어 파일 I/O 없음
2. **자동 업데이트**: `model_evaluation.py` 실행 시 search.py 자동 업데이트
3. **빠른 inference**: 단순 numpy 연산만으로 예측 (sklearn 객체 없음)
4. **높은 정확도**: F1 Score ~0.96으로 PRE/POS 선택 최적화

---

## 🚨 주의사항

- `model_evaluation.py`를 실행하면 **search.py가 자동으로 수정**됩니다
- Git에서 search.py 변경사항 확인 후 커밋하세요
- 모델 재학습 시 이전 파라미터는 덮어씌워집니다

---

## 📚 참고 자료

- Features: `[k, num_survivors, selectivity]`
- Total docs: 150,000
- Model: Logistic Regression with StandardScaler
- Training: 80/20 split, stratified sampling

끝! 🎉
