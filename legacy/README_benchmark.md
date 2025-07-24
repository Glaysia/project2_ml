# ANN vs LightGBM Model Benchmark

이 프로젝트는 ANN(Artificial Neural Network)과 LightGBM 모델의 성능을 비교하는 벤치마크 도구입니다. 100만회 회귀 수행 시간을 측정하여 두 모델의 성능 차이를 분석합니다.

## 📁 파일 구조

```
├── model_benchmark.py          # 합성 데이터를 사용한 벤치마크
├── real_data_benchmark.py      # 실제 데이터를 사용한 벤치마크  
├── run_benchmark.py           # 통합 실행 스크립트
└── README_benchmark.md        # 이 파일
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install torch lightgbm scikit-learn matplotlib seaborn pandas numpy
```

### 2. 벤치마크 실행

#### 기본 실행 (둘 다 실행)
```bash
python run_benchmark.py
```

#### 합성 데이터만 실행
```bash
python run_benchmark.py --synthetic
```

#### 실제 데이터만 실행
```bash
python run_benchmark.py --real
```

#### 빠른 테스트 (1만회 반복)
```bash
python run_benchmark.py --quick
```

#### 반복 횟수 지정
```bash
python run_benchmark.py --real --iterations 500000
```

## 📊 벤치마크 내용

### 측정 지표
- **실행 시간**: 각 모델이 100만회 반복을 완료하는데 걸리는 시간
- **초당 반복 횟수**: 처리 속도 비교
- **MSE (Mean Squared Error)**: 예측 정확도
- **R² Score**: 모델 성능 지표
- **MAE (Mean Absolute Error)**: 절대 오차

### 모델 구성

#### ANN (PyTorch)
- 3층 신경망 (128 유닛)
- ReLU 활성화 함수
- Batch Normalization
- Dropout (20%)
- Adam 옵티마이저

#### LightGBM
- GBDT (Gradient Boosting Decision Trees)
- 31개 리프 노드
- 학습률 0.05
- Feature fraction 0.9
- Bagging fraction 0.8

## 📈 결과 해석

### 실행 시간 비교
- **ANN**: GPU 가속을 활용한 빠른 학습
- **LightGBM**: CPU 기반의 효율적인 트리 모델

### 정확도 비교
- **ANN**: 복잡한 비선형 관계 학습에 유리
- **LightGBM**: 구조화된 데이터에서 우수한 성능

## 📁 출력 파일

### JSON 결과 파일
- `benchmark_results.json`: 합성 데이터 벤치마크 결과
- `real_data_benchmark_results.json`: 실제 데이터 벤치마크 결과

### 시각화 파일
- `model_benchmark_results.png`: 합성 데이터 결과 그래프
- `benchmark_results_*.png`: 각 실제 데이터셋별 결과 그래프

## 🔧 고급 사용법

### 개별 스크립트 실행

#### 합성 데이터 벤치마크
```bash
python model_benchmark.py
```

#### 실제 데이터 벤치마크
```bash
python real_data_benchmark.py
```

### 코드 수정

#### 반복 횟수 변경
```python
# model_benchmark.py 또는 real_data_benchmark.py에서
num_iterations = 500000  # 원하는 반복 횟수로 변경
```

#### 모델 파라미터 조정
```python
# ANN 모델 수정
model = SimpleANN(input_dim=X_train.shape[1], 
                  hidden_dim=256,  # 은닉층 크기
                  num_layers=4,    # 층 수
                  dropout_rate=0.3) # 드롭아웃 비율

# LightGBM 파라미터 수정
params = {
    'num_leaves': 63,        # 리프 노드 수
    'learning_rate': 0.1,    # 학습률
    'feature_fraction': 0.8, # 특성 샘플링 비율
    # ... 기타 파라미터
}
```

## ⚠️ 주의사항

1. **GPU 메모리**: ANN 벤치마크는 GPU 메모리를 사용합니다. 충분한 메모리가 있는지 확인하세요.
2. **실행 시간**: 100만회 반복은 상당한 시간이 소요될 수 있습니다. 빠른 테스트를 위해 `--quick` 옵션을 사용하세요.
3. **데이터 파일**: 실제 데이터 벤치마크를 위해서는 해당 CSV 파일들이 필요합니다.

## 🐛 문제 해결

### ImportError 발생 시
```bash
pip install --upgrade torch lightgbm scikit-learn matplotlib seaborn
```

### GPU 메모리 부족 시
```python
# CPU 사용으로 강제 설정
device = torch.device("cpu")
```

### 데이터 파일 없음
실제 데이터 벤치마크를 실행하려면 다음 파일들이 필요합니다:
- `data_Lmt2.csv`
- `data_Llt2.csv`
- `data_P_winding1.csv`
- `data_P_winding2.csv`
- `data_P_core.csv`
- `data_B_mean_leg_left.csv`
- `data_B_mean_leg_right.csv`
- `data_Temp_max_core.csv`

## 📝 예상 결과

일반적으로 다음과 같은 결과를 기대할 수 있습니다:

- **ANN**: GPU 환경에서 빠른 학습 속도, 복잡한 패턴 학습에 유리
- **LightGBM**: CPU 환경에서도 효율적, 구조화된 데이터에서 우수한 성능

실제 결과는 하드웨어 사양과 데이터 특성에 따라 달라질 수 있습니다. 