import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import random
import warnings
import json
warnings.filterwarnings('ignore')

# 시드 설정
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ANN 모델 클래스 (기존 코드와 유사)
class SimpleANN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout_rate=0.2):
        super(SimpleANN, self).__init__()
        layers = []
        in_features = input_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 데이터 로딩 및 전처리 함수
def load_and_preprocess_data(file_path):
    """실제 데이터 로딩 및 전처리"""
    print(f"Loading data from: {file_path}")
    
    # 데이터 로드
    data = pd.read_csv(file_path)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # 입력 변수 (처음 36개 컬럼)
    input_cols = data.columns[:36]
    # 출력 변수 (마지막 컬럼)
    output_col = data.columns[-1]
    
    X = data[input_cols].values
    y = data[output_col].values
    
    # 출력 변수 전처리 (기존 코드와 동일한 방식)
    import ast
    def process_item(item):
        if isinstance(item, bytes):
            item = item.decode()
        if isinstance(item, list):
            return float(item[0])
        if isinstance(item, str):
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return float(parsed[0])
                else:
                    return float(parsed)
            except Exception:
                try:
                    return float(item)
                except Exception as e:
                    raise ValueError(f"Cannot convert item: {item}") from e
        try:
            return float(item)
        except Exception as e:
            raise ValueError(f"Cannot convert item: {item}") from e
    
    # y 전처리
    if y.dtype == np.dtype('O') or y.dtype.kind in 'SU':
        y = np.array([process_item(item) for item in y])
    else:
        y = y.astype(np.float32)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    
    return X, y, input_cols, output_col

# ANN 벤치마크 함수
def benchmark_ann(X_train, y_train, X_test, y_test, num_iterations=1000000):
    """ANN 모델 벤치마크"""
    print("=== ANN Model Benchmark ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 텐서 변환
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    # 모델 초기화
    model = SimpleANN(input_dim=X_train.shape[1], hidden_dim=128, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 학습 시작 시간
    start_time = time.time()
    
    # 학습 루프
    for iteration in range(num_iterations):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 진행상황 출력 (10%마다)
        if (iteration + 1) % (num_iterations // 10) == 0:
            elapsed = time.time() - start_time
            print(f"ANN Iteration {iteration + 1}/{num_iterations} - Elapsed: {elapsed:.2f}s")
    
    # 총 소요 시간
    total_time = time.time() - start_time
    
    # 최종 평가
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"ANN Total Time: {total_time:.2f} seconds")
    print(f"ANN Final MSE: {mse:.6f}")
    print(f"ANN Final R²: {r2:.6f}")
    print(f"ANN Final MAE: {mae:.6f}")
    
    return {
        'total_time': total_time,
        'iterations_per_second': num_iterations / total_time,
        'final_mse': mse,
        'final_r2': r2,
        'final_mae': mae
    }

# LightGBM 벤치마크 함수
def benchmark_lightgbm(X_train, y_train, X_test, y_test, num_iterations=1000000):
    """LightGBM 모델 벤치마크"""
    print("\n=== LightGBM Model Benchmark ===")
    
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # LightGBM 파라미터
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 학습 시작 시간
    start_time = time.time()
    
    # LightGBM은 iteration 단위가 다르므로, 적절한 num_boost_round 계산
    num_boost_round = min(num_iterations // 1000, 1000)  # 최대 1000라운드로 제한
    
    print(f"LightGBM will run {num_boost_round} boosting rounds")
    
    # 모델 학습
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[test_data],
        callbacks=[lgb.log_evaluation(period=num_boost_round//10)]
    )
    
    # 총 소요 시간
    total_time = time.time() - start_time
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"LightGBM Total Time: {total_time:.2f} seconds")
    print(f"LightGBM Final MSE: {mse:.6f}")
    print(f"LightGBM Final R²: {r2:.6f}")
    print(f"LightGBM Final MAE: {mae:.6f}")
    
    return {
        'total_time': total_time,
        'iterations_per_second': num_boost_round / total_time,
        'final_mse': mse,
        'final_r2': r2,
        'final_mae': mae,
        'num_boost_round': num_boost_round
    }

# 결과 시각화 함수
def plot_benchmark_results(ann_results, lgb_results, output_name, save_path=None):
    """벤치마크 결과 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 실행 시간 비교
    models = ['ANN', 'LightGBM']
    times = [ann_results['total_time'], lgb_results['total_time']]
    
    bars1 = ax1.bar(models, times, color=['skyblue', 'lightcoral'])
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_ylim(0, max(times) * 1.1)
    
    # 시간 값을 막대 위에 표시
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 2. 초당 반복 횟수 비교
    iterations_per_sec = [ann_results['iterations_per_second'], lgb_results['iterations_per_second']]
    
    bars2 = ax2.bar(models, iterations_per_sec, color=['lightgreen', 'orange'])
    ax2.set_title('Iterations per Second', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Iterations/Second')
    
    for bar, iter_val in zip(bars2, iterations_per_sec):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(iterations_per_sec)*0.01,
                f'{iter_val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. MSE 비교
    mse_values = [ann_results['final_mse'], lgb_results['final_mse']]
    
    bars3 = ax3.bar(models, mse_values, color=['lightblue', 'lightpink'])
    ax3.set_title('Mean Squared Error Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MSE')
    
    for bar, mse_val in zip(bars3, mse_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(mse_values)*0.01,
                f'{mse_val:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. R² 비교
    r2_values = [ann_results['final_r2'], lgb_results['final_r2']]
    
    bars4 = ax4.bar(models, r2_values, color=['lightyellow', 'lightcyan'])
    ax4.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R² Score')
    ax4.set_ylim(0, 1)
    
    for bar, r2_val in zip(bars4, r2_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'Model Benchmark Results for {output_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark results saved to {save_path}")
    
    plt.show()

# 단일 데이터셋 벤치마크 함수
def benchmark_single_dataset(file_path, num_iterations=1000000):
    """단일 데이터셋에 대한 벤치마크 실행"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    # 데이터 로드 및 전처리
    X, y, input_cols, output_col = load_and_preprocess_data(file_path)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 벤치마크 실행
    ann_results = benchmark_ann(X_train, y_train, X_test, y_test, num_iterations)
    lgb_results = benchmark_lightgbm(X_train, y_train, X_test, y_test, num_iterations)
    
    # 결과 요약
    print(f"\n{'='*50}")
    print(f"BENCHMARK SUMMARY - {output_col}")
    print(f"{'='*50}")
    print(f"ANN Execution Time: {ann_results['total_time']:.2f} seconds")
    print(f"LightGBM Execution Time: {lgb_results['total_time']:.2f} seconds")
    print(f"Speed Ratio (LightGBM/ANN): {lgb_results['total_time']/ann_results['total_time']:.2f}x")
    print(f"ANN MSE: {ann_results['final_mse']:.6f}")
    print(f"LightGBM MSE: {lgb_results['final_mse']:.6f}")
    print(f"ANN R²: {ann_results['final_r2']:.6f}")
    print(f"LightGBM R²: {lgb_results['final_r2']:.6f}")
    
    # 결과 시각화
    output_name = os.path.basename(file_path).replace('.csv', '')
    save_path = f"benchmark_results_{output_name}.png"
    plot_benchmark_results(ann_results, lgb_results, output_name, save_path)
    
    return {
        'dataset': output_name,
        'output_column': output_col,
        'ann_results': ann_results,
        'lightgbm_results': lgb_results,
        'speed_ratio': lgb_results['total_time']/ann_results['total_time']
    }

# 메인 실행 함수
def main():
    print("=== Real Data Model Benchmark: ANN vs LightGBM ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 시드 설정
    set_seed(42)
    
    # 데이터 파일 리스트 (기존 코드에서 사용하는 파일들)
    data_files = [
        "data_Lmt2.csv",
        "data_Llt2.csv", 
        "data_P_winding1.csv",
        "data_P_winding2.csv",
        "data_P_core.csv",
        "data_B_mean_leg_left.csv",
        "data_B_mean_leg_right.csv",
        "data_Temp_max_core.csv"
    ]
    
    # 현재 디렉토리 기준으로 파일 경로 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    all_results = []
    num_iterations = 1000000  # 100만회
    
    for file_name in data_files:
        file_path = os.path.join(base_path, file_name)
        
        if os.path.exists(file_path):
            try:
                result = benchmark_single_dataset(file_path, num_iterations)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue
        else:
            print(f"File not found: {file_path}")
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    print("OVERALL BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"\n{result['dataset']} ({result['output_column']}):")
        print(f"  Speed Ratio: {result['speed_ratio']:.2f}x")
        print(f"  ANN R²: {result['ann_results']['final_r2']:.4f}")
        print(f"  LightGBM R²: {result['lightgbm_results']['final_r2']:.4f}")
    
    # 평균 성능 계산
    avg_speed_ratio = np.mean([r['speed_ratio'] for r in all_results])
    avg_ann_r2 = np.mean([r['ann_results']['final_r2'] for r in all_results])
    avg_lgb_r2 = np.mean([r['lightgbm_results']['final_r2'] for r in all_results])
    
    print(f"\nAverage Speed Ratio: {avg_speed_ratio:.2f}x")
    print(f"Average ANN R²: {avg_ann_r2:.4f}")
    print(f"Average LightGBM R²: {avg_lgb_r2:.4f}")
    
    # 결과를 JSON으로 저장
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'num_iterations': num_iterations,
        'total_datasets': len(all_results),
        'average_speed_ratio': avg_speed_ratio,
        'average_ann_r2': avg_ann_r2,
        'average_lightgbm_r2': avg_lgb_r2,
        'detailed_results': all_results
    }
    
    with open('real_data_benchmark_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to real_data_benchmark_results.json")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 