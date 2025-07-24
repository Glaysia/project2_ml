import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import numpy as np
import os
import random
import wandb
import matplotlib.pyplot as plt
import pickle
import datetime
import json
import shutil  # 파일 복사용
import uuid

import ast

torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ================================================
# Data 클래스
# ================================================
class Data:
    def __init__(self):
        self.raw_data = None
        self.X = None
        self.Y = None
        self.scaler = None
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None
        self.test_X = None
        self.test_Y = None

    def load_data(self, path):
        self.raw_data = pd.read_csv(path)
        # self.raw_data.dropna(inplace=True)
        print("Data loaded successfully!")
        print(f"Data Shape: {self.raw_data.shape}")
        print(f"Columns: {self.raw_data.columns.tolist()}")

    def split_data(self, input_cols, output_cols):
        self.X = self.raw_data[input_cols]
        self.Y = self.raw_data[output_cols]

    def normalize_data(self):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        # scaler는 기존 saved_models 폴더에 저장
        os.makedirs(f"{PATH}/saved_models/{ARTIFACT_NM}", exist_ok=True)
        with open(f"{PATH}/saved_models/{ARTIFACT_NM}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        print("Scaler saved to saved_models/scaler.pkl")
    
    def split_train_val_test(self, test_size=0.2, val_size=0.2, random_state=42):
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=relative_val_size, random_state=random_state)
        self.train_X = X_train
        self.train_Y = Y_train
        self.val_X = X_val
        self.val_Y = Y_val
        self.test_X = X_test
        self.test_Y = Y_test
        print(f"Data split into: Train {self.train_X.shape}, Val {self.val_X.shape}, Test {self.test_X.shape}")

# ================================================
# ANN 클래스 (Data 상속)
# ================================================
class ANN(Data):
    def __init__(self):
        super().__init__()
        self.models = {}

    def build_model(self, input_dim, n_layers, n_units, activation='relu', dropout_rate=0.3):
        layers_list = []
        in_features = input_dim
        for i in range(n_layers):
            layers_list.append(nn.Linear(in_features, n_units))
            if activation == 'relu':
                layers_list.append(nn.ReLU())
            elif activation == 'tanh':
                layers_list.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers_list.append(nn.LeakyReLU())
            else:
                layers_list.append(nn.ReLU())
            layers_list.append(nn.BatchNorm1d(n_units))
            layers_list.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers_list.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers_list)

    

    def convert_data(self, X, y, device=None, view_y=True):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if device:
            X_tensor = X_tensor.to(device)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        # 각 항목을 float으로 변환하는 함수
        def process_item(item):
            # bytes형이면 decode
            if isinstance(item, bytes):
                item = item.decode()
            # 리스트이면 첫 번째 요소 추출 후 float 변환
            if isinstance(item, list):
                return float(item[0])
            # 문자열인 경우
            if isinstance(item, str):
                try:
                    # 문자열이 리스트 형태라면 ast.literal_eval로 변환
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
            # 그 외의 경우 float 변환 시도
            try:
                return float(item)
            except Exception as e:
                raise ValueError(f"Cannot convert item: {item}") from e

        # y가 문자열 또는 객체 타입이면 각 요소를 처리하여 float 배열로 변환
        if y.dtype == np.dtype('O') or y.dtype.kind in 'SU':
            y = np.array([process_item(item) for item in y])
        else:
            y = y.astype(np.float32)
        
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if view_y:
            y_tensor = y_tensor.view(-1, 1)
        if device:
            y_tensor = y_tensor.to(device)
        return X_tensor, y_tensor

    def train_one_epoch(self, model, optimizer, train_loader, device, criterion):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate_epoch(self, model, X_val_tensor, y_val_tensor, device, criterion):
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            loss = criterion(outputs, y_val_tensor).item()
        return loss

    def evaluate_split(self, model, X, y, device):
        X_tensor, _ = self.convert_data(X, y, device=device, view_y=True)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor).detach().cpu().numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_true = y.values.flatten()
        else:
            y_true = np.array(y).flatten()
        mae = nn.L1Loss()(torch.tensor(y_true, dtype=torch.float32).view(-1,1),
                          torch.tensor(y_pred, dtype=torch.float32)).item()
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse, "R2": r2}, y_pred

    def plot_scatter(self, output_col, X_data, y_data, save_path=None, metrics=None):
        """
        output_col: 출력 변수 이름 (모델 딕셔너리 키)
        X_data, y_data: 실제 데이터 (예측에 사용)
        save_path: 결과 이미지 저장 경로 (선택)
        metrics: 미리 계산된 평가 지표 딕셔너리, 예:
                 {"R2": 0.95, "MAE": 0.1, "MSE": 0.02, "RMSE": 0.14, "MAPE": 0.05}
                 만약 None이면 내부에서 sklearn을 사용해 재계산함.
        """
        import seaborn as sns
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.models.get(output_col, None)
        if model is None:
            print(f"No trained model found for {output_col}.")
            return

        # 데이터 변환
        X_tensor, _ = self.convert_data(X_data, y_data, device=device, view_y=False)
        with torch.no_grad():
            y_pred_tensor = model(X_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        if isinstance(y_data, (pd.Series, pd.DataFrame)):
            y_true = y_data.values.flatten()
        else:
            y_true = np.array(y_data).flatten()
        
        # metrics가 제공되지 않으면 내부에서 계산
        if metrics is None:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            metrics = {
                "R2": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape
            }
        
        # 플롯 스타일 설정
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 8))
        
        # 산점도
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', label="Data points")
        
        # 1:1 이상적인 선 (대각선)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit")
        
        # 회귀선 추가: 1차 선형 회귀선 계산
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        reg_line = slope * np.array([min_val, max_val]) + intercept
        plt.plot([min_val, max_val], reg_line, 'b-', lw=2, label="Regression Line")
        
        # 축, 제목, 범례 설정
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(f"Scatter Plot for {output_col}", fontsize=14)
        plt.legend(fontsize=10)
        
        # 평가 지표 텍스트 박스 추가
        metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Scatter plot saved to {save_path}")
        plt.close()

# ================================================
# Best Model Artifact 업데이트 함수 (덮어쓰기 방식)
# ================================================
def update_best_model_artifact(splits, model_file, config_file, scatter_files=None):
    """
    splits: {"train": metrics_train, "val": metrics_val, "test": metrics_test, "best_val_loss": best_val_loss}
    model_file, config_file: 임시 저장된 모델 및 config 파일 경로
    scatter_files: scatter plot 파일 경로 리스트
    각 metric(key: 예, "test_R2", "train_MAE", "best_val_loss" 등)별로,
    {PATH}/best_model/{ARTIFACT_NM} 폴더 내에 저장하며,
    새로운 최고 모델이 나오면 해당 폴더의 파일을 덮어씁니다.
    Artifact 업데이트는 개선이 있을 때만 수행합니다.
    """
    best_model_dir = os.path.join(PATH, "best_model", ARTIFACT_NM)
    os.makedirs(best_model_dir, exist_ok=True)
    best_values_file = os.path.join(best_model_dir, "best_values.json")
    
    if os.path.exists(best_values_file):
        with open(best_values_file, "r") as f:
            best_values = json.load(f)
    else:
        best_values = {}
    
    improved_any = False
    improvements_description = []
    
    for split, metrics in splits.items():
        if split == "best_val_loss":
            key = "best_val_loss"
            current_value = metrics
            is_improved = False
            if key not in best_values or current_value < best_values[key]:
                is_improved = True
            if is_improved:
                old_val_str = "N/A" if key not in best_values else f"{float(best_values[key]):.4f}"
                improvements_description.append(f"{key} improved from {old_val_str} to {current_value:.4f}")
                best_values[key] = current_value
                improved_any = True
                key_dir = os.path.join(best_model_dir, key)
                os.makedirs(key_dir, exist_ok=True)
                model_dest = os.path.join(key_dir, "model.pth")
                config_dest = os.path.join(key_dir, "config.json")
                shutil.copyfile(model_file, model_dest)
                shutil.copyfile(config_file, config_dest)
                print(f"Updated best {key} model with value: {current_value:.4f}")
                if scatter_files:
                    fixed_names = ["train_scatter.png", "val_scatter.png", "test_scatter.png"]
                    for scatter_path, fixed_name in zip(scatter_files, fixed_names):
                        if os.path.exists(scatter_path):
                            dest = os.path.join(key_dir, fixed_name)
                            shutil.copyfile(scatter_path, dest)
                            print(f"Updated scatter plot for {key} in {dest}")
        else:
            for metric_name, current_value in metrics.items():
                key = f"{split}_{metric_name}"
                is_improved = False
                if key not in best_values:
                    is_improved = True
                else:
                    best_val = best_values[key]
                    if metric_name == "R2":
                        if current_value > best_val:
                            is_improved = True
                    else:
                        if current_value < best_val:
                            is_improved = True
                if is_improved:
                    old_val_str = "N/A" if key not in best_values else f"{float(best_values[key]):.4f}"
                    improvements_description.append(f"{key} improved from {old_val_str} to {current_value:.4f}")
                    best_values[key] = current_value
                    improved_any = True
                    key_dir = os.path.join(best_model_dir, key)
                    os.makedirs(key_dir, exist_ok=True)
                    model_dest = os.path.join(key_dir, "model.pth")
                    config_dest = os.path.join(key_dir, "config.json")
                    shutil.copyfile(model_file, model_dest)
                    shutil.copyfile(config_file, config_dest)
                    print(f"Updated best {key} model with value: {current_value:.4f}")
                    if scatter_files:
                        fixed_names = ["train_scatter.png", "val_scatter.png", "test_scatter.png"]
                        for scatter_path, fixed_name in zip(scatter_files, fixed_names):
                            if os.path.exists(scatter_path):
                                dest = os.path.join(key_dir, fixed_name)
                                shutil.copyfile(scatter_path, dest)
                                print(f"Updated scatter plot for {key} in {dest}")
    
    with open(best_values_file, "w") as f:
        json.dump(best_values, f)
    
    description_str = " ; ".join(improvements_description) if improvements_description else "No improvements this run."
    
    if improved_any:
        run = wandb.run
        art = wandb.Artifact(f"{ARTIFACT_NM}", type="model", description=description_str, metadata=best_values)
        art.add_dir(best_model_dir, name="best_model")
        run.log_artifact(art)
        print(f"Logged best_model artifact with description: {description_str}")
    else:
        print("No improvements this run. Best model artifact not updated.")
    
    return improved_any

# ================================================
# Sweep용 단일 실험 함수 (wandb.init 내부에서 실행)
# ================================================
def sweep_train():
    """
    wandb Sweep 에이전트에서 호출되는 함수.
    wandb.config에 설정된 하이퍼파라미터로 하나의 Run을 실행합니다.
    """
    wandb.init(project=WANDB_PR, entity="schwalbe-university-of-seoul")
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global ann_model_instance
    ann = ann_model_instance

    # 고유 식별자 생성
    unique_id = str(uuid.uuid4())[:8]

    # 데이터 준비
    X_train = ann.train_X
    y_train = ann.train_Y[ann.Y_col[0]]
    X_val   = ann.val_X
    y_val   = ann.val_Y[ann.Y_col[0]]
    X_test  = ann.test_X
    y_test  = ann.test_Y[ann.Y_col[0]]
    
    X_train_tensor, y_train_tensor = ann.convert_data(X_train, y_train, device=None, view_y=True)
    X_val_tensor, y_val_tensor     = ann.convert_data(X_val, y_val, device=device, view_y=True)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                               num_workers=NUM_WORKER, pin_memory=True)
    
    # 모델 생성
    model = ann.build_model(
        input_dim = X_train.shape[1],
        n_layers  = config.n_layers,
        n_units   = config.n_units,
        activation= "relu",
        dropout_rate = config.dropout_rate
    )
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    
    best_val_loss = np.inf
    patience_counter = 0
    for epoch in range(config.epochs):
        train_loss = ann.train_one_epoch(model, optimizer, train_loader, device, criterion)
        val_loss = ann.validate_epoch(model, X_val_tensor, y_val_tensor, device, criterion)
        scheduler.step(val_loss)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    metrics_train, _ = ann.evaluate_split(model, X_train, y_train, device)
    metrics_val, _   = ann.evaluate_split(model, X_val, y_val, device)
    metrics_test, _  = ann.evaluate_split(model, X_test, y_test, device)
    wandb.log({
        "final_train_metrics": metrics_train,
        "final_val_metrics": metrics_val,
        "final_test_metrics": metrics_test,
        "best_val_loss": best_val_loss
    })
    
    # 모델 저장: 고유 식별자를 포함하여 saved_models/{ARTIFACT_NM} 폴더에 저장
    temp_dir = os.path.join(PATH, "saved_models", ARTIFACT_NM)
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file  = os.path.join(temp_dir, f"temp_model_{unique_id}_{timestamp}.pth")
    config_file = os.path.join(temp_dir, f"temp_config_{unique_id}_{timestamp}.json")
    torch.save(model.state_dict(), model_file)
    with open(config_file, "w") as f:
        json.dump(dict(config), f)
    
    # 평가 지표(splits) 구성 및 best_val_loss 추가
    splits = {"train": metrics_train, "val": metrics_val, "test": metrics_test}
    splits["best_val_loss"] = best_val_loss
    
    # 개선 여부 판단 (기존 best_values.json은 best_model/{ARTIFACT_NM} 폴더에 있음)
    base_best_values_file = os.path.join(PATH, "best_model", ARTIFACT_NM, "best_values.json")
    if os.path.exists(base_best_values_file):
        with open(base_best_values_file, "r") as f:
            base_best_values = json.load(f)
    else:
        base_best_values = {}
    
    improved_flag = False
    for split, metrics in splits.items():
        if split == "best_val_loss":
            key = "best_val_loss"
            if key not in base_best_values or metrics < base_best_values[key]:
                improved_flag = True
        else:
            for metric_name, current_value in metrics.items():
                key = f"{split}_{metric_name}"
                if key not in base_best_values:
                    improved_flag = True
                else:
                    best_value = base_best_values[key]
                    if metric_name == "R2":
                        if current_value > best_value:
                            improved_flag = True
                    else:
                        if current_value < best_value:
                            improved_flag = True
    
    scatter_files = None
    if improved_flag:
        scatter_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        train_scatter = os.path.join(temp_dir, f"{ann.Y_col[0]}_train_scatter_{scatter_timestamp}_{unique_id}.png")
        val_scatter   = os.path.join(temp_dir, f"{ann.Y_col[0]}_val_scatter_{scatter_timestamp}_{unique_id}.png")
        test_scatter  = os.path.join(temp_dir, f"{ann.Y_col[0]}_test_scatter_{scatter_timestamp}_{unique_id}.png")
        ann.plot_scatter(ann.Y_col[0], X_train, y_train, save_path=train_scatter, metrics=metrics_train)
        ann.plot_scatter(ann.Y_col[0], X_val, y_val, save_path=val_scatter, metrics=metrics_val)
        ann.plot_scatter(ann.Y_col[0], X_test, y_test, save_path=test_scatter, metrics=metrics_test)
        scatter_files = [train_scatter, val_scatter, test_scatter]
    
    update_best_model_artifact(splits, model_file, config_file, scatter_files=scatter_files)
    
    if improved_flag:
        ann.models[ann.Y_col[0]] = model

    wandb.finish()


# ================================================
# Sweep ID 저장 및 불러오기 함수
# ================================================
def get_or_create_sweep_id(sweep_config, project):

    if os.path.exists(SWEEP_ID_PATH):
        with open(SWEEP_ID_PATH, "r") as f:
            try:
                sweep_data = json.load(f)
            except json.decoder.JSONDecodeError:
                sweep_data = {}
    else:
        sweep_data = {}

    key = f"sweep_id_{SWEEP_NM}"
    if key in sweep_data:
        print(f"Loaded existing sweep_id: {sweep_data[key]}")
        return sweep_data[key]
    sweep_id = wandb.sweep(sweep_config, project=project, entity="schwalbe-university-of-seoul")
    sweep_data[key] = sweep_id
    with open(SWEEP_ID_PATH, "w") as f:
        json.dump(sweep_data, f)
    print(f"Created new sweep_id: {sweep_id}")
    return sweep_id

# ================================================
# 메인 실행부
# ================================================
NUM_WORKER = 1
NORMALIZE = False
BASE_PATH = "/gpfs/home1/r1jae262/jupyter/TIE_SST_MFT_2025/MFT_Maxwell_regression"

# 모델 설정 리스트
MODEL_CONFIGS = [
    {
        "name": "Lmt_250624",
        "file": "data_Lmt2.csv",
        "wandb_project": "SST_tuning_test"
    },
    {
        "name": "Llt_250624",
        "file": "data_Llt2.csv",
        "wandb_project": "SST_tuning_test"
    },
    {
        "name": "P_winding1_250624",
        "file": "data_P_winding1.csv",
        "wandb_project": "SST_tuning_test"
    },
    {
        "name": "P_winding2_250624",
        "file": "data_P_winding2.csv",
        "wandb_project": "SST_tuning_test"
    },
    {
        "name": "P_core_250624",
        "file": "data_P_core.csv",
        "wandb_project": "SST_tuning_test"
    },
    {
        "name": "B_mean_leg_left_250624",
        "file": "data_B_mean_leg_left.csv",
        "wandb_project": "SST_tuning_test"
    },
    {
        "name": "B_mean_leg_right_250624",
        "file": "data_B_mean_leg_right.csv",
        "wandb_project": "SST_tuning_test"
    },
    {
        "name": "temp_max_core_250624",
        "file": "data_Temp_max_core.csv",
        "wandb_project": "SST_tuning_test"
    }
    
    # 필요한 만큼 모델 설정 추가
]

def run_pipeline(seed, model_config):
    global PATH, SWEEP_NM, FILE_NAME, WANDB_PR, SWEEP_ID_PATH, ARTIFACT_NM
    
    PATH = BASE_PATH
    SWEEP_NM = model_config["name"]
    FILE_NAME = model_config["file"]
    WANDB_PR = model_config["wandb_project"]
    SWEEP_ID_PATH = f"{PATH}/wandb_id/sweep_id.json"
    ARTIFACT_NM = f"{WANDB_PR}_{SWEEP_NM}"

    print(f"\n=== Running with seed {seed} for model {SWEEP_NM} ===")
    set_seed(seed)
    ann_model = ANN()
    data_path = f"{PATH}/{FILE_NAME}"
    ann_model.load_data(data_path)
    input_cols = ann_model.raw_data.columns[0:36]  # 입력 변수
    output_cols = ann_model.raw_data.iloc[:, [36]].columns.tolist()  # 출력 변수
    ann_model.split_data(input_cols, output_cols)
    ann_model.Y_col = output_cols  # 출력 컬럼 이름 저장
    
    Y_converted = ann_model.Y[output_cols[0]]
    Y_processed = Y_converted
    ann_model.Y = pd.DataFrame({output_cols[0]: Y_processed})

    if NORMALIZE:
        ann_model.normalize_data()
    ann_model.split_train_val_test(test_size=0.2, val_size=0.2, random_state=seed)
    global ann_model_instance
    ann_model_instance = ann_model

def run_sweep():
    sweep_config = {
        'name': SWEEP_NM,
        'method': 'bayes',
        'metric': {'name': 'best_val_loss', 'goal': 'minimize'},
        'parameters': {
            'lr': {'values': [1e-5, 1e-4, 1e-3, 1e-2]},
            'n_layers': {'values': [2, 3, 4, 5, 6]},
            'n_units': {'values': [32, 64, 96, 128, 192, 256]},
            'dropout_rate': {'values': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
            'batch_size': {'values': [32, 64, 92, 128, 192, 256, 384, 512, 768, 1024]},
            'epochs': {'values': [50, 100, 150, 200, 300, 500]},
            'patience': {'values': [10, 20, 30, 40]}
        }
    }
    sweep_id = get_or_create_sweep_id(sweep_config, project=WANDB_PR)
    wandb.agent(sweep_id, function=sweep_train, count=10, project=WANDB_PR, entity="schwalbe-university-of-seoul")

if __name__ == '__main__':
    seed = random.randint(1, 10000)
    for i in range(10):
        for model_config in MODEL_CONFIGS:
            run_pipeline(seed, model_config)
            run_sweep()
