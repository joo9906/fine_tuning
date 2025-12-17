import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import mlflow

# MLflow 실험 이름 설정
MLFLOW_EXPERIMENT_NAME = "AirsMedical_SynthSeg_Spine"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 데이터 경로 및 모델 저장 경로 설정 (실제 학습 시 로컬 컴퓨터의 경로가 꼬여 절대 경로로 학습하였으나 제대로 바꿔 두었습니다. 만약 경로가 꼬일 경우 파일명에 맞게 수정하시면 됩니다.)
SYNTH_IMG_DIR = "SYNTH_T1_SEG/images"
SYNTH_MASK_DIR = "SYNTH_T1_SEG/masks"
REAL_IMG_DIR = "SPIDER_T1_train/images"
REAL_MASK_DIR = "SPIDER_T1_train/masks"

OUTPUT_PATH = "models/saved"
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"Model checkpoints will be saved to: {OUTPUT_PATH}")


# --- 주어진 Unet 모델 ---

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Lightweight U-Net for 2D segmentation."""
    # NOTE: num_classes를 과제 요구사항인 4개(BG, Vertebra, Disc, Canal)로 설정합니다.
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_features: int = 16):
        super().__init__()
        f = base_features 
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.bottleneck = ConvBlock(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)
        self.out_conv = nn.Conv2d(f, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

# --- 손실 함수 ---

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        # 배경(클래스 0)을 제외하고 평균 (SynthSeg 접근 방식)
        return 1 - dice[:, 1:].mean() 

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)



#이미지와 마스크를 로드하고 정규화하는 데이터셋
class SpineDataset(Dataset):
    MAX_SAMPLES = 50000 # 총 5만개의 PNG 이미지로 학습

    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # 1. 모든 파일 이름 로드
        all_filenames = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        # 2. 크기 제한 및 랜덤 샘플링
        if len(all_filenames) > self.MAX_SAMPLES:
            print(f"Dataset too large ({len(all_filenames)}). Subsampling to {self.MAX_SAMPLES}...")
            # NumPy를 사용하여 랜덤 샘플링
            self.filenames = np.random.choice(all_filenames, self.MAX_SAMPLES, replace=False).tolist()
        else:
            self.filenames = all_filenames
            
        print(f"✅ Training data loaded: {len(self.filenames)} samples")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)
        
        # 이미지 로드 (PIL Image 객체로)
        img_pil = Image.open(img_path).convert('L')
        mask_pil = Image.open(mask_path)
        
        # ----------------------------------------------------
        # 고정 크기로 리사이즈 (256x256), 학습과 평가 데이터셋의 크기만 맞추면 되지만 사람이 봤을 때 알아보기 힘들 정도로 가로폭이 작아 임의로 조정하였습니다.
        TARGET_SIZE = (256, 256) 
        
        # 보간(interpolation)을 사용하여 리사이즈
        img_pil = img_pil.resize(TARGET_SIZE, Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
        
        # 최근접 이웃(NEAREST) 방식으로 리사이즈 (클래스 값 보존)
        mask_pil = mask_pil.resize(TARGET_SIZE, Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)
        # ----------------------------------------------------

        # NumPy 배열로 변환
        img = np.array(img_pil, dtype=np.float32)
        mask = np.array(mask_pil, dtype=np.int64)

        # Min-Max 정규화 (0~1 범위)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # PyTorch Tensor로 변환: Image(1, H, W), Mask(H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    run_name: str,
    output_path: str,
    learning_rate: float
):    
    with mlflow.start_run(run_name=run_name) as run:    # mlflow로 추적
        print(f"\n--- Starting MLflow Run: {run_name} (Run ID: {run.info.run_id}) ---")
        
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            "mode": run_name.split('_')[2] if run_name.startswith('Model') else 'unknown',
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": type(optimizer).__name__,
            "loss_function": type(criterion).__name__,
            "num_classes": 4, 
            "dataset_size": len(train_loader.dataset),
            "batch_size": train_loader.batch_size
        })

        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} ({run_name})")

            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / len(train_loader.dataset)
            
            # MLflow에 에포크별 지표 로깅
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"Epoch {epoch} finished. Average Loss: {epoch_loss:.4f}")

            # 모델 저장 (최적의 모델만 저장)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                model_name = f"{run_name}_best.pth"
                model_save_path = os.path.join(output_path, model_name)
                torch.save(model.state_dict(), model_save_path)
                
                # MLflow에 모델 파일 로깅
                mlflow.log_artifact(model_save_path)
                print(f"Saved new best model to {model_save_path}")
            
        # 훈련 완료 후 최종 지표 로깅
        mlflow.log_metric("final_best_loss", best_loss)
        
        print(f"Training Complete. Run ID: {run.info.run_id}")
        return model

# --- 모델 학습 하이퍼파라미터 ---
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-4

# 데이터 로드
synth_dataset = SpineDataset(SYNTH_IMG_DIR, SYNTH_MASK_DIR)
synth_loader = DataLoader(synth_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 모델, 손실, 옵티마이저 초기화
model_A = UNet(in_channels=1, num_classes=4, base_features=16).to(DEVICE)
criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
optimizer_A = torch.optim.Adam(model_A.parameters(), lr=LR)

print(f"\n--- Training Model A (Synthetic) ---")
print(f"Dataset Size: {len(synth_dataset)}")

# 훈련 시작
model_A_trained = train_model(
    model_A, synth_loader, optimizer_A, criterion, DEVICE, EPOCHS,
    run_name="Model_A_Synthetic", output_path=OUTPUT_PATH, learning_rate=LR
)


# 데이터 로드
real_dataset = SpineDataset(REAL_IMG_DIR, REAL_MASK_DIR)
real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 모델, 손실, 옵티마이저 초기화 (새 모델 인스턴스)
model_B = UNet(in_channels=1, num_classes=4, base_features=16).to(DEVICE)
criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
optimizer_B = torch.optim.Adam(model_B.parameters(), lr=LR)

print(f"\n--- Training Model B (Real Data Baseline) ---")
print(f"Dataset Size: {len(real_dataset)}")

# 훈련 시작
model_B_trained = train_model(
    model_B, real_loader, optimizer_B, criterion, DEVICE, EPOCHS,
    run_name="Model_B_Real_Baseline", output_path=OUTPUT_PATH, learning_rate=LR
)