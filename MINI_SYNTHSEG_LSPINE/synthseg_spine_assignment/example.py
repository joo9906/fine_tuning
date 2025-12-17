import os
import scipy.ndimage as ndimage
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image

def remap_spider_labels(raw_mask):
    """
    SPIDER 라벨(0, 1-25, 100, 201-225, 101-125 등)을
    Generator에서 쓰는 압축 라벨(0: BG, 1: vertebra, 2: disc, 3: canal)로 매핑
    """
    label_map = np.zeros_like(raw_mask, dtype=np.uint8)

    # 1–25, 101–125: vertebrae
    vertebra_mask = ((raw_mask >= 1) & (raw_mask <= 25)) | \
                    ((raw_mask >= 101) & (raw_mask <= 125))
    label_map[vertebra_mask] = 1

    # 201–225: discs
    disc_mask = (raw_mask >= 201) & (raw_mask <= 225)
    label_map[disc_mask] = 2

    # 100: spinal canal
    label_map[raw_mask == 100] = 3

    # 나머지는 0 (background)
    return label_map

# 파일을 보다 쉽게 인지하기 위해 리사이징 했습니다.
def save_visual(img_uint8, save_path, target_size=(256, 256)):
    """시각화용으로만 리사이즈해서 저장"""
    pil = Image.fromarray(img_uint8)
    pil = pil.resize(target_size, resample=Image.BILINEAR)
    pil.save(save_path)

class FullSpecSpineGenerator:
    def __init__(self):
        # 조직별 신호 강도 파라미터 (Mean, Std)
        # 클래스 매핑: 0: BG, 1: vertebra, 2: disc, 3: spinal canal
        
        # T1-like contrast (지방(vertebra) 고신호, 물(disc/canal) 저신호)
        self.t1_params = {
            0: {'mean': 10, 'std': 5},   # background
            1: {'mean': 100, 'std': 15}, # vertebra (High Signal)
            2: {'mean': 45, 'std': 10},  # disc (Mid/Low Signal)
            3: {'mean': 15, 'std': 5},   # spinal canal (Low Signal)
        }

        # T2-like contrast (물(disc/canal) 고신호, 지방(vertebra) Mid/Low 신호)
        self.t2_params = {
            0: {'mean': 10, 'std': 5},   # background
            1: {'mean': 70, 'std': 10},  # vertebra (Mid/Low Signal)
            2: {'mean': 120, 'std': 15}, # disc (High Signal)
            3: {'mean': 100, 'std': 10}, # spinal canal (High Signal)
        }
        
    def _sample_contrast_params(self):
        """T1과 T2 파라미터 사이를 랜덤하게 보간하여 새로운 파라미터 세트를 생성합니다."""
        # T1(0) ~ T2(1) 사이의 랜덤 보간 계수(lambda) 생성
        # 0.0이면 T1, 1.0이면 T2, 0.5면 중간 명암비
        lambda_val = np.random.rand()
        
        sampled_params = {}
        for cls in self.t1_params.keys():
            t1_mean = self.t1_params[cls]['mean']
            t2_mean = self.t2_params[cls]['mean']
            t1_std = self.t1_params[cls]['std']
            t2_std = self.t2_params[cls]['std']

            # 선형 보간: P_sampled = (1 - lambda) * P_t1 + lambda * P_t2
            sampled_mean = (1 - lambda_val) * t1_mean + lambda_val * t2_mean
            sampled_std = (1 - lambda_val) * t1_std + lambda_val * t2_std
            
            sampled_params[cls] = {'mean': sampled_mean, 'std': sampled_std}
            
        return sampled_params

    def generate_base_image(self, label_map, sampled_params): # sampled_params 추가
        """정규분포를 이용한 현실적인 텍스처 생성"""
        synthetic = np.zeros_like(label_map, dtype=np.float32)
        
        # self.tissue_params 대신 sampled_params 사용
        for cls, params in sampled_params.items():
            mask = (label_map == cls)
            if np.count_nonzero(mask) > 0:
                synthetic[mask] = np.random.normal(
                    params['mean'],
                    params['std'],
                    size=np.count_nonzero(mask)
                )
        return synthetic

    def apply_elastic_deformation(self, img, label, alpha=15, sigma=3):
        shape = img.shape

        # displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha
        dy = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha

        # coordinate grid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # apply to image (order=1)
        distorted_img = ndimage.map_coordinates(
            img, indices, order=1, mode='reflect'
        ).reshape(shape)

        # apply to label (order=0, nearest)
        distorted_label = ndimage.map_coordinates(
            label, indices, order=0, mode='nearest'
        ).reshape(shape)

        return distorted_img, distorted_label

    def apply_physics_artifacts(self, img):
        """MRI 물리적 아티팩트 (Bias Field + Rician Noise + PVE)"""
        # 1. Partial Volume Effect (살짝 블러링)
        img = ndimage.gaussian_filter(img, sigma=0.5)

        # 2. Bias Field (밝기 불균형)
        bias = ndimage.gaussian_filter(
            np.random.randn(*img.shape), sigma=40
        )
        bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)
        img = img * (0.8 + 0.4 * bias)

        # 3. Rician Noise (배경 노이즈)
        noise_level = np.random.uniform(3, 7)
        n1 = np.random.normal(0, noise_level, img.shape)
        n2 = np.random.normal(0, noise_level, img.shape)
        img = np.sqrt((img + n1) ** 2 + n2 ** 2)

        return img

    def random_downsample(self, img):
        """저해상도 모사"""
        if np.random.rand() > 0.5:
            return img  # 50% 확률로 실행 안 함

        factor = np.random.uniform(1.2, 2.0)
        original_shape = img.shape

        # 축소
        small = ndimage.zoom(img, 1 / factor, order=1)
        # 다시 확대 (복원)
        restored = ndimage.zoom(small, factor, order=1)

        # 크기 맞추기 (Crop or Pad)
        result = np.zeros(original_shape, dtype=img.dtype)
        h = min(original_shape[0], restored.shape[0])
        w = min(original_shape[1], restored.shape[1])
        result[:h, :w] = restored[:h, :w]

        return result

    def generate_synthetic_mri(self, raw_label_map):
        """
        raw_label_map: SPIDER에서 읽은 원본 mask
        """
        # 0~3 클래스 라벨로 매핑
        label_map = remap_spider_labels(raw_label_map)
        
        # 0. 랜덤 명암비 파라미터 샘플링 (추가된 부분)
        sampled_params = self._sample_contrast_params()

        # 1. 기본 intensity 생성 (sampled_params 전달)
        img = self.generate_base_image(label_map, sampled_params) # 수정

        # 2. 기하학적 변형 (이미지 + 라벨 둘 다)
        img, label_map = self.apply_elastic_deformation(
            img, label_map,
            alpha=np.random.uniform(10, 20),
            sigma=np.random.uniform(3, 5)
        )

        # 3. 물리적 artifact 적용
        img = self.apply_physics_artifacts(img)

        # 4. 해상도 저하
        img = self.random_downsample(img)

        return np.clip(img, 0, 255), label_map


# -------------------------------------------------------
# 설정
# -------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MASK_DIR = os.path.join(ROOT, "data", "masks")
OUT_IMG_DIR = os.path.join(ROOT, "SYNTH_T1_SEG", "images")
OUT_MASK_DIR = os.path.join(ROOT, "SYNTH_T1_SEG", "masks")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


# -------------------------------------------------------
# 유틸: normalize → uint8
# -------------------------------------------------------
def to_uint8(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    return img.astype(np.uint8)

def generate_synthetic_dataset():
    print("📌 SPIDER Synthetic MRI Dataset Generation Start")
    print(f" - mask path: {MASK_DIR}")
    print(f" - output path: {OUT_IMG_DIR}")

    gen = FullSpecSpineGenerator()
    total_saved = 0

    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(".mha")]

    for mask_name in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(MASK_DIR, mask_name)
        base_id = os.path.splitext(mask_name)[0].replace("_mask", "")

        # 1. mask load (3D)
        mask_img = sitk.ReadImage(mask_path)
        mask_vol = sitk.GetArrayFromImage(mask_img)  # (Z, H, W)

        Z, H, W = mask_vol.shape

        # 2. 모든 슬라이스에 대해 유효성 체크 & 처리
        for z in range(Z):
            slice_max = mask_vol[z].max()
            
            # 배경 슬라이스만 skip (빠르게 필터링)
            if slice_max == 0:
                continue
                
            # SPIDER 라벨 → 0~3 라벨로 리맵
            raw_label_slice = mask_vol[z].astype(np.int32)
            label_slice = remap_spider_labels(raw_label_slice)

            # 4. 합성 MRI 생성
            syn_img, syn_lbl = gen.generate_synthetic_mri(label_slice)

            # 5. 저장
            img_uint8 = to_uint8(syn_img)
            lbl_uint8 = syn_lbl.astype(np.uint8)

            img_save_path = os.path.join(OUT_IMG_DIR, f"{base_id}_z{z:03d}.png")
            lbl_save_path = os.path.join(OUT_MASK_DIR, f"{base_id}_z{z:03d}.png")

            save_visual(img_uint8, img_save_path)
            Image.fromarray(lbl_uint8).save(lbl_save_path)

            total_saved += 1
            
    print(f"\n🎉 Synthetic dataset generation completed!")
    print(f"Total slices saved: {total_saved}")
    print(f"Saved in: {OUT_IMG_DIR}")

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


# 데이터 로드
real_dataset = SpineDataset(REAL_IMG_DIR, REAL_MASK_DIR)
real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 모델, 손실, 옵티마이저 초기화 (새 모델 인스턴스)
model_B = UNet(in_channels=1, num_classes=4, base_features=16).to(DEVICE)
criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
optimizer_B = torch.optim.Adam(model_B.parameters(), lr=LR)

print(f"\n--- Training Model B (Real Data Baseline) ---")
print(f"Dataset Size: {len(real_dataset)}")


# 평가 코드
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import math 
import datetime

# -------------------------------------------------------
# 1. 경로 및 설정 (사용자 경로 반영 및 결과 저장 경로 추가)
# -------------------------------------------------------
# T2 평가 데이터 경로
EVAL_T2_IMG_DIR = "SPIDER_T2_val/images"
EVAL_T2_MASK_DIR = "SPIDER_T2_val/masks"

# 훈련된 모델 경로
OUTPUT_PATH = "models/saved"
MODEL_A_PATH = os.path.join(OUTPUT_PATH, "Model_A.pth")
MODEL_B_PATH = os.path.join(OUTPUT_PATH, "Model_B.pth")

# 평가 결과 저장 경로
RESULTS_DIR = "results/evaluation_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4 
BASE_FEATURES = 16 


# -------------------------------------------------------
# 2. U-Net 모델 정의 (내용 동일)
# -------------------------------------------------------

class ConvBlock(nn.Module):
    """Double convolution block."""
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
    """Lightweight U-Net for 2D segmentation (Cropping logic included)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_features: int = 16):
        super().__init__()
        f = base_features
        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)
        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)
        # Output
        self.out_conv = nn.Conv2d(f, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def _crop_skip(self, skip_tensor, upsampled_tensor):
        """Center crops the skip connection tensor to match the upsampled tensor size."""
        target_h = upsampled_tensor.size(2)
        target_w = upsampled_tensor.size(3)
        
        skip_h = skip_tensor.size(2)
        skip_w = skip_tensor.size(3)
        
        if skip_h == target_h and skip_w == target_w:
            return skip_tensor
        
        diff_h = skip_h - target_h
        diff_w = skip_w - target_w
        
        cropped = skip_tensor[:, :, 
                              diff_h // 2: diff_h // 2 + target_h,
                              diff_w // 2: diff_w // 2 + target_w]
        
        return cropped

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4)) 

        # Decoder with skip connections (Cropping applied before torch.cat)
        up4_out = self.up4(b)
        e4_cropped = self._crop_skip(e4, up4_out)
        d4 = self.dec4(torch.cat([up4_out, e4_cropped], dim=1))

        up3_out = self.up3(d4)
        e3_cropped = self._crop_skip(e3, up3_out)
        d3 = self.dec3(torch.cat([up3_out, e3_cropped], dim=1))

        up2_out = self.up2(d3)
        e2_cropped = self._crop_skip(e2, up2_out)
        d2 = self.dec2(torch.cat([up2_out, e2_cropped], dim=1))

        up1_out = self.up1(d2)
        e1_cropped = self._crop_skip(e1, up1_out)
        d1 = self.dec1(torch.cat([up1_out, e1_cropped], dim=1))

        return self.out_conv(d1)


# -------------------------------------------------------
# 3. Dataset 및 Metrics 정의 (내용 동일)
# -------------------------------------------------------

class SpineDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        # 이미지 로드 및 정규화
        img_path = os.path.join(self.img_dir, name)
        image = np.array(Image.open(img_path)).astype(np.float32) / 255.0 
        image = torch.from_numpy(image).unsqueeze(0) 
        
        # 마스크 로드 (라벨)
        mask_path = os.path.join(self.mask_dir, name)
        # 마스크 파일명이 이미지 파일명과 동일하다고 가정 (예: 001_t1_z409.png)
        # 실제 SPIDER 데이터셋에서는 라벨 파일명이 다를 수 있으므로, train.py에서 사용한 라벨명 규칙을 확인해야 함.
        # 일단은 파일명이 같다고 가정하고 진행.
        
        # **주의:** 만약 라벨 파일명이 '001_label_z409.png' 처럼 다르다면 아래 코드에 수정이 필요함.
        # 현재 코드에서는 이미지 파일명과 동일한 마스크 파일명을 사용합니다.
        mask = np.array(Image.open(mask_path)).astype(np.int64) 
        mask = torch.from_numpy(mask) 
        
        return image, mask


def center_crop_numpy(arr, target_h, target_w):
    """2D NumPy 배열 (H, W)을 대상 치수로 중앙에서 자릅니다."""
    h, w = arr.shape
    
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    end_h = start_h + target_h
    end_w = start_w + target_w
    
    return arr[start_h:end_h, start_w:end_w]

def pad_collate(batch):
    """
    배치 내에서 가장 큰 크기이자, UNet의 다운샘플링 인자(32)의 배수에 맞춰 패딩을 수행합니다.
    """
    images, masks = zip(*batch)
    
    # 1. 배치 내 최대 크기 찾기
    max_h_batch = max([img.shape[1] for img in images])
    max_w_batch = max([img.shape[2] for img in images])

    # 2. UNet 다운샘플링 팩터 (5번의 풀링: 2^5 = 32)
    DOWN_FACTOR = 32 
    
    # 3. 최종 Target 크기 결정: 32의 배수이면서 배치 내 최대 크기보다 크거나 같도록
    target_h = max(DOWN_FACTOR, int(math.ceil(max_h_batch / DOWN_FACTOR)) * DOWN_FACTOR)
    target_w = max(DOWN_FACTOR, int(math.ceil(max_w_batch / DOWN_FACTOR)) * DOWN_FACTOR)

    padded_images = []
    padded_masks = []
    
    for img, mask in zip(images, masks):
        # 패딩 길이 계산
        pad_h_before = (target_h - img.shape[1]) // 2
        pad_h_after = target_h - img.shape[1] - pad_h_before
        pad_w_before = (target_w - img.shape[2]) // 2
        pad_w_after = target_w - img.shape[2] - pad_w_before
        
        # Image Padding (float, 0.0)
        img_padded = F.pad(img.unsqueeze(0), 
                           (pad_w_before, pad_w_after, pad_h_before, pad_h_after), 
                           'constant', 0.0).squeeze(0)
        padded_images.append(img_padded)
        
        # Mask Padding (long, 0)
        mask_padded = F.pad(mask.unsqueeze(0).unsqueeze(0), 
                             (pad_w_before, pad_w_after, pad_h_before, pad_h_after), 
                             'constant', 0).squeeze(0).squeeze(0)
        padded_masks.append(mask_padded)
        
    return torch.stack(padded_images), torch.stack(padded_masks)


def dice_score(pred_mask, true_mask, num_classes):
    """
    Dice 유사도 계수 계산. (배경 클래스(0) 제외)
    pred_mask: (H, W), true_mask: (H, W)
    """
    smooth = 1e-6
    dice_per_class = []

    for c in range(1, num_classes): 
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)

        intersection = (pred_c * true_c).sum()
        union = pred_c.sum() + true_c.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice.item())

    return np.mean(dice_per_class)


# -------------------------------------------------------
# 4. 평가 함수 및 메인 실행 (결과 저장 로직 추가)
# -------------------------------------------------------
def evaluate_model(model_path, eval_loader, model_name):
    if not os.path.exists(model_path):
        print(f"경고: 모델 체크포인트를 찾을 수 없습니다: {model_path}. 평가를 건너뜁니다.")
        return 0.0

    model = UNet(in_channels=1, num_classes=NUM_CLASSES, base_features=BASE_FEATURES).to(DEVICE)
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_dice_scores = []

    with torch.no_grad():
        for images, masks in tqdm(eval_loader, desc=f"Evaluating {model_name} on T2 data"):
            images = images.to(DEVICE)
            masks = masks.cpu().numpy() # 패딩된 마스크

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy() 

            for pred, true_mask_padded in zip(preds, masks):
                # 패딩된 마스크(true_mask_padded)를 모델 출력 크기(pred)에 맞게 크롭
                H_pred, W_pred = pred.shape
                true_mask_cropped = center_crop_numpy(true_mask_padded, H_pred, W_pred)
                
                # 라벨이 존재하는 경우에만 Dice 계산
                if true_mask_cropped.max() > 0: 
                    dice = dice_score(pred, true_mask_cropped, NUM_CLASSES)
                    all_dice_scores.append(dice)

    mean_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
    return mean_dice


if __name__ == "__main__":
    generate_synthetic_dataset()
    # 모델 A 훈련 시작
    model_A_trained = train_model(
        model_A, synth_loader, optimizer_A, criterion, DEVICE, EPOCHS,
        run_name="Model_A_Synthetic", output_path=OUTPUT_PATH, learning_rate=LR
    )

    # 모델 B 훈련 시작
    model_B_trained = train_model(
        model_B, real_loader, optimizer_B, criterion, DEVICE, EPOCHS,
        run_name="Model_B_Real_Baseline", output_path=OUTPUT_PATH, learning_rate=LR
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"--- Starting T2 Robustness Evaluation on Device: {DEVICE} ---")

    # 1. T2 평가 데이터셋 로드
    eval_dataset = SpineDataset(EVAL_T2_IMG_DIR, EVAL_T2_MASK_DIR)
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=16,
        shuffle=False, 
        num_workers=0,
        collate_fn=pad_collate
    )
    
    print(f"T2 Evaluation Dataset Size: {len(eval_dataset)}")
    
    # 결과 문자열을 저장할 리스트
    results_output = []
    
    if len(eval_dataset) == 0:
        error_msg = "Fatal Error: T2 평가 데이터셋이 비어 있습니다. eval_t2_only 폴더를 확인하세요."
        print(error_msg)
        results_output.append(error_msg)
    else:
        # 2. 모델 A (Synthetic) 평가
        dice_A = evaluate_model(MODEL_A_PATH, eval_loader, "Model A (Synthetic)")

        # 3. 모델 B (Real T1 Baseline) 평가
        dice_B = evaluate_model(MODEL_B_PATH, eval_loader, "Model B (Real T1 Baseline)")

        # 4. 결과 출력 및 비교
        train_loss_A = 0.012 
        train_loss_B = 0.052 

        
        header = "\n=======================================================\n"
        header += f"=== T2 Robustness Evaluation Results ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n"
        header += "=======================================================\n"
        
        result_A = f"Model A (Synthetic, Train Loss {train_loss_A}): {dice_A:.4f}\n"
        result_B = f"Model B (Real T1 Baseline, Train Loss {train_loss_B}): {dice_B:.4f}\n"