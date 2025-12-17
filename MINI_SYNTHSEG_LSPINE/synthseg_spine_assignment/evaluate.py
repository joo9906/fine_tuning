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
EVAL_T2_IMG_DIR = r"C:\Users\jooyoung\Desktop\Code\03_fine_tuning\MINI_SYNTHSEG_LSPINE\submit\SPIDER_T2_val\images" 
EVAL_T2_MASK_DIR = r"C:\Users\jooyoung\Desktop\Code\03_fine_tuning\MINI_SYNTHSEG_LSPINE\submit\SPIDER_T2_val\masks"

# 훈련된 모델 경로
OUTPUT_PATH = r"C:\Users\jooyoung\Desktop\Code\03_fine_tuning\MINI_SYNTHSEG_LSPINE\submit\models\saved"
MODEL_A_PATH = os.path.join(OUTPUT_PATH, "Model_A.pth")
MODEL_B_PATH = os.path.join(OUTPUT_PATH, "Model_B.pth")

# 평가 결과 저장 경로
RESULTS_DIR = "results/evaluation_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "robustness_results.txt")

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
    