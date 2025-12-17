import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image

# -------------------------------------------------------
# 설정: 경로 변수 (train.ipynb 경로 설정을 따름)
# -------------------------------------------------------
# T1/T2 파일이 모두 포함된 원본 MHA 데이터 폴더
INPUT_MHA_DIR = r"C:\Users\jooyoung\Desktop\Code\03_fine_tuning\MINI_SYNTHSEG_LSPINE\data"

# T1 이미지/마스크 파일이 있는 하위 폴더 이름 (예: data/images, data/masks)
# 사용자님의 train.ipynb를 참고하여 MHA 파일의 실제 위치에 맞게 수정해주세요.
INPUT_T1_IMG_MHA_DIR = os.path.join(INPUT_MHA_DIR, "images") 
INPUT_T1_MASK_MHA_DIR = os.path.join(INPUT_MHA_DIR, "masks") 

# 출력 경로: Model B 훈련용 Pure T1 PNG 슬라이스가 저장될 폴더
OUTPUT_T1_IMG_DIR = r"C:\Users\jooyoung\Desktop\Code\03_fine_tuning\MINI_SYNTHSEG_LSPINE\real_data\images" 
OUTPUT_T1_MASK_DIR = r"C:\Users\jooyoung\Desktop\Code\03_fine_tuning\MINI_SYNTHSEG_LSPINE\real_data\masks"

os.makedirs(OUTPUT_T1_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_T1_MASK_DIR, exist_ok=True)


# -------------------------------------------------------
# 유틸리티 함수 (기존 코드와 동일)
# -------------------------------------------------------
def remap_spider_labels(raw_mask):
    """
    SPIDER 라벨을 압축 라벨(0: BG, 1: vertebra, 2: disc, 3: canal)로 매핑
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

    return label_map

def to_uint8(img_array):
    """0~255 범위로 Min-Max 정규화"""
    img_array = img_array.astype(np.float32)
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max > img_min:
        img_array = (img_array - img_min) / (img_max - img_min + 1e-8) * 255
    else:
        img_array = np.zeros_like(img_array)
        
    return img_array.astype(np.uint8)


# -------------------------------------------------------
# 메인 변환 로직 (T1 필터링 추가)
# -------------------------------------------------------
def convert_t1_mha_to_png_slices(img_mha_dir, mask_mha_dir, img_out_dir, mask_out_dir):
    """3D T1 MHA 파일에서 2D 슬라이스를 추출하여 PNG로 저장"""
    
    # 1. 파일 목록 가져오기 (T1 파일만 필터링)
    # T1 파일 이름 규칙: 'X_t1.mha' 형태를 가정합니다.
    mha_files = sorted([f for f in os.listdir(mask_mha_dir) if f.endswith('.mha') and '_t1' in f.lower()])
    
    if len(mha_files) == 0:
        print("🚨 Error: No T1 MHA files found in the mask directory. Check the path and file naming.")
        return

    for mha_name in tqdm(mha_files, desc="Slicing Pure T1 MHA Files"):
        # 파일 이름을 ID로 사용 (예: 1_t1.mha -> 1_t1)
        base_id = os.path.splitext(mha_name)[0]
        
        # 이미지 파일 이름은 마스크 파일 이름과 ID가 일치한다고 가정
        # 주의: SPIDER 데이터는 이미지와 마스크 파일명이 다를 수 있습니다.
        # 이 스크립트는 이미지와 마스크의 기본 ID가 동일하다고 가정합니다.
        img_name = base_id.replace("_mask", "") + ".mha" 
        
        # 2. Image Load (3D)
        try:
            img_vol = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_mha_dir, img_name))) # (Z, H, W)
            mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_mha_dir, mha_name))) # (Z, H, W)
        except RuntimeError:
            print(f"Skipping {base_id}: Matching image file not found or read error.")
            continue
        
        Z = img_vol.shape[0]

        # 3. 모든 슬라이스 순회 (배경 아닌 슬라이스만 추출)
        for z in range(Z):
            raw_mask_slice = mask_vol[z] 
            
            # 배경 슬라이스는 스킵
            if raw_mask_slice.max() == 0:
                continue
                
            # 마스크 슬라이스 리매핑
            mask_slice_remapped = remap_spider_labels(raw_mask_slice)
            img_slice = img_vol[z]

            # 이미지 정규화 및 uint8 변환
            img_uint8 = to_uint8(img_slice)
            mask_uint8 = mask_slice_remapped.astype(np.uint8) 

            # 4. 저장
            filename = f"{base_id}_z{z:03d}.png"
            
            Image.fromarray(img_uint8).save(os.path.join(img_out_dir, filename))
            Image.fromarray(mask_uint8).save(os.path.join(mask_out_dir, filename))

    print(f"\n✅ Pure T1 Conversion Complete. PNG slices saved to:\n - {OUTPUT_T1_IMG_DIR}\n - {OUTPUT_T1_MASK_DIR}")

# 변환 함수 실행
convert_t1_mha_to_png_slices(INPUT_T1_IMG_MHA_DIR, INPUT_T1_MASK_MHA_DIR, OUTPUT_T1_IMG_DIR, OUTPUT_T1_MASK_DIR)