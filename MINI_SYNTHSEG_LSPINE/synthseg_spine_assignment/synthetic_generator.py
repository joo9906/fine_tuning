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



if __name__ == "__main__":
    generate_synthetic_dataset()
