import numpy as np
import scipy.ndimage as ndimage

class FullSpecSpineGenerator:
    def __init__(self):
        # 조직별 신호 강도 (Mean, Std)
        self.tissue_params = {
            0: {'mean': 10, 'std': 5},     # background
            1: {'mean': 100, 'std': 15},   # vertebra
            2: {'mean': 45, 'std': 10},    # disc
            3: {'mean': 15, 'std': 5},     # spinal canal
        }

    def generate_base_image(self, label_map):
        """A: 정규분포를 이용한 현실적인 텍스처 생성"""
        synthetic = np.zeros_like(label_map, dtype=np.float32)
        for cls, params in self.tissue_params.items():
            mask = (label_map == cls)
            if np.count_nonzero(mask) > 0:
                synthetic[mask] = np.random.normal(params['mean'], params['std'], size=np.count_nonzero(mask))
        return synthetic

    def apply_elastic_deformation(self, img, alpha=15, sigma=3):
        """B: 탄성 변형 (Elastic Deformation) - 척추가 휘거나 눌리는 효과"""
        shape = img.shape
        
        # 1. 랜덤한 변위장(Displacement field) 생성
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        # 2. 그리드 생성
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # 3. 이미지 왜곡 적용
        # map_coordinates는 1차원으로 펼쳐서 연산하므로 다시 shape 복구 필요
        distorted_img = ndimage.map_coordinates(img, indices, order=1, mode='reflect')
        return distorted_img.reshape(shape)

    def apply_physics_artifacts(self, img):
        """A: MRI 물리적 아티팩트 (Bias Field + Rician Noise + PVE)"""
        # 1. Partial Volume Effect (살짝 블러링)
        img = ndimage.gaussian_filter(img, sigma=0.5)

        # 2. Bias Field (밝기 불균형)
        bias = ndimage.gaussian_filter(np.random.randn(*img.shape), sigma=40)
        bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)
        img = img * (0.8 + 0.4 * bias)

        # 3. Rician Noise (배경 노이즈)
        noise_level = np.random.uniform(3, 7)
        n1 = np.random.normal(0, noise_level, img.shape)
        n2 = np.random.normal(0, noise_level, img.shape)
        img = np.sqrt((img + n1)**2 + n2**2)
        
        return img

    def random_downsample(self, img):
        """저해상도 모사"""
        if np.random.rand() > 0.5: return img # 50% 확률로 실행 안 함
        
        factor = np.random.uniform(1.2, 2.0)
        original_shape = img.shape
        
        # 축소
        small = ndimage.zoom(img, 1/factor, order=1)
        # 다시 확대 (복원)
        restored = ndimage.zoom(small, factor, order=1)
        
        # 크기 맞추기 (Crop or Pad)
        result = np.zeros(original_shape)
        h, w = min(original_shape[0], restored.shape[0]), min(original_shape[1], restored.shape[1])
        result[:h, :w] = restored[:h, :w]
        
        return result

    def generate(self, label_map):
        # 1. 기본 텍스처 입히기
        img = self.generate_base_image(label_map)
        
        # 2. 기하학적 변형 (B 반영: 척추 모양 휘기)
        # 주의: Label Map도 같이 변형해야 학습 데이터로 쓸 수 있음.
        # 여기서는 이미지 합성 로직만 보여드리지만, 실제로는 label_map도 같은 seed로 변형해야 함.
        img = self.apply_elastic_deformation(img, alpha=np.random.uniform(10, 20), sigma=np.random.uniform(3, 5))
        
        # 3. 물리적 효과 (A 반영)
        img = self.apply_physics_artifacts(img)
        
        # 4. 해상도 저하
        img = self.random_downsample(img)
        
        return np.clip(img, 0, 255)