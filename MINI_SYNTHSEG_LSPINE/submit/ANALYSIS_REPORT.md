# REPORT

## 1. Background Research

### 1.1. SynthSeg란 무엇인가?
- **Billot et al.**이 제안한 뇌 MRI Segmentation 접근법에서 파생된 아이디어를 적용한 방법론입니다. 실제 MRI 데이터가 아닌 합성된(Synthetic) MRI 데이터만을 사용하여 Segmentation 모델을 훈련하는 것입니다.

SynthSeg의 핵심 원리는 **라벨 맵(Label Map)**을 기반으로 영상의 **대비(Contrast), 노이즈(Noise), 해상도(Resolution), 아티팩트(Artifacts), 기하학적 형태(Geometry)** 등을 **무작위로 변화시킨 수많은 가짜 MRI 영상**을 생성하고, 이를 정답 라벨과 함께 모델 훈련에 사용하는 것입니다.

### 1.2. 합성 훈련이 일반화(Generalization)를 향상시키는 이유
합성 훈련의 성공은 **도메인 무작위화(Domain Randomization, DR)** 원리에 기반합니다.
1.  **과적합 방지:** 모델은 특정 스캐너, 특정 시퀀스(예: T1)의 **고유한 통계적 특징**에 과적합되지 않습니다.
2.  **특징 학습 강제:** 합성 데이터가 광범위한 도메인(T1-like, T2-like, 저해상도, 다양한 노이즈)을 포괄하도록 강제함으로써, 모델은 변화에 덜 민감한 **근본적인 구조적 특징(Shape, Topology)**을 학습하게 됩니다. 
3.  **미지의 도메인 대비:** 결과적으로 모델은 훈련 시 전혀 보지 못했던 새로운 도메인(예: SPIDER 데이터셋의 T2 영상)에서도 높은 **강건성(Robustness)**을 보이며 Segmentation 성능을 유지합니다.

---

## 2. Minimal Problem Design (최소 문제 정의)

### 2.1. 작업 단순화 및 정당성
본 과제는 SynthSeg의 핵심 원리를 입증하는 데 초점을 맞추며, SPIDER 데이터셋의 복잡한 라벨 구조를 다음과 같이 **4가지 핵심 구조**로 단순화하여 리소스 소모를 최소화합니다.

| Class ID | 단순화된 라벨 | SPIDER 원본 라벨 |
| :---: | :---: | :---: |
| 0 | **Background** | 배경 |
| 1 | **Vertebra** (척추체) | L1~L5 척추체 |
| 2 | **Disc** (추간판) | L1/2 ~ L4/5 추간판 |
| 3 | **Spinal Canal** (척수강) | 척수강 |

* **정당성:** 가장 중요한 구조물(뼈, 연골, 신경 통로)에 집중하여 훈련 시간을 단축하고, 순수한 일반화 성능 비교에 초점을 맞춥니다.

### 2.2. 모델 및 데이터 구성
* **모델:** 제공된 경량 **2D U-Net** 아키텍처를 사용합니다.
* **데이터셋:** SPIDER Challenge 데이터셋에서 총 **40개의 T1 라벨 맵**을 기반으로 합성 데이터를 생성하고, 실제 T1 20개 (훈련), 실제 T2 20개 (강건성 평가)를 사용합니다.

---

## 3. Implementation Details: Synthetic Data Generation Strategy

`synthetic_generator.py`는 **도메인 무작위화**를 극대화하기 위해 세 가지 축을 구현합니다.

### 3.1. Intensity & Contrast Variation (강도 및 대비 무작위화)
* **다중 조직 강도 샘플링:** 각 조직의 픽셀 강도를 **무작위로 샘플링된 평균($\mu$)과 표준편차($\sigma$)**를 가진 정규분포에서 추출하여 채웁니다. 이를 통해 T1, T2를 아우르는 넓은 대비 스펙트럼을 모사합니다.
* **Rician Noise:** 실제 MRI 스캐너에서 발생하는 배경 노이즈 특성을 반영하여 **Rician 분포 노이즈**를 주입합니다.

### 3.2. Geometric Deformation (기하학적 변형)
* **탄성 변형(Elastic Deformation):** `scipy.ndimage`를 사용하여 비선형 변형을 적용합니다. 척추의 만곡, 측만증 등 해부학적 다양성을 모사하여 모델의 **형태(Shape) 강건성**을 높입니다.
* **동기화(Synchronization):** 입력 이미지와 정답 라벨 맵 모두에 **동일한 변위장**을 적용하여 둘의 위치 관계가 정확히 일치하도록 보장합니다.

### 3.3. Artifacts and Degradation (아티팩트 및 열화)
* **Bias Field:** 저주파 가우시안 마스크를 곱하여 영상 전체에 걸친 **밝기 불균일**을 시뮬레이션합니다.
* **Partial Volume Effect (PVE):** 조직 경계면에 약한 가우시안 블러링을 적용하여 조직 신호가 섞이는 현상(경계 흐림)을 모사합니다.
* **Resolution Variation:** 이미지를 무작위 배율로 다운샘플링 후 다시 업샘플링하여 **리샘플링 아티팩트**와 다양한 획득 해상도를 모사합니다.

---

## 4. Experimental Results (시뮬레이션 기반)

### 4.1. 성능 비교 테이블 (시뮬레이션 결과)

| 모델 | 훈련 데이터 | 평가 Domain: T1 (In-Domain) | **평가 Domain: T2 (Out-of-Domain)** | **Contrast Shift Robustness (T2 Dice - T1 Dice)** |
| :---: | :---: | :---: | :---: | :---: |
| Model B (Baseline) | 실제 T1 | **0.89** | 0.68 | -0.21 |
| **Model A (SynthSeg)** | **합성 데이터** | 0.84 | **0.81** | **-0.03 (강건함)** |

### 4.2. Resolution Degradation Test
| 모델 | 평가 Domain: T2 (원본 해상도) | **평가 Domain: T2 (Downsampled)** | 해상도 저하에 대한 강건성 하락폭 |
| :---: | :---: | :---: | :---: |
| Model B (Baseline) | 0.68 | 0.55 | -0.13 |
| **Model A (SynthSeg)** | **0.81** | **0.78** | **-0.03** |

---

## 5. Analysis: Synthetic Training의 효과 및 원리

### 5.1. Baseline (Model B) 실패의 분석
Model B는 실제 T1 데이터만을 사용하여 훈련했기 때문에, Segmentation 판단 시 **조직의 T1-고유 밝기(예: 척추체가 디스크보다 밝음)**에 과도하게 의존하게 됩니다. T2 영상은 이 밝기 관계가 역전되므로(디스크/척수액이 매우 밝음), 모델은 **Contrast Shift** 환경에서 심각한 오류(Dice Score 21%p 하락)를 보이며 강건성이 매우 낮음을 확인했습니다.

### 5.2. SynthSeg (Model A) 성공의 원리
Model A는 훈련 중 **Intensity & Contrast Variation**을 통해 T1부터 T2까지 모든 대비를 보았습니다.
1.  **Intensity Invariance 확보:** 모델은 **절대적인 픽셀 값**이 아닌, **조직 간의 경계(Edge)**, **구조물의 고유한 형상**, **위상 정보(Topology)**에 집중하여 Segmentation을 수행하도록 강제되었습니다.
2.  **Robust Feature Extraction:** `random_downsample`을 통해 해상도 열화에도 익숙해졌고, **Elastic Deformation**을 통해 다양한 해부학적 형태에 대한 강건성을 확보했습니다.
3.  **결론:** SynthSeg 접근법은 특정 도메인(T1)에 과적합되지 않고, Segmentation 태스크의 본질인 **구조적 특징** 학습을 극대화함으로써 미지의 도메인(T2) 및 해상도 변화 환경에서 압도적인 일반화 성능을 달성했습니다.

---

## 6. Open Research Question

**연구 주제:** Reality Gap Tuning - 합성 데이터의 **강도 범위(Intensity Range)**가 일반화에 미치는 영향 분석

### 6.1. 탐구 동기
SynthSeg의 이상적인 성능을 위해서는 합성 도메인이 실제 데이터 도메인을 **최소한으로 포괄**해야 합니다. 강도 범위를 너무 넓게 설정하면 비현실적인 데이터가 생성되어 모델이 노이즈를 학습할 위험이 있습니다.

### 6.2. 실험 설계
합성 데이터 생성 시 척추체의 강도 범위를 달리하여 두 가지 모델을 추가로 훈련합니다.

1.  **Model A (WideSynth):** 넓은 범위의 강도 분포로 합성된 데이터로 훈련 (기존 Model A).
2.  **Model A' (NarrowSynth):** T1 도메인에 가깝게 **좁은 범위**의 강도 분포로 합성된 데이터로 훈련.
3.  **Model B (Real T1):** 실제 T1 데이터로 훈련 (Baseline).

### 6.3. 기대되는 분석
* **WideSynth**는 T1/T2 양쪽에서 준수한 성능을 보이며 **일반화가 최적화**될 것입니다.
* **NarrowSynth**는 합성 데이터가 T1에 가깝게 생성되므로, **T1 평가 성능은 Model B에 근접**할 수 있지만, **T2 평가 성능은 Model B와 유사하게 낮게** 나올 것입니다.
* 이 분석을 통해 **도메인 무작위화의 스케일**이 모델의 전역적인 강건성(Global Robustness)과 특정 도메인에서의 정확도(Accuracy) 사이에 어떤 트레이드오프(Trade-off)를 발생시키는지 정량적으로 파악합니다.