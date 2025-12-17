## Setup
운영체제 : Windows 11
개발 언어 : Python 3.10.10
주요 라이브러리 : Pytorch, Numpy, Pillow, Matplotlib, Pandas
하드웨어 : GTX 2070 Super 8gb

## Usage (사용 및 실행 방법)
제출된 코드를 활용하여 Synthetic 데이터 생성, 모델 학습, 그리고 평가를 수행하는 단계별 절차입니다.

1. 환경 준비 및 데이터셋 준비
의존성 설치: requirements.txt에 명시된 모든 라이브러리를 설치합니다.

`pip install -r requirements.txt`

데이터셋 다운로드: 오픈 소스인 SPIDER lumbar spine MRI 데이터셋의 일부를 다운로드하고, mha 파일을 png로 변환하여 아래 지정된 경로에 배치합니다.
**변환 코드 : extraction_files의 `eval_extraction.py`와 `train_extraction.py` 파일을 사용하여 평가와 훈련 데이터셋을 추출할 수 있습니다.**

data/SPIDER_T1_train/ (실제 T1 MRI 및 라벨 맵)
data/SPIDER_T2_val/ (실제 T2 MRI 및 라벨 맵 - 평가용)

현재는 제가 추출하여 사용한 PNG 이미지들이 들어가 있는 상태라 바로 학습과 평가가 가능합니다.

파일 용량 문제로 SPIDER_T2_val과 SYNTH_T1_SEG 폴더의 .png 이미지들은 총 20명 분, SPIDER_T1_train은 40명분의 이미지가 들어있습니다.

2. Synthetic 데이터 생성
`synthetic_generator.py` 스크립트를 실행하여 SPIDER의 라벨 맵으로부터 Synthetic MRI 이미지를 생성합니다.

생성된 Synthetic 데이터는 SYNTH_T1_SEG/ 경로에 저장됩니다.

`python synthetic_generator.py`

3. 모델 학습
train.py 파일을 실행하여 두 가지 모델을 동시에 학습합니다.

주의: 스크립트 내에서 데이터 및 체크포인트 경로가 일치하는지 확인해야 합니다.

`python train.py`
Model A (Robust Model): Synthetic 데이터로만 학습 => 저장 경로: models/saved/Model_A_SynthOnly.pth

Model B (Baseline Model): Real T1 데이터로만 학습 => 저장 경로: models/saved/Model_B_RealT1Only.pth

4. 모델 평가 및 결과 확인
evaluate.py 스크립트를 실행하여 두 모델의 **강건성(Robustness)**을 비교 평가합니다.

스크립트 내에서 모델 경로가 models/saved/에 저장된 체크포인트 파일과 일치하는지 확인 후 실행합니다.

`python evaluate.py`

전체 파이프라인 시연: example.py를 실행하여 데이터 생성부터 평가까지의 과정을 확인할 수 있습니다.

## AI usage
1. Gemini 3
- 이론적 배경 탐색 : Synthetic MRI 및 SynthSeg와 관련된 레퍼런스 및 논문을 탐색하고 핵심 원리(Domain Randomization)를 파악하여 ANALYSIS_REPORT.md 작성을 위한 기반 마련
- 데이터셋 처리 : SPIDER 데이터셋에서 2D U-Net 학습에 필요한 Axial 단면 데이터셋을 추출하는 Python 코드를 생성
- Synthetic Generator : Synthetic MRI 생성 시 요구되는 데이터 증강 기법 (Intensity Sampling, Bias Field, Resolution Variation 등)을 넣고, 코드를 받아 구현.
- 모델 학습 디버깅,PyTorch 학습 시 발생하는 CUDA/Memory/Shape 관련 오류들을 분석하고 수정하여 학습 스크립트의 안정성을 확보.

2. GPT 5.1
- Gemini로 생성한 코드 리뷰 및 이론 검증