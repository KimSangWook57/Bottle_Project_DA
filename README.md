# Bottle_Project_DA
- 병 판별 프로젝트를 수행하며 배운 데이터 분석 관련 모델과 코드를 모아 놓았습니다. 아직 난잡한 코드가 많습니다.
- 기획서 docs : https://docs.google.com/document/d/1MNx_ZBxme7EzghV2o5YCM3HD9PrhmfMrc4ZL8C719ng/edit?usp=sharing
- 분석 보고서 : https://github.com/KimSangWook57/Bottle_Project_DA/blob/main/notebooks/report.ipynb
## 실행방법
### 가상환경 및 모델학습에 사용한 버전
- Windows 10
- Python 3.10.10
- cuda 11.2.0
- cuDNN v8.1.0 

## pip 가상환경 설치
```
py -3.10 -m venv venv
.\venv\Scripts\activate
```

### 설치 예시(cuda ver.)
Windows
```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Linux
```
pip3 install torch torchvision torchaudio
```

### 폴더 설명
- notebooks = jupyter notebook으로 모델 생성 코드를 돌린 결과들을 모아 놓았음
  - report.ipynb = 병 판별 데이터셋 및 모델 분석 보고서
  - model.ipynb = 모델 생성 풀코드
  - Image_Preprocessing.ipynb = 이미지 전처리 관련 코드

### 코드 설명
- app.py = yolov5 모델 실행(이미지를 넣으면 판별한 이미지와 bbox 값을 반환)
- app copy.py = 커스텀 모델 실행(병 이미지를 넣으면 갈색/투명/녹색 병 중 어떤 병일지를 반환)
- cuda.py = cuda 실행 테스트(방법은 공부중)
- test01.py = yolov5 테스트
- work01.py = 이미지 분류 코드(data 폴더의 sample_image와 sample_json의 파일들을 preprocessed_image와 preprocessed_json에 정리)
- work02.py = 이미지 전처리 코드(분류 기준을 나누어서 각 폴더에 원 이미지들을 bbox 기준으로 잘라내어 224x224 크기로 맞추었음)
- work03.py = sample_data 폴더 생성 후 샘플 이미지를 모아놓기 위한 코드

### 진행과정

- Week03 : 모델 구축을 위한 기초 이미지 수집
- Week04 : 모델 학습을 위한 이미지 전처리
- Week05 : 간단한 이미지 판별 모델 생성 및 훈련
- Week06 : 모델 정확도 업데이트
- Week07 : 데이터셋 정리 및 새로운 라벨링 예정(yolov5용)

### ToDo

- [x] step 0. 모델 학습을 위한 이미지 전처리하기
- [x] step 1. CNN을 활용해 간단한 병 판별 모델 만들어보기
- [x] step 2. 더 큰 모델로 수정하는 과정과 더 정교한 이미지 전처리 과정을 수행하여 정확도 높히기
- [ ] step 3. yolov5 커스텀 모델 만들어보기


