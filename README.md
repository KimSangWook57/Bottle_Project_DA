# Bottle_Project_DA
- 병 판별 프로젝트를 수행하며 배운 데이터 분석 관련 모델과 코드를 모아 놓았습니다. 아직 난잡한 코드가 많습니다.
- 기획서 docs : https://docs.google.com/document/d/1MNx_ZBxme7EzghV2o5YCM3HD9PrhmfMrc4ZL8C719ng/edit?usp=sharing
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

### 설치 예시
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
- sample_data = 전처리 과정(원본 이미지에서, json 파일에 적힌 bbox 좌표값대로 자른 이미지를 224x224 크기로 변환)을 거친 이미지들을 데이터셋에 따른 분류 방식(갈색병/갈색병+포장재/녹색병/녹색병+포장재/투명한병/투명한병+포장재/재사용병(소주병+맥주병)/재사용병+포장재/기타)으로 9개의 폴더에 각각 100개씩 넣어 놓았음

### 코드 설명
- app.py = 모델 실행 
- cuda.py = cuda 실행 테스트(방법은 공부중)
- test01.py = yolov5 테스트
- work01.py = 이미지 분류 코드(data 폴더의 sample_image와 sample_json의 파일들을 preprocessed_image와 preprocessed_json에 정리)
- work02.py = 이미지 전처리 코드(분류 기준을 나누어서 각 폴더에 원 이미지들을 bbox 기준으로 잘라내어 224x224 크기로 맞추었음)
- work03.py = sample_data 폴더 생성 후 샘플 이미지를 모아놓기 위한 코드

### 진행과정

- Week03 : 모델 구축을 위한 기초 이미지 수집
- Week04 : 모델 학습을 위한 이미지 전처리
- Week05 : 간단한 이미지 판별 모델을 만들어 돌릴 예정

### ToDo

- [x] step 0. 모델 학습을 위한 이미지 전처리하기
- [ ] step 1. CNN을 활용해 간단한 병 판별 모델 만들어보기
- [ ] step 2. CNN과 LSTM을 활용해 병 종류와 색깔을 구분하는 모델로 업데이트
- [ ] step 3. 깨진 병 이미지, 상표 이미지 데이터를 구해서 더 세세한 종류, 파손 여부까지 판별할 수 있도록 하기


