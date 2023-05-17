# Bottle_Project_DA
- 병 판별 프로젝트를 수행하며 배운 데이터 분석 관련 모델과 코드를 모아 놓았습니다. 아직 난잡한 코드가 많습니다.

## 실행방법
### 가상환경 및 모델학습에 사용한 버전
- Windows 10
- Python 3.10.10
- tensorflow 2.10.0
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
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 코드 설명
- app.py = 모델 실행 
- cuda.py = cuda 실행 테스트(방법은 공부중)
- test01.py = yolov5 테스트
- work02.py = 이미지 전처리 코드(data 폴더의 sample_image와 sample_json의 파일들을 preprocessed_image와 preprocessed_json에 정리)



