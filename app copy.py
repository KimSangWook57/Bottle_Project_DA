import io
import os
import base64
import yolov5
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from flask import Flask, request, jsonify
from PIL import Image
from testmodel import Model

app = Flask(__name__)
model = None

DETECTION_URL = '/api/predict'
RESULTS_DIR = 'results'  # 결과 이미지 저장할 폴더

@app.route(DETECTION_URL, methods=['POST'])
def predict():
    try:
        # POST 형식인지 확인
        if request.method != 'POST':
            raise ValueError('POST 형식이 아닙니다.')
        # 이미지를 받았는지 확인
        if not request.files.get('image'):
            raise ValueError('이미지 파일을 찾을 수 없습니다.')
        # 모델이 로드되었는지 확인
        if not model:
            raise ValueError('모델이 로드되지 않았습니다.')
        # 받은 이미지 로딩
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        # 이미지 크기 조정
        target_size = (400, 400)
        transform = transforms.Resize(target_size)
        im = transform(im)

        # 이미지를 Tensor로 변환
        transform = transforms.ToTensor()
        im_tensor = transform(im)

        # 이미지를 Tensor로 변환
        transform = transforms.ToTensor()
        # 이미지 차원을 추가하여 4D Tensor로 변환
        im_tensor = torch.unsqueeze(transform(im), 0)
        # 모델에 Tensor 이미지를 넣어서 판별
        results = model(im_tensor)

        # 결과를 사진으로 저장
        # results.save(save_dir=RESULTS_DIR)
        # image_path = os.path.join(RESULTS_DIR, 'image0.jpg')
        # 이미지를 Base64로 인코딩하여 JSON 응답에 포함
        # with open(image_path, 'rb') as f:
        #     encoded_image = base64.b64encode(f.read()).decode('utf-8')
        # 변환한 이미지와 bbox 값을 딕셔너리에 담아서 JSON으로 전달
        response_data = {
            'bbox': results.tolist()
        }

        # 결과 폴더 삭제
        # shutil.rmtree(RESULTS_DIR)
        return jsonify(response_data)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # model = yolov5.load('keremberke/yolov5s-garbage')
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5n - yolov5x6 official model
        # model = torch.load('best_model.pt')  # custom model
        # # 모델 초기화
        # model = TheModelClass()

        # # 옵티마이저 초기화
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # # 모델의 state_dict 출력
        # print("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # # 옵티마이저의 state_dict 출력
        # print("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        #     print(var_name, "\t", optimizer.state_dict()[var_name])

        # 모델 초기화
        model = Model()

        # 저장된 모델 로드
        model_path = 'best_model.pt'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # cpu load
        model.eval()

    except Exception as e:
        print(f'Failed to load model: {str(e)}')
        exit(1)
    
    app.run(host='0.0.0.0', debug=True, port=55000)
