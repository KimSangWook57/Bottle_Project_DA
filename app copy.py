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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 색감 조정
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 이미지 정규화
        ])

        # 이미지 차원을 추가하여 4D Tensor로 변환
        im_tensor = torch.unsqueeze(transform(im), 0)
        # 모델에 Tensor 이미지를 넣어서 판별
        results = model(im_tensor)

        # 변환한 이미지와 bbox 값을 딕셔너리에 담아서 JSON으로 전달
        response_data = {
            'key': results.tolist()
        }

        # 모델의 출력값 가져오기
        output_values = results[0].tolist()

        # 가장 높은 출력값과 해당 인덱스 가져오기
        max_value = max(output_values)
        max_index = output_values.index(max_value)

        # 인덱스에 해당하는 클래스 가져오기
        class_mapping = ["brown_bottle", "clear_bottle", "green_bottle"]
        predicted_class = class_mapping[max_index]

        # 결과 출력
        print("모델이 판단한 클래스:", predicted_class)

        # 결과를 JSON으로 전달
        response_data['predicted_class'] = predicted_class

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
