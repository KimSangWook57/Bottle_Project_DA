import io
import os
import base64
import yolov5
import torch

from flask import Flask, request, jsonify
from PIL import Image

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
        # 모델에 사진을 넣어서 판별
        results = model(im, size=640)
        # 결과를 사진으로 저장
        results.save(save_dir=RESULTS_DIR)
        image_path = os.path.join(RESULTS_DIR, 'image0.jpg')
        # 이미지를 Base64로 인코딩하여 JSON 응답에 포함
        with open(image_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
        # 변환한 이미지와 bbox 값을 딕셔너리에 담아서 JSON으로 전달
        response_data = {
            'image': {
                'data': encoded_image,
                'format': 'base64'
            },
            'bbox': results.pandas().xyxy[0].to_dict('records')
        }
        return jsonify(response_data)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model = yolov5.load('keremberke/yolov5s-garbage')
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5n - yolov5x6 official model
        # model = torch.load('best_model.pt')  # custom model
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
    except Exception as e:
        print(f'Failed to load model: {str(e)}')
        exit(1)
    
    app.run(host='0.0.0.0', debug=True, port=55000)
