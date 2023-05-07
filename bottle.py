from flask import Flask, jsonify, request
import torch
import cv2
import numpy as np
import json
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

app = Flask(__name__)

device = select_device('')
model = attempt_load('yolov5x.pt', map_location=device)

def predict(image):
    # 이미지 전처리
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    # 모델 추론
    with torch.no_grad():
        detections = model(img.to(device))[0]
        detections = non_max_suppression(detections, conf_thres=0.3, iou_thres=0.45)

    # 결과 출력
    result = []
    for detection in detections:
        for x1, y1, x2, y2, conf, cls in detection:
            result.append({'class': int(cls), 'confidence': float(conf), 'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)})
    return result

@app.route('/predict', methods=['POST'])
def predict_api():
    # 이미지 파일을 받습니다.
    file = request.files['file']
    # 이미지를 numpy 배열로 변환합니다.
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # 내용물을 추출합니다.
    result = predict(image)
    # 결과를 JSON 파일로 변환합니다.
    json_result = json.dumps(result)
    return json_result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
