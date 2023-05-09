import argparse
import io
import torch
import yolov5

from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)
model = None

DETECTION_URL = '/api/predict'

@app.route(DETECTION_URL, methods=['POST'])
def predict():
    try:
        if request.method != 'POST':
            raise ValueError('Only POST requests are allowed')
        
        if not request.files.get('image'):
            raise ValueError('No image file found in the request')
        
        if not model:
            raise ValueError('Model has not been loaded yet')
        
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        
        results = model(im, size=640)
        return jsonify(results.pandas().xyxy[0].to_dict('records'))
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    try:
        model = yolov5.load('keremberke/yolov5s-garbage')
    except Exception as e:
        print(f'Failed to load model: {str(e)}')
        exit(1)
    
    app.run(host='0.0.0.0', debug=True, port=5000)
