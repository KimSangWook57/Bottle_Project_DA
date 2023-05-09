import torch
import yolov5

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5m")  # or yolov5n - yolov5x6, custom
# model = yolov5.load("ultralytics/yolov5", "fcakyon/yolov5s-v7.0") - yolov5 최신 버전에서만 작동함
# 기본 모델 중에서는 m이 결과가 가장 좋은듯?

# Images
# or file, Path, PIL, OpenCV, numpy, list
# img = "https://ultralytics.com/images/bus.jpg"  
img = "images\\20230506_115535.jpg"

# Inference
results = model(img)

# Results
# or .show(), .save(), .crop(), .pandas(), etc.

# 결과를 터미널에 출력
results.print()  
print("============================")
# 결과를 창을 띄워 표시
results.show()
print("============================")
# 검출한 사진을 파일로 저장
# results.save()
# print("============================")
# # 각 검출한 물체들을 개별 파일로 저장
# results.crop()
# print("============================")