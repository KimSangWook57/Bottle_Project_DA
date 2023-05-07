import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
# or file, Path, PIL, OpenCV, numpy, list
# img = "https://ultralytics.com/images/bus.jpg"  
# img = "flask-yolov5\yolov5\data\images\\bus.jpg"
img = "flask-yolov5\yolov5\data\images\\20230506_115535.jpg"

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
# # 각 검출한 물체를 개별로 저장
# results.crop()
# print("============================")