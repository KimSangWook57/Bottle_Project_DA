import os
import json
import shutil
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 이미지와 JSON 파일이 있는 폴더 경로
image_folder = 'bottle_photos\\unclassified_glass'
json_folder = 'bottle_labels\\unclassified_glass_json'

# 전처리된 이미지와 JSON 파일을 저장할 폴더 경로
preprocessed_image_folder = 'preprocessed_image\\preprocessed_unclassified_glass'
preprocessed_json_folder = 'preprocessed_json\\preprocessed_unclassified_glass_json'

# 이미지 크기를 통일하기 위한 기준 크기
target_width = 224
target_height = 224

# 이미지 폴더 내의 파일 목록 가져오기
image_files = os.listdir(image_folder)

# 이미지 폴더의 경로와 파일 목록 출력
# print(f"Image folder: {image_folder}")
# print(f"Image files: {image_files}")

# 이미지와 JSON 파일을 순회하며 전처리 작업 수행
for image_file in image_files:
    # 이미지 파일 경로
    image_path = os.path.join(image_folder, image_file)

    # JSON 파일 경로
    json_file = image_file.replace('.jpg', '.json')
    json_file = json_file.replace('.JPG', '.json')
    json_path = os.path.join(json_folder, json_file)

    # JSON 파일 열기
    with open(json_path, encoding="UTF-8") as f:
        json_data = json.load(f)

    # BBOX 값 추출
    objects = json_data['objects']
    bbox_list = []
    for obj in objects:
        if 'class_name' in obj and obj['class_name'].startswith('c_4'):
            bbox = obj['annotation']['coord']
            bbox_list.append(bbox)

    # 이미지 잘라내기 및 전처리
    for idx, bbox in enumerate(bbox_list):
        x, y = bbox['x'], bbox['y']
        width, height = bbox['width'], bbox['height']
        x2, y2 = x + width, y + height

        # 이미지 파일 열기
        image = Image.open(image_path)

        # 이미지 잘라내기
        cropped_image = image.crop((x, y, x2, y2))

        # 이미지 크기를 통일하기 위해 리사이즈
        resized_image = cropped_image.resize((target_width, target_height))

        # 전처리된 이미지 저장
        preprocessed_image_file = f'{image_file.split(".")[0]}_{idx}.jpg'
        preprocessed_image_path = os.path.join(preprocessed_image_folder, preprocessed_image_file)
        resized_image.save(preprocessed_image_path)

    # 전처리된 JSON 파일 저장
    preprocessed_json_path = os.path.join(preprocessed_json_folder, json_file)
    shutil.copy2(json_path, preprocessed_json_path)

    print(f'{image_file} 전처리 완료')

print('이미지 전처리 작업이 완료되었습니다.')
