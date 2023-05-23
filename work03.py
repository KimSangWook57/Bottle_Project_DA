import os
import random
import shutil

# 입력과 출력 폴더 경로 설정
input_folder = "preprocessed_image"
output_folder = "sample_data"

# 폴더 리스트 생성
folders = [
    "preprocessed_brown_glass",
    "preprocessed_brown_glass_packaging",
    "preprocessed_clear_glass",
    "preprocessed_clear_glass_packaging",
    "preprocessed_green_glass",
    "preprocessed_green_glass_packaging",
    "preprocessed_reused_glass",
    "preprocessed_reused_glass_packaging",
    "preprocessed_unclassified_glass"
]

# 샘플 이미지 개수 설정
sample_count = 100

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 각 폴더에서 샘플 이미지 선택
for folder in folders:
    input_path = os.path.join(input_folder, folder)
    output_path = os.path.join(output_folder, "sample_" + folder)

    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 폴더에서 이미지 리스트 가져오기
    images = os.listdir(input_path)

    # 이미지 개수가 충분하지 않으면 모든 이미지 선택
    if len(images) <= sample_count:
        sampled_images = images
    else:
        # 샘플 이미지 선택
        sampled_images = random.sample(images, sample_count)

    # 샘플 이미지를 출력 폴더로 복사
    for image in sampled_images:
        image_path = os.path.join(input_path, image)
        output_image_path = os.path.join(output_path, image)
        shutil.copy(image_path, output_image_path)
