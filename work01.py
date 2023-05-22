import json
import os
import shutil
# 원 데이터셋을 1차 분류하는 코드
image_path = "bottle_image"
folder_path = "bottle_label"

def copyFile(image_folder_path, label_folder_path):
    image_filename = data['Image']
    # Copy image file
    shutil.copyfile(os.path.join(image_path, image_filename), os.path.join(image_folder_path, image_filename))
    # Copy JSON file
    shutil.copyfile(os.path.join(folder_path, filename), os.path.join(label_folder_path, filename)) 
    print(f'{image_filename} 전처리 완료')

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), encoding="UTF-8") as json_file:
            data = json.load(json_file)
            for obj in data['objects']:
                if 'class_name' in obj and obj['class_name'].startswith('c_4_01_01'):
                    try:
                        copyFile("bottle_photos\\reused_glass_label", "bottle_labels\\reused_glass_label_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_01_02'):
                    try:
                        copyFile("bottle_photos\\reused_glass", "bottle_labels\\reused_glass_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_02_01_01'):
                    try:
                        copyFile("bottle_photos\\brown_glass_label", "bottle_labels\\brown_glass_label_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_02_01_02'):
                    try:
                        copyFile("bottle_photos\\brown_glass", "bottle_labels\\brown_glass_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_02_02_01'):
                    try:
                        copyFile("bottle_photos\\green_glass_label", "bottle_labels\\green_glass_label_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_02_02_02'):
                    try:
                        copyFile("bottle_photos\\green_glass", "bottle_labels\\green_glass_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break    

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_02_03_01'):
                    try:
                        copyFile("bottle_photos\\clear_glass_label", "bottle_labels\\clear_glass_label_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_02_03_02'):
                    try:
                        copyFile("bottle_photos\\clear_glass", "bottle_labels\\clear_glass_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_03'):
                    try:
                        copyFile("bottle_photos\\unclassified_glass", "bottle_labels\\unclassified_glass_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break

                elif 'class_name' in obj and obj['class_name'].startswith('c_4_03_01'):
                    try:
                        copyFile("bottle_photos\\unclassified_glass_foreign_matter", "bottle_labels\\unclassified_glass_foreign_matter_json")
                    except FileNotFoundError:
                        #print("no image found!")
                        continue
                    break


print("job finish!")            

