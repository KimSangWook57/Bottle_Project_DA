import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# CUDA 지원 활성화
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA 사용 가능")
else:
    device = torch.device("cpu")
    print("CUDA 사용 불가")

# 시드 고정
seed = 42
torch.manual_seed(seed)

# GPU 메모리 절약
torch.backends.cudnn.benchmark = True

# 이미지 라벨 생성
def get_label_from_foldername(foldername):
    # 각 폴더의 이름에 따라 라벨을 할당
    if foldername == "preprocessed_brown_glass":
        return "0"
    elif foldername == "preprocessed_brown_glass_packaging":
        return "1"
    elif foldername == "preprocessed_clear_glass":
        return "2"
    elif foldername == "preprocessed_clear_glass_packaging":
        return "3"
    elif foldername == "preprocessed_green_glass":
        return "4"
    elif foldername == "preprocessed_green_glass_packaging":
        return "5"
    elif foldername == "preprocessed_reused_glass":
        return "6"
    elif foldername == "preprocessed_reused_glass_packaging":
        return "7"
    elif foldername == "preprocessed_unclassified_glass":
        return "8"
    else:
        raise ValueError(f"Invalid folder name: {foldername}")


# 데이터셋 경로
dataset_path = "preprocessed_image"

# 데이터셋 불러오기
dataset = datasets.ImageFolder(
    dataset_path,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

# 데이터로더 생성
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 레이어 생성
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 111 * 111, 64)
        self.fc2 = nn.Linear(64, 9)  # 클래스 수에 맞게 계속 수정
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.to(device)

# 모델 인스턴스 생성 및 CUDA 장치로 이동
model = Model().to(device)

# 모델 학습시키기
# 손실 함수와 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss().to(device)  # 손실 함수 수정
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 반복
epochs = 10
best_accuracy = 0.0  # 가장 좋은 정확도를 저장할 변수 초기화
best_model_path = "best_model.pt"  # 가장 좋은 모델의 저장 경로

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        # 이미지와 라벨을 CUDA 장치로 이동
        images = images.to(device)
        labels = labels.to(device)
        # 그래디언트 초기화
        optimizer.zero_grad()
        # 모델에 이미지 전달하여 예측 수행
        outputs = model(images)
        # 손실 계산
        loss = criterion(outputs, labels)
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(dataloader)}")

# 테스트 데이터셋을 통한 모델 평가
test_dataset_path = "preprocessed_test_image"
test_dataset = datasets.ImageFolder(
    test_dataset_path,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        # 이미지와 라벨을 CUDA 장치로 이동
        images = images.to(device)
        labels = labels.to(device)

        # 모델에 이미지 전달하여 예측 수행
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 예측 수정

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

    # 가장 좋은 정확도를 가진 모델 저장
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved.")
