import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from aistudy01_self import JY_CNN
from aistudy01_png_loader import CustomPNGDataset

# 1-1. 이미지에 적용할 변환(Transform) 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)) # 흑백 이미지 정규화 (옵션)
])

# 3-1. 모델 구조가 똑같은 새 모델 객체를 '빈 껍데기'로 생성
model_to_evaluate = JY_CNN()

# 3-2. 저장된 가중치 파일 경로 지정
MODEL_SAVE_PATH = "./MNIST_model.pt"

# 3-3. 저장된 가중치(state_dict)를 불러와서 모델에 덮어씌우기
model_to_evaluate.load_state_dict(torch.load(MODEL_SAVE_PATH))

# 3-4. 평가용 데이터로더 준비
test_dataset = CustomPNGDataset(image_dir='C:/Users/Junyeob/Desktop/Study/AI/data/MNIST_PNG/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# --- 모델 평가 단계 ---

print("--- 저장된 모델로 평가 시작 ---")

model_to_evaluate.eval() # 모델을 평가 모드로 전환

with torch.no_grad(): # 기울기 계산 비활성화
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model_to_evaluate(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'테스트 데이터에 대한 정확도: {accuracy:.2f} %')