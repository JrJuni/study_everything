import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from aistudy01_self import JY_CNN
from aistudy01_png_loader import CustomPNGDataset

# 1-1. 이미지에 적용할 변환(Transform) 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)) # 흑백 이미지 정규화 (옵션)
])

# 1-2. 커스텀 Dataset 객체 생성 (메뉴판 만들기)
# './MNIST_PNG/train'는 이전에 PNG 파일들을 저장한 실제 경로로 지정해야 합니다.
train_dataset = CustomPNGDataset(image_dir='C:/Users/Junyeob/Desktop/Study/AI/data/MNIST_PNG/train', transform=transform)

# 1-3. Dataset을 DataLoader에 전달 (서버 고용하기)
# 이 train_loader를 아래 학습 루프에서 사용합니다.
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 2-1. 모델, 손실 함수, 옵티마이저 정의
model = JY_CNN()
criterion = nn.CrossEntropyLoss()
# optimizer에게 모델의 어떤 파라미터를 수정할지(model.parameters()) 알려주고,
# 학습률(learning rate, lr)을 설정합니다. lr은 '얼마나 큰 보폭으로 가중치를 수정할지'를 의미합니다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 2-2. 학습 루프 설정
num_epochs = 20 # 테스트를 위해 우선 1 에포크만 실행

print("--- 학습 시작 ---")

# 2-3. 에포크(Epoch) 만큼 반복
for epoch in range(num_epochs):
    # train_loader에서 미니배치(mini-batch)를 하나씩 꺼내옴
    # images: (64, 1, 28, 28) 모양의 텐서
    # labels: (64) 모양의 텐서
    for i, (images, labels) in enumerate(train_loader):
        
        # --- 여기가 바로 우리가 배운 5단계 사이클! ---
        # 1. 순전파 (Forward pass)
        outputs = model(images)
        
        # 2. 오차 계산
        loss = criterion(outputs, labels)
        
        # 3. 기울기 초기화
        optimizer.zero_grad()
        
        # 4. 역전파 (Backward pass)
        loss.backward()
        
        # 5. 파라미터 업데이트
        optimizer.step()
        # --- 5단계 사이클 끝 ---

        # 100번째 배치마다 로그 출력
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("--- 학습 종료 ---")

# 2-4. --- 모델 저장 단계 ---
MODEL_SAVE_PATH = "./MNIST_model.pt" # .pth 또는 .pt 확장자를 주로 사용합니다.
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"학습된 모델이 {MODEL_SAVE_PATH} 경로에 저장되었습니다.")


