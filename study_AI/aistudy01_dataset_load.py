# (JY_CNN 클래스와 model 객체 생성 코드는 이미 위에 있다고 가정)

# 필요한 라이브러리 불러오기
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 데이터 변환(Transform) 정의
# 다운로드한 MNIST 데이터는 이미지 파일(PIL Image) 형태입니다.
# 이걸 우리 모델이 이해할 수 있는 PyTorch Tensor로 변환해줘야 합니다.
transform = transforms.ToTensor()

# 2. MNIST 데이터셋 다운로드 및 로드 (학습용 / 테스트용)
# root='./data': 데이터를 내려받을 폴더
# train=True: 학습용 데이터를 불러옴
# download=True: 해당 폴더에 데이터가 없으면 자동으로 다운로드
# transform=transform: 위에서 정의한 변환 규칙을 적용
train_dataset = datasets.MNIST(root='C:/Users/Junyeob/Desktop/Study/AI/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='C:/Users/Junyeob/Desktop/Study/AI/data', train=False, download=True, transform=transform)

# 3. 데이터로더 생성
# dataset: 사용할 데이터셋
# batch_size: 한 번에 모델에 공급할 데이터 개수
# shuffle=True: 데이터 순서를 매번 섞어서, 모델이 데이터 순서를 외우는 과적합(overfitting)을 방지 (학습 시에만 사용)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 4. 데이터가 잘 로드되었는지 확인해봅시다.
print("--- 데이터 로드 완료 ---")
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"한 배치(batch)의 이미지 모양: {images.shape}")
print(f"한 배치(batch)의 레이블 모양: {labels.shape}")