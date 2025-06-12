import torch
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image
import os

# 1. 변환된 이미지들을 저장할 기본 폴더 생성
output_dir = 'C:/Users/Junyeob/Desktop/Study/AI/data/MNIST_PNG'
os.makedirs(output_dir, exist_ok=True)

# 2. torchvision으로 원본 MNIST 데이터셋 로드 (데이터를 읽는 가장 쉬운 방법)
# transform을 적용하지 않고 순수한 PIL Image 형태로 받기 위해 transform=None으로 설정
train_dataset = datasets.MNIST(root='C:/Users/Junyeob/Desktop/Study/AI/data', train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root='C:/Users/Junyeob/Desktop/Study/AI/data', train=False, download=True, transform=None)

# 3. 데이터셋을 이미지 파일로 변환하여 저장하는 함수
def convert_dataset_to_images(dataset, subset_name):
    """
    dataset: 변환할 데이터셋 (train_dataset 또는 test_dataset)
    subset_name: 저장할 폴더 이름 ('train' 또는 'test')
    """
    subset_dir = os.path.join(output_dir, subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    
    print(f"'{subset_name}' 데이터셋 변환 시작...")
    
    # 데이터셋의 모든 이미지를 하나씩 순회
    for i, (image, label) in enumerate(dataset):
        # 파일명 지정: 예) 00001_5.png (1번 이미지, 정답은 5)
        # i:05d -> 5자리 숫자로 맞추고 앞을 0으로 채움 (예: 1 -> 00001)
        filename = f"{i:05d}_{label}.png"
        filepath = os.path.join(subset_dir, filename)
        
        # PIL Image를 파일로 저장
        image.save(filepath)
        
        # 진행 상황 출력
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(dataset)} 개 이미지 변환 완료...")
            
    print(f"'{subset_name}' 데이터셋 변환 완료! -> {subset_dir} 폴더 확인")


# 4. 학습 데이터셋과 테스트 데이터셋 변환 실행
convert_dataset_to_images(train_dataset, 'train')
convert_dataset_to_images(test_dataset, 'test')