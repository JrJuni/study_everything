from torch.utils.data import Dataset, DataLoader
from PIL import Image  # <-- 지적해주신 Image 라이브러리
import os

# (참고용) PNG 파일들을 읽는 간단한 Dataset 클래스 예시
class CustomPNGDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir) # 폴더 안의 모든 파일 리스트를 가져옴

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # '00001_5.png' 와 같은 파일명
        filename = self.image_files[idx]
        filepath = os.path.join(self.image_dir, filename)
        
        # PIL 라이브러리로 이미지 열기
        image = Image.open(filepath).convert("L") # 'L'은 흑백 모드
        
        # 파일명에서 레이블(정답) 추출
        label = int(filename.split('_')[1].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    