import kagglehub
import shutil
import os
import kaggle

# 1. 기본 경로로 다운로드
path = kagglehub.dataset_download("zaraks/pascal-voc-2007")
print("기본 다운로드 경로:", path)

# 2. 원하는 경로 지정
save_path = "C:\\Users\\Junyeob\\Desktop\\Study\\AI\\data\\pascal_voc_2007"
# 경로 설정 시 r"\U" 주의

# 3. 이동
if not os.path.exists(save_path):
    shutil.copytree(path, save_path)

print("복사 완료:", save_path)
