import torch
import torchvision

print("✅ PyTorch 버전:", torch.__version__)
print("✅ TorchVision 버전:", torchvision.__version__)
print("✅ CUDA 사용 가능 여부:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("✅ CUDA 버전 (드라이버 기준):", torch.version.cuda)
    print("✅ 사용 중인 GPU 이름:", torch.cuda.get_device_name(0))
    print("✅ 현재 디바이스 ID:", torch.cuda.current_device())
else:
    print("⚠️ CUDA 사용 불가 (CPU 모드로 실행 중)")
