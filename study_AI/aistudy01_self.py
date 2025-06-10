import torch
import torch.nn as nn

class JY_CNN(nn.Module):
    def __init__(self):
        super(JY_CNN, self).__init__()
        # 1. 여기에 모델에 필요한 '부품(레이어)'들을 정의합니다.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size= 3, padding=1) # Kernel 정의 필요
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2) # Kernel size만 필요
        self.fc1 = nn.Linear(in_features=16*14*14, out_features=10)

    def forward(self, x):
        # 2. 위에서 정의한 부품들을 '조립'하는 순서를 정의합니다.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 16*14*14)
        
        x = self.fc1(x)

        return x
    
# (이전에 작성한 JY_CNN 클래스 코드가 위에 있다고 가정합니다)

# 1. 우리가 설계한 JY_CNN 클래스로 실제 모델 객체를 만듭니다.
model = JY_CNN()
print("모델 구조:\n", model)
print("-" * 30)


# 2. 모델에 입력할 가짜(dummy) 데이터를 생성합니다.
#    (배치크기=1, 채널=1, 높이=28, 너비=28) 모양의 무작위 텐서
dummy_input = torch.randn(1, 1, 28, 28) 
print(f"입력 Tensor 모양: {dummy_input.shape}")


# 3. 모델에 가짜 데이터를 입력하여 순전파(forward pass)를 실행합니다.
#    이때 에러가 나지 않는지 확인하는 것이 핵심!
try:
    output = model(dummy_input)
    print(f"출력 Tensor 모양: {output.shape}")
    print("🎉 성공! 모델이 에러 없이 데이터를 처리했습니다.")

except Exception as e:
    print(f"🔥 에러 발생: {e}")