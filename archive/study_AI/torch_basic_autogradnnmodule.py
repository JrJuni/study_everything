import torch
import torch.nn as nn

'''
# Autograd : Sample code
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 2*x + 1
print(f"y: {y}")

y.backward()
print(f"dy/dx: {x.grad}")

# nn.Module 
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 1. 부모 클래스의 생성자 호출_필수수
        super(SimpleNet, self).__init__()

        # 2. 모델에서 사용할 계층들 정의
        self.layer1 = nn.Linear(input_size, hidden_size)  # 선형 계층 (입력 -> 은닉)
        self.relu = nn.ReLU()                             # 활성화 함수 ReLU
        self.layer2 = nn.Linear(hidden_size, output_size) # 선형 계층 (은닉 -> 출력)

    def forward(self, x):
        # 3. 순전파 로직 정의: 입력 x가 계층들을 통과하는 과정
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# 모델 사용 예시 (간단히)
input_size = 10  # 입력 데이터의 특징 수
hidden_size = 20 # 은닉층의 뉴런 수
output_size = 1  # 출력 값의 수 (예: 회귀 문제)

# 모델 객체 생성
model = SimpleNet(input_size, hidden_size, output_size)
print(model) # 모델 구조 출력

# 임의의 입력 데이터 생성 (배치 크기 5, 입력 특징 10)
input_tensor = torch.randn(5, input_size)

# 모델에 입력 데이터를 넣어 출력 얻기
output = model(input_tensor) # model.forward(input_tensor)와 동일
print(output.shape)

class MyCustomModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyCustomModule, self).__init__()
        # 1: 첫 번째 Linear 계층을 정의하세요.
        # 이름은 self.fc1 로 하고, input_dim에서 hidden_dim으로 변환합니다.
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 2: ReLU 활성화 함수를 정의하세요.
        # 이름은 self.relu 로 합니다.
        self.relu = nn.ReLU()

        # 3: 두 번째 Linear 계층을 정의하세요.
        # 이름은 self.fc2 로 하고, hidden_dim에서 output_dim으로 변환합니다.
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 4: 순전파 로직을 구현하세요.
        # 입력 x가 self.fc1 -> self.relu -> self.fc2 를 순서대로 통과하도록 합니다.
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # 여기에 순서대로 연산을 적용하는 코드를 작성하세요.
        # out = self.fc1(out)
        # out = ...
        # out = ...
        return out

# --- 이 아래는 테스트를 위한 코드입니다. 수정하지 마세요. ---
input_dim = 5
hidden_dim = 8
output_dim = 2

# 모델 인스턴스 생성
try:
    model_test = MyCustomModule(input_dim, hidden_dim, output_dim)
    print("모델 구조:")
    print(model_test)

#     # 테스트용 더미 입력 데이터 생성 (배치 크기 1, 입력 특징 5)
    dummy_input = torch.randn(1, input_dim)
    print(f"\n더미 입력 (shape: {dummy_input.shape}):\n{dummy_input}")

#     # 모델 포워드 패스 실행
    output = model_test(dummy_input)
    print(f"\n모델 출력 (shape: {output.shape}):\n{output}")

#     # 파라미터 확인 (옵션)
    print("\n모델 파라미터:")
    for name, param in model_test.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

except Exception as e:
    print(f"코드를 완성하는 중 오류가 발생했을 수 있습니다: {e}")
    print("TODO 부분을 확인하고 다시 시도해주세요!")
# ---------------------------------------------------------
'''