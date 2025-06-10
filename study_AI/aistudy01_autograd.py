# 1. PyTorch 라이브러리를 불러옵니다.
import torch

# 2. 'x'라는 Tensor를 만듭니다. 값은 2.0.
#    requires_grad=True 는 "이 Tensor에 대한 미분값을 계산할 거야" 라고 PyTorch에게 알려주는 중요한 표시입니다.
x = torch.tensor(2.0, requires_grad=True)

# 3. 간단한 수식을 정의합니다. y = x² + 5
y = x**2 + 5

# 4. y를 x에 대해 미분하라는 "마법의 명령어"입니다.
#    y를 x로 미분하면 2x가 되죠. x가 2.0이었으니, 미분 결과(기울기)는 4.0이 될 겁니다.
y.backward()

# 5. x의 미분값(gradient)을 출력해봅니다.
print(f"x의 값: {x.item()}")
print(f"y의 값: {y.item()}")
print(f"y를 x로 미분한 값 (x.grad): {x.grad.item()}")

# 예상 출력:
# x의 값: 2.0
# y의 값: 9.0
# y를 x로 미분한 값 (x.grad): 4.0