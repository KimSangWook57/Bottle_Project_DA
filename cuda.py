import tensorflow as tf
import torch
# CUDA 환경이 설치되었는지 확인하는 코드
print(tf.test.is_built_with_cuda()) # 텐서플로우
print(torch.cuda.is_available()) # 파이토치
