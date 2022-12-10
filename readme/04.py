from torch import nn

import torch

# 要扩展的图片
input_tensor = torch.randn([64 * 8, 4, 4])
in_dim = 64 * 8
out_dim = 64 * 4

# 这套参数可以吧图片的宽高扩展一倍
model = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                           padding=2, output_padding=1, bias=False)

output_tensor = model(input_tensor)
print(output_tensor.shape)  # [256,8,8]
