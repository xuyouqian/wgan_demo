### 参考链接

李宏毅2022 机器学习课程homework 6链接

https://colab.research.google.com/drive/10lHBPFoNhTiiPe-yZ7SwAV1wwrkGc4En?usp=sharing#scrollTo=Jg4YdRVPYJSj

wgan 链接
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py





### 模型损失函数简单说明:

模型G（生成模型）： 接受一个10维向量  返回 [3,64,64 ]的向量

模型D （判别模型）：接受[3,64,64] 向量，返回一个1维向量



#### GAN 的损失函数

判别模型的损失函数:

生成模型会生成一堆图片  给他们打标为0

训练集本来的图片打标为1 

把这些数据丢给判别模型，让损失最小

```python
r_label = torch.ones((bs)).cuda()  # 原来训练集的标签
f_label = torch.zeros((bs)).cuda() # 生成器生成图片的标签
r_logit = self.D(r_imgs)
f_logit = self.D(f_imgs)

r_loss = self.loss(r_logit, r_label)
f_loss = self.loss(f_logit, f_label)
loss_D = (r_loss + f_loss) / 2
```

生成模型的损失函数:

```python
# 生成模型的损失函数要用真实标签 1
loss_G = self.loss(f_logit, r_label)
```

#### wgan 的损失函数

注意：

wgan 要修改原来的模型代码，取消掉判别模型最后的sigmoid 函数

判别模型的损失函数：

```python
loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
```

生成模型的损失函数

```python
loss_G = -torch.mean(f_logit)
```

