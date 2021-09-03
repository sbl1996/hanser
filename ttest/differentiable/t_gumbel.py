import torch
import torch.nn.functional as F

weight = torch.tensor([2., 3., 4.], requires_grad=True)
hardwts = F.gumbel_softmax(weight, hard=True)
index = torch.max(hardwts, axis=0)[1]

x = torch.tensor([1, 2, 3.])

y = sum(x * w if i == index else w for i, w in enumerate(hardwts))
y.sum().backward()