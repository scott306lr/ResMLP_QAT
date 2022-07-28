import torch

#print(torch.zeros(1).view(1, -1))
#print(torch.)
T1 = torch.tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# T2 = torch.tensor([[10, 20, 30],
#                 [40, 50, 60],
#                 [70, 80, 90]])

# # print(torch.stack((T1, T2), dim=0))
# print(torch.stack((T1, T2), dim=1))
# # print(torch.stack((T1, T2), dim=2))

# print(torch.max(torch.stack([T1, T2], dim=1), dim=1))
print(T1)

print(T1.view(T1.size(0), -1))