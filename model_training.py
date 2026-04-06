import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np



class TerrainDataset(Dataset):
    def __init__(self,filepath):
        xy=pd.read_csv(filepath)

        self.len=xy.shape[0]

        self.x_data=torch.from_numpy(xy[['x','y']].values.copy()).to(torch.float32)
        self.y_data=torch.from_numpy(xy[['z']].values.copy()).to(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

dataset = TerrainDataset('terrain_data.csv')
print("拿到第一条数据：",dataset[0])
train_loader =DataLoader(dataset=dataset,
                         batch_size=32,
                         shuffle=True,
                         num_workers=0)

class TerrainModel(nn.Module):
    def __init__(self):
        super(TerrainModel,self).__init__()

        self.linear1=nn.Linear(2,64)

        self.linear2=nn.Linear(64,32)

        self.linear3=nn.Linear(32,1)

        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out
model = TerrainModel()

criterion = nn.MSELoss()
optimizer =torch.optim.SGD(model.parameters(),lr=0.001)

loss_history = []  # 新增：用来记录画图数据的本子

for epoch in range(1000):
    total_loss=0.0
    for batch_idx,(x_data,y_data) in enumerate(train_loader):
        y_pred =model(x_data)
        loss=criterion(y_pred,y_data)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss +=loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    if(epoch+1)%10==0:

        print(f'第 {epoch + 1} 遍刷题完成 | 平均误差 (Loss): {avg_loss:.6f}')

plt.plot(loss_history)
plt.title('Terrain AI - Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()

model.eval()

x_test = np.linspace(-10,10,50)
y_test = np.linspace(-10,10,50)

x_grid,y_grid = np.meshgrid(x_test,y_test)

test_features=torch.tensor(
    np.column_stack((x_grid.flatten(),y_grid.flatten())),
    dtype=torch.float32
)

with torch.no_grad():
    z_pred =model(test_features).numpy()

z_grid = z_pred.reshape(x_grid.shape)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='coolwarm', alpha=0.8)
ax.set_title('AI Reconstructed 3D Terrain')
ax.set_xlabel('X')
...
plt.show()
