import torch
from model import NCF
import torch.optim as optim

train_inter = torch.tensor([[1,0,1,4],
                            [1,1,1,1],
                            [1,2,1,1],
                            [1,3,1,5],
                            [1,4,1,1],
                            [1,5,1,4],
                            [1,6,0,0],
                            [1,7,0,0],
                            [1,8,0,0],
                            [0,0,1,4],
                            [0,1,1,1],
                            [0,2,1,1],
                            [0,3,1,5],
                            [0,4,1,1],
                            [0,5,1,4],
                            [0,6,0,0],
                            [0,7,0,0],
                            [0,8,0,0]])

model = NCF(2,9,30,2,0.2,'MLP')
BCE_loss = torch.nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

user_list = train_inter[:,0]
item_list = train_inter[:,1]
label = train_inter[:,2]

for i in range(10):
    model.zero_grad()
    pred = model(user_list, item_list)
    loss = BCE_loss(pred, label.float())
    loss.backward()
    optimizer.step()

BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
pred = model(user_list, item_list)
loss = BCE_loss(pred, label.float())
print(loss)
