import torch
from torch import nn
from torch.nn import functional as func
class SingleLayer(nn.Module): 
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.flatten = nn.Flatten(0, -1)
    def forward(self, x): 
        r =  func.relu(self.fc1(x))
        r = func.sigmoid(self.fc2(r))
        return self.flatten(r)
        # return r

class Perceptron(nn.Module): 
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        self.flatten = nn.Flatten(0, -1)
    def forward(self, x): 
        r = func.sigmoid(self.fc1(x))
        return self.flatten(r)

from torch import optim    
def train(model: nn.Module, train_x, train_y, epochs=100, lr=0.01): 
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses

def test(model: nn.Module, test_x, test_y): 
    pred = model(test_x)
    correct = 0
    model.eval()
    for i in range(len(test_y)): 
        if pred[i].item() > 0.5: 
            prediction = 1
        else: 
            prediction = 0
        if prediction == test_y[i].item(): 
            correct += 1
        print(pred[i].item(), test_y[i].item())
    print("Accuracy:", correct / len(test_y) * 100, "%",
           "   ;    or ", "{}/{}".format(correct, len(test_y)))

import seaborn as sns

def build_dataset(io: list): 
    ins = []
    outs = []
    for item in io: 
        ins.append(item[0])
        outs.append(item[1])
    X = torch.tensor(ins, dtype=torch.float32)
    y = torch.tensor(outs, dtype=torch.float32)
    return X, y

if __name__ == "__main__":
    ip = torch.randn(1, 1)
    print(ip)
    f = nn.Flatten()
    print(f(ip))
    print(torch.flatten(ip))