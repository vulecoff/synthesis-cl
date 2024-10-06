import torch
from torch import nn

class Hebbian(nn.Module): 
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        self.fc1.requires_grad_(False)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.flatten = nn.Flatten(0, -1)
    
    def forward(self, x): 
        r = self.fc1(x)
        return nn.ReLU(r)


def main():
    x = Hebbian(2)
    print(x.state_dict())

if __name__ == "__main__":
    main()