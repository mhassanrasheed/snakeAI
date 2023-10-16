import torch.nn as nn


class SnakeAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(24, 16, bias=True),
            nn.ReLU(),
            # nn.Linear(16, 8, bias=True),
            # nn.ReLU(),
            nn.Linear(16, 4, bias=True),
            nn.Softmax(-1)
        )

    def forward(self, inputs):
        x = self.fc(inputs)
        return x
