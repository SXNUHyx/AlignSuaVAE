import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='lrelu', slope=0.1, device='cpu'):
        super().__init__()
        layers = []
        act_cls = nn.LeakyReLU(slope) if activation == 'lrelu' else nn.ReLU()

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_cls)
        # 中间隐藏层
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls)
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers).to(device)
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)