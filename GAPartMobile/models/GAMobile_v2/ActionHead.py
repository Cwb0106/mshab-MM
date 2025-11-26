import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActionHead(nn.Module):
    def __init__(self, in_features, hidden_dim=256, action_dim=2, head_type="mlp"):
        super().__init__()
        self.head_type = head_type
        if self.head_type == "flowmatching":
            self.net = nn.Sequential(
            nn.Linear(action_dim + in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            )
        elif self.head_type == "mlp":
            self.net = nn.Sequential(
                layer_init(nn.Linear(in_features, 2048)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(2048, 1024)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(1024, 512)),
                nn.ReLU(inplace=True),
                layer_init(
                    nn.Linear(512, np.prod(action_dim)),
                    std=0.01 * np.sqrt(2),
                ),
            )
        else:
            raise ValueError(f"Unsupported action head type: '{self.head_type}'. Must be 'flowmatching' or 'mlp'.")

    def forward(self, in_features, noisy_action=None, time_step=None, condition=None):
        if self.head_type == "flowmatching":
            time_step_emb = time_step.expand(-1, condition.size(0) // time_step.size(0) if self.training else 1)
            net_input = torch.cat([noisy_action, condition], dim=1)
            predicted_vector_field = self.net(net_input)
        elif self.head_type == "mlp":
            predicted_vector_field = self.net(in_features)
        return predicted_vector_field


class FlowMatchingHead(nn.Module):
    def __init__(self, condition_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, time_step, condition):
        time_step_emb = time_step.expand(-1, condition.size(0) // time_step.size(0) if self.training else 1)
        net_input = torch.cat([noisy_action, condition], dim=1)
        predicted_vector_field = self.net(net_input)
        return predicted_vector_field