import torch
from functools import partial

from torch import nn


class XmodAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bottleneck_size = config.hidden_size // config.adapter_reduction_factor
        self.dense1 = nn.Linear(config.hidden_size, self.bottleneck_size)
        self.dense2 = nn.Linear(self.bottleneck_size, config.hidden_size)
        self.adapter_act_fn = nn.functional.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.adapter_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states