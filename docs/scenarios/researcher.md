# Scenario: You are a PDE + ML researcher

## Simple Example

Say you want to evaluate a conditioned version of `DilatedResNet`.
You can add a version of `DilatedBasicBlock` that works with an embedding vector as follows:

```py
class CondDilatedBasicBlock(nn.Module):
    """Basic block for Dilated ResNet

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.cond_emb = nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = x
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]

        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x + emb_out
```

Now we can add an appropriate instantiation of the model in the [`COND_MODEL_REGISTRY`](https://github.com/microsoft/pdearena/blob/main/pdearena/models/registry.py):

```py
COND_MODEL_REGISTRY["DilResNet-128"] = {
    "class_path": "pdearena.modules.conditioned.twod_resnet.ResNet",
    "init_args": {
        "hidden_channels": 128,
        "norm": True,
        "num_blocks": [1, 1, 1, 1],
        "block": CondDilatedBasicBlock
    }
}
```

Finally you can train this model by appropriately setting up `model.name=DilResNet-128` in the training config:
```yaml
model:
    name: DilResNet-128
    max_num_steps: 5
    activation: "gelu"
    criterion: mse
    lr: 1e-3
    param_conditioning: "scalar"
```
