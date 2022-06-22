import torch
from e3nn import o3, nn

## Helpers ##
class o3GatedLinear(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=torch.tanh):
        super(o3GatedLinear, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify()
        self.gate = nn.Gate(irreps_scalars, [act for _, ir in irreps_scalars],
                            irreps_gates, [act for _, ir in irreps_gates], irreps_gated)
        self.lin = o3.Linear(self.irreps_input, self.gate.irreps_in)

    def forward(self, x):

        return self.gate(self.lin(x))

class LayerNorm(torch.nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
    
    def forward(self, data):
        if torch.is_tensor(data):
            data = data - data.mean(0, keepdim=True)
            data = data * (data.pow(2).mean(1, keepdim=True) + 1e-12).pow(-0.5)
        else:
            data.hn = data.hn - data.hn.mean(0, keepdim=True)
            data.hn = data.hn * (data.hn.pow(2).mean(1, keepdim=True) + 1e-12).pow(-0.5)
            data.he = data.he - data.he.mean(0, keepdim=True)
            data.he = data.he * (data.he.pow(2).mean(1, keepdim=True) + 1e-12).pow(-0.5)
        return data
