from .subnets import *
from .layers import *
ACT_DICT = {'tanh': Tanh(), 'relu': ReLU(), 'silu': SiLU(), 'sigmoid': Sigmoid()}

## Build Model ##
def build_model(model_type, irreps_in, irreps_out,
                latent_layers, latent_scalars, latent_vectors, latent_tensors=0, 
                node_basis=1, reservoir=False, **kwargs):
    if 'act' in kwargs:
        kwargs['act'] = ACT_DICT[kwargs['act']]
    irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
    irreps_latent = o3.Irreps(f'{latent_scalars:g}'+'x0e + '+ \
        f'{latent_vectors:g}'+'x1o + '+ \
        f'{latent_tensors:g}'+'x2e')
    irreps_out = o3.Irreps(irreps_out)

    if model_type == 'equivariant':
        layer_type = Eq_NLMP2
    elif model_type == 'non-equivariant':
        layer_type = nEq_NLMP2
    elif model_type == 'non-equivariant isotropic':
        layer_type = nEq_NLMP_iso
    elif model_type == 'non-equivariant anisotropic':
        layer_type = nEq_NLMP_aniso
    elif model_type == 'GCN':
        layer_type = GCN
    elif model_type == 'Edge':
        layer_type = Edge
    else:
        print('Unknown model type: ', model_type)

    enc = Encoder(irreps_in, irreps_latent, model_type, **kwargs)
    f = Latent(layer_type, irreps_latent, latent_layers, **kwargs)
    dec = Decoder(irreps_latent, irreps_out, model_type, **kwargs)
    if reservoir:
        for p in enc.parameters():
            p.requires_grad = False
        for p in f.parameters():
            p.requires_grad = False
        for p in dec.node_dec[0].parameters():
            p.requires_grad = False
    return NODE(enc, f, dec, **kwargs)