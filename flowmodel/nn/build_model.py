from .subnets import *
from .layers import *
ACT_DICT = {'tanh': Tanh(), 'relu': ReLU(), 'silu': SiLU(), 'sigmoid': Sigmoid()}

## Build Model ##
def build_model(model_type, irreps_in, irreps_out,
                latent_layers, latent_scalars, latent_vectors, latent_tensors=0, 
                node_basis=1, **kwargs):
    if 'act' in kwargs:
        kwargs['act'] = ACT_DICT[kwargs['act']]

    if model_type == 'equivariant':
        # irreps_in = f'{in_scalars:g}'+'x0e + '+f'{in_vectors:g}'+'x1o'
        # irreps_out = f'{out_scalars:g}'+'x0e + '+f'{out_vectors:g}'+'x1o'
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_latent = f'{latent_scalars:g}'+'x0e + '+ \
            f'{latent_vectors:g}'+'x1o + '+ \
            f'{latent_tensors:g}'+'x2e'
        enc = Encoder(irreps_in, irreps_latent, model_type, **kwargs)
        f = Latent(Eq_NLMP2, irreps_latent, latent_layers, **kwargs)
        dec = Decoder(irreps_latent, irreps_out, model_type, **kwargs)
    elif model_type == 'reservoir':
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_latent = f'{latent_scalars:g}'+'x0e + '+ \
            f'{latent_vectors:g}'+'x1o + '+ \
            f'{latent_tensors:g}'+'x2e'
        enc = Encoder(irreps_in, irreps_latent, model_type, **kwargs)
        f = Latent(Eq_NLMP2, irreps_latent, latent_layers, **kwargs)
        dec = Decoder(irreps_latent, irreps_out, model_type, **kwargs)
        for p in enc.parameters():
            p.requires_grad = False
        for p in f.parameters():
            p.requires_grad = False
        for p in dec.node_dec[0].parameters():
            p.requires_grad = False
    elif model_type == 'non-equivariant':
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Encoder(irreps_in, irreps_latent, model_type, **kwargs)
        f = Latent(nEq_NLMP2, irreps_latent, latent_layers, **kwargs)
        dec = Decoder(irreps_latent, irreps_out, model_type, **kwargs)
    elif model_type == 'non-equivariant isotropic':
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Linear(irreps_in, irreps_latent)
        f = Latent(nEq_NLMP_iso, irreps_latent, latent_layers)
        dec = Linear(irreps_latent, irreps_out)
    elif model_type == 'non-equivariant anisotropic':
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Linear(irreps_in, irreps_latent)
        f = Latent(nEq_NLMP_aniso, irreps_latent, latent_layers)
        dec = Linear(irreps_latent, irreps_out)
    elif model_type == 'GCN':
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Encoder(irreps_in, irreps_latent, model_type)
        f = Latent(GCN, irreps_latent, latent_layers)
        dec = Decoder(irreps_latent, irreps_out, model_type)
    elif model_type == 'Edge':
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Linear(irreps_in, irreps_latent)
        f = Latent(Edge, irreps_latent, latent_layers)
        dec = Linear(irreps_latent, irreps_out)
    else:
        print('Unknown model type: ', model_type)
    return NODE(enc, f, dec, **kwargs)