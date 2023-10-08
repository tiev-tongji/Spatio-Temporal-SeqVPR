import torch
from torch import nn
from svpr.models import pooling
from svpr.models.encoder import STEcoder, SeqVladModel

class Net(nn.Module):
    """Network for VG applied to sequences.The used networks are composed of 
    an encoder, a pooling layer and an aggregator."""
    def __init__(self, args):
        super().__init__()
        self.encoder = get_encoder(args)
        # args.features_dim = 384
        self.aggregator = get_aggregator(args)
        # args.features_dim *= args.clusters
        self.meta = {'outputdim': args.features_dim}
        self.args = args
        
    def forward(self, x):
        if self.args.arch == "seqvlad":
            x = self.encoder(x)
        elif self.args.arch == "stformer":
            spatial, temporal = self.encoder(x)
            if self.args.part == 'only_spatial':
                x = spatial
            elif self.args.part == 'only_temporal':
                x = temporal
            else:
                x = spatial+temporal
        x = self.aggregator(x) 
        return x

def get_encoder(args):
    if args.arch == "seqvlad":
        encoder = SeqVladModel(args)
    elif args.arch == "stformer":
        encoder = STEcoder(layer_s=args.trunc_te, layer_t=args.trunc_te_tatt, 
                    freeze_te=args.freeze_te, freeze_te_tatt=args.freeze_te_tatt, 
                    rel_pos_temporal=args.rel_pos_temporal, rel_pos_spatial=args.rel_pos_spatial)
        args.features_dim = 384
    return encoder

def get_aggregator(args):
    aggregator = pooling.SeqVLAD(seq_length=args.seq_length, dim=args.features_dim, clusters_num=args.clusters)
    args.features_dim *= args.clusters
    return aggregator
        

def get_output_channels_dim(model,img_size):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, img_size[0], img_size[1]])).shape[1]

def get_output_tensor_dim(model,img_size):
    """Return the tensor shape in the output of a model."""
    return model(torch.ones([1, 3, img_size[0], img_size[1]])).shape[1:]
