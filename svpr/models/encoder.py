import torch
import torch.nn as nn
import logging
from svpr.models.cct import cct_14_7x2_384

class STEcoder(nn.Module):
    def __init__(self,layer_s=None,layer_t=None,freeze_te=None,freeze_te_tatt=None,*args, **kwargs) -> None:
        super().__init__()
        self.trunc_te = layer_s
        self.trunc_te_t = layer_t
        self.freeze_te = freeze_te
        self.freeze_te_t = freeze_te_tatt
        # self.seqlen = 5
        self.is_inference=False
        self.rel_pos_temporal = kwargs.get('rel_pos_temporal', False)
        self.rel_pos_spatial = kwargs.get('rel_pos_spatial', False)
        
        self.spatial = cct_14_7x2_384(pretrained=True, progress=True, use_all_tokens=False, rel_pos_spatial=self.rel_pos_spatial, abs_pos_embed=kwargs.get('abs_pos_embed', False))
        if self.trunc_te:
            logging.debug(f"Truncate CCT at spatial transformers encoder {self.trunc_te}")
            self.spatial.classifier.blocks = torch.nn.ModuleList(self.spatial.classifier.blocks[:self.trunc_te].children())
        if self.freeze_te:
            logging.debug(f"Freeze all the layers up to spatial tranformer encoder {self.freeze_te}")
            for p in self.spatial.parameters():
                p.requires_grad = False
            for name, child in self.spatial.classifier.blocks.named_children():#name from 0 to args.trunc_te-1
                if int(name) > self.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        
        
        self.temporal = cct_14_7x2_384(pretrained=True, progress=True, use_all_token=False, use_for_t=True, rel_pos_temporal=self.rel_pos_temporal, abs_pos_embed=kwargs.get('abs_pos_embed', False)).classifier#对应到区域时间注意力
        
        if self.trunc_te_t:
            logging.debug(f"Truncate CCT at temporal transformers encoder {self.trunc_te_t}")
            self.temporal.blocks = torch.nn.ModuleList(self.temporal.blocks[:self.trunc_te_t].children())
        if self.freeze_te_t:
            logging.debug(f"Freeze all the layers up to temporal tranformer encoder {self.freeze_te_t}")
            for p in self.temporal.parameters():
                p.requires_grad = False
            for name, child in self.temporal.blocks.named_children():#name from 0 to args.trunc_te-1
                if int(name) > self.freeze_te_t:#if freeze_te=1 freeze layer1h&layer2
                    for params in child.parameters():
                        params.requires_grad = True
        
        
    def forward(self, x):
        patches = self.spatial.tokenizer(x)
        spatial_x = self.spatial.classifier(patches)
        
        temporal_x = self.temporal(patches)
        
        return spatial_x, temporal_x
        
    def forward_seqvlad(self, x):
        patches = self.spatial.tokenizer(x)       
        spatial_x = self.spatial.classifier(patches)
        return spatial_x
        
def SeqVladModel(args):  
    encoder = cct_14_7x2_384(pretrained=True, progress=True, use_all_tokens=False, rel_pos_spatial=args.rel_pos_spatial, abs_pos_embed=args.abs_pos_embed)   
    if args.trunc_te:
        logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
        encoder.classifier.blocks = torch.nn.ModuleList(encoder.classifier.blocks[:args.trunc_te].children())
    if args.freeze_te:
        logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
        for p in encoder.parameters():
            p.requires_grad = False
        for name, child in encoder.classifier.blocks.named_children():#name from 0 to args.trunc_te-1
            if int(name) > args.freeze_te:
                for params in child.parameters():
                    params.requires_grad = True
    args.features_dim = 384
    return encoder