import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from collections import OrderedDict
import logging
from datetime import datetime
import einops
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import random
import argparse
# our imports
# from init_scripts import init_script
# init_script()
from svpr.datasets import BaseDataset, PCADataset
from svpr.evals import test
from svpr.models.net import Net
from svpr.utils import configure_transform, setup_logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Spatio-Temporal VPR",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_city", type=str, default='', help='subset of cities from test set')
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Number of images in each sequence")
    parser.add_argument("--val_posDistThr", type=int, default=10, help="_")
    parser.add_argument('--img_shape', type=int, default=[384, 384], nargs=2,
                        help="Resizing shape for images (HxW).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--arch", type=str, default="seqvlad", choices=['seqvlad', 'stformer'])
    parser.add_argument("--trunc_te", type=int, default=4, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=-1, choices=list(range(-1, 14)))
    parser.add_argument("--trunc_te_tatt", type=int, default=4, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te_tatt", type=int, default=-1, choices=list(range(-1, 14)))
    parser.add_argument('--clusters', type=int, default=64)
    parser.add_argument("--resume", type=str, default="checkpoints/seqvlad/msls.pth",
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--pca_outdim", type=int, help='output size with PCA', default=None)
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--dataset_path", type=str, default="", help="Path of the dataset")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the test result")
    parser.add_argument('--rel_pos_temporal', action='store_true', default=False)
    parser.add_argument('--rel_pos_spatial', action='store_true', default=False)
    parser.add_argument('--abs_pos_embed', action='store_true', default=False)
    parser.add_argument('--part', type=str, default=None, choices=['only_spatial','only_temporal'])
    args = parser.parse_args()
    return args

def compute_pca(args, model, transform, full_features_dim):
    model = model.eval()
    pca_ds = PCADataset(dataset_folder=args.dataset_path, split='train',
                        base_transform=transform, seq_len=args.seq_length)
    logging.info(f'PCA dataset: {pca_ds}')
    num_images = min(len(pca_ds), 2 ** 14)
    if num_images < len(pca_ds):
        idxs = random.sample(range(0, len(pca_ds)), k=num_images)
    else:
        idxs = list(range(len(pca_ds)))
    subset_ds = Subset(pca_ds, idxs)
    dl = torch.utils.data.DataLoader(subset_ds, args.infer_batch_size)

    pca_features = np.empty([num_images, full_features_dim])
    with torch.no_grad():
        for i, sequences in enumerate(tqdm(dl, ncols=100, desc="Database sequence descriptors for PCA: ")):
            if len(sequences.shape) == 5:
                sequences = einops.rearrange(sequences, "b s c h w -> (b s) c h w")
            features = model(sequences).cpu().numpy()
            pca_features[i * args.infer_batch_size : (i * args.infer_batch_size ) + len(features)] = features
    pca = PCA(args.pca_outdim)
    logging.info(f'Fitting PCA from {full_features_dim} to {args.pca_outdim}...')
    pca.fit(pca_features)
    return pca


def evaluation():
    args = parse_arguments()
    start_time = datetime.now()
    args.output_folder = f"test/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    setup_logging(args.output_folder, console="info")
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")
    model = Net(args)

    if args.resume:
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict)

    model = model.to(args.device)
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img_shape = (args.img_shape[0], args.img_shape[1])
    transform = configure_transform(image_dim=img_shape, meta=meta)

    eval_ds = BaseDataset(cities=args.test_city, dataset_folder=args.dataset_path, split='test',
                          base_transform=transform, seq_len=args.seq_length,
                          pos_thresh=args.val_posDistThr, reverse_frames=False)
    logging.info(f"Test set: {eval_ds}")

    if args.pca_outdim:
        full_features_dim = args.features_dim
        args.features_dim = args.pca_outdim
        pca = compute_pca(args, model, transform, full_features_dim)
        model.module.meta['outputdim'] = args.pca_outdim
    else:
        pca = None

    logging.info(f"Output dimension of the model is {model.meta['outputdim']}")
    recalls, recalls_str = test(args, eval_ds, model, pca=pca, output_folder=args.output_folder, eval=False)
    logging.info(f"Recalls on test set: {recalls_str}")
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

if __name__ == "__main__":
    evaluation()
