import torch
import logging
import numpy as np
from tqdm import tqdm
import faiss
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from os.path import join
import csv
from time import time
from thop import profile
import copy

def test(args, eval_ds, model, pca=None, output_folder=None, eval=False):
    # print(torch.cuda.memory_allocated()/(1024**2))
    # print("====================================>>")
    model = model.eval()
    outputdim = model.meta['outputdim']
    # seq_len = args.seq_length
    # n_gpus = args.n_gpus

    query_num = eval_ds.queries_num
    gallery_num = eval_ds.database_num
    all_features = np.empty((query_num + gallery_num, outputdim), dtype=np.float32)

    with torch.no_grad():
        logging.debug("Extracting gallery features for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=8,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        # print(torch.cuda.memory_allocated()/(1024**2))
        # print("====================================>>")
        for images, indices, _ in tqdm(database_dataloader, ncols=100):
            # if args.arch != "crossvpr4":
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])
            # else:
            #     images = images.permute(0,2,1,3,4)
            # if (images.shape[0] % (seq_len * n_gpus) != 0) and n_gpus > 1:
            #     # handle last batch, if it is has less than batch_size sequences
            #     model.module = model.module.to('cuda:1')
            #     # shape[0] is always a multiple of seq_length, sequences are always full size
            #     for sequence in range(images.shape[0] // seq_len):
            #         n_seq = sequence * seq_len
            #         seq_images = images[n_seq: n_seq + seq_len].to('cuda:1')
            #         features = model.module(seq_images).cpu().numpy()
            #         if pca:
            #             features = pca.transform(features)
            #         all_features[indices.numpy()[sequence], :] = features

            #     model = model.cuda()
            # else:
            features = model(images.to(args.device))
            features = features.cpu().numpy()
            if pca:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features

        logging.debug("Extracting queries features for evaluation/testing")
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=8,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        # FLAG = False
        # print(torch.cuda.memory_allocated()/(1024**2))
        # print("====================================>>")
        for images, _, indices in tqdm(queries_dataloader, ncols=100):
            # if args.arch != "crossvpr4":
            images = images.contiguous().view(-1, 3, args.img_shape[0], args.img_shape[1])
            # else:
            #     images = images.contiguous().permute(0,2,1,3,4)
            # if (images.shape[0] % (seq_len * n_gpus) != 0) and n_gpus > 1:
            #     # handle last batch, if it is has less than batch_size sequences
            #     model.module = model.module.to('cuda:1')
            #     # shape[0] is always a multiple of seq_length, sequences are always full size
            #     for sequence in range(images.shape[0] // seq_len):
            #         n_seq = sequence * seq_len
            #         seq_images = images[n_seq: n_seq + seq_len].to('cuda:1')
            #         features = model.module(seq_images).cpu().numpy()
            #         if pca:
            #             features = pca.transform(features)
            #         all_features[indices.numpy()[sequence], :] = features

            #     model = model.cuda()
            # else:
            # memory_before = torch.cuda.memory_allocated()
            images = images.to(args.device)
            features = model(images)
            # memory_after = torch.cuda.memory_allocated()
            # memory_usage = (memory_after - memory_before) / (1024 ** 2)  # in MB
            # print("Memory usage: ", memory_usage, " MB")
            # print("memory_before: {} MB , memory_after: {} MB. ".format(str(memory_before/(1024**2)), str(memory_after/(1024**2))) )
            # if FLAG:
            #     input = images.to(args.device)
            #     m = copy.deepcopy(model.module.to('cuda:0'))
            #     flops, params = profile(m, inputs=(input, ))
            #     print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
            #     print("params=", str(params/1e6)+'{}'.format("M"))
            #     FLAG = False
            features = features.cpu().numpy()
            if pca:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features

    # logging.info(f"GPU memory is {torch.cuda.memory_allocated() / 1024 / 1024} MB.")
    # logging.info(f"Pytorch memory is {torch.cuda.memory_reserved() / 1024 / 1024 / 1024} GB.")
    # torch.cuda.empty_cache()
    queries_features = all_features[eval_ds.database_num:]
    gallery_features = all_features[:eval_ds.database_num]

    time1 = time()
    faiss_index = faiss.IndexFlatL2(outputdim)
    faiss_index.add(gallery_features)

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_features, 20)
    print("====>> match time {} s/sequence".format(str((time()-time1)/len(queries_features))))
    
    #=====save
    if eval:
        # valid_q = eval_ds.valid_q
        np.savetxt(join(output_folder,'query.csv'), np.array(eval_ds.valid_q), delimiter='\n', fmt='%s')
        predictions_save = np.empty(len(eval_ds.valid_q)*20, dtype=object)
        database = np.array(eval_ds.db_paths)
        for i, pre in enumerate(predictions):
            predictions_save[i*20:(i+1)*20] = database[pre]
        np.savetxt(join(output_folder,'pre.csv'), predictions_save, delimiter='\n', fmt='%s')
    #=====
    
    # For each query, check if the predictions are correct
    positives_per_query = eval_ds.pIdx
    recall_values = [1, 5, 10, 20]  # recall@1, recall@5, recall@10
    recalls = np.zeros(len(recall_values))
    #===
    if eval:   
        recalls_save = np.zeros((len(eval_ds.valid_q),len(recall_values)),dtype=int)
        with open(join(output_folder,'gt.csv'),'a',newline='') as f:
            writer = csv.writer(f)
            for i, gt in enumerate(positives_per_query,start=1):
                writer.writerow([str(i)])
                for j in gt:
                    writer.writerow([database[j]])
    #===
    
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                #===save
                if eval:
                    recalls_save[query_index][i:] += 1
                #===
                break
            
    #======
    if eval:
        np.savetxt(join(output_folder,'recall.csv'), recalls_save, fmt="%d")
    #======
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / len(eval_ds.qIdx) * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.3f}" for val, rec in zip(recall_values, recalls)])
    return recalls, recalls_str
