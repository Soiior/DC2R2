import math
import sys
from typing import Iterable

import numpy as np
import torch
from sklearn.cluster import KMeans

import utils
from utils import adjust_learning_config


def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable, data_loader_test: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    state_logger=None,
                    args=None):
    if hasattr(args, 'warmup_epochs') and epoch == args.warmup_epochs:
        model.eval()
        all_features = []
        with torch.no_grad():
            for _, samples, _ in data_loader_test:
                samples = samples.to(device, non_blocking=True)
                feat = model._extract_feature(samples)
                all_features.append(feat)

        all_features = torch.cat(all_features, dim=0)
        all_features = torch.nn.functional.normalize(all_features, dim=-1)
        centers = model.kmeans_clustering(all_features, n_clusters=args.n_classes, seed=args.seed)
        model.cluster_centers.data = centers.clone()
        print(">>> DEC cluster centers initialized.\n")

    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    num_batches = 0

    if hasattr(args, 'epochs') and args.epochs > 0:
        current_temp = args.temperature + (args.maxtemperature - args.temperature) * (epoch / args.epochs)
        if hasattr(model, 'cl'):
            model.cl.temperature = current_temp

    for samples, original_data in data_loader_train:
        adjust_learning_config(optimizer, epoch, args)
        mmt = args.momentum
        if len(samples) != 2:
            raise ValueError(f"Expected two masked views, got {len(samples)}")
        samples = [sample.to(device, non_blocking=True) for sample in samples]

        original_data = original_data.to(device, non_blocking=True)
        
        max_k = min(20, args.batch_size // args.n_classes)
        growth_epochs = max_k - 3
        dynamic_weight = utils.get_dynamic_weight(
            epoch, max_weight=0.1, warmup_epochs=args.warmup_epochs,
            dec_warmup_len=growth_epochs)
        autocast_enabled = device.type == 'cuda'
        with torch.autocast(device_type=device.type, enabled=autocast_enabled,
                            dtype=torch.float16):
            loss = model(
                samples,
                original_data=original_data,
                momentum=mmt,
                warm_up=(args.warmup_epochs > 0 and epoch >= args.warmup_epochs),
                dynamic_weight=dynamic_weight,
            )
            
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss_value
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    lr_current = optimizer.param_groups[0]['lr']

    if args.print_this_epoch:
        eval_result = evaluate(model, data_loader_test, device, args)
        if eval_result:
            log_str = f"Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {avg_loss:.4f} | LR: {lr_current:.2e} | ACC: {eval_result['acc']*100:.2f}% | NMI: {eval_result['nmi']*100:.2f}% | ARI: {eval_result['ari']*100:.2f}% | F-Score: {eval_result['f']*100:.2f}%"
            if state_logger is not None:
                state_logger.write(log_str)
            else:
                print(log_str)
    else:
        eval_result = None
        
    return eval_result, avg_loss

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader_test: Iterable,
             device: torch.device, args=None):
    model.eval()
    extracter = model._extract_feature
    
    with torch.no_grad():
        args.n_sample = len(data_loader_test.dataset)
        actual_embed_dim = getattr(model, 'feature_dim', args.embed_dim)
        if actual_embed_dim != args.embed_dim:
            print(f">>> [Auto-Adapt] Adjusted embed_dim from {args.embed_dim} to {actual_embed_dim} inside evaluate().")
            args.embed_dim = actual_embed_dim
            
        actual_n_classes = getattr(model, 'n_classes', args.n_classes)
        if actual_n_classes != args.n_classes:
            print(f">>> [Auto-Adapt] Adjusted n_classes from {args.n_classes} to {actual_n_classes} inside evaluate().")
            args.n_classes = actual_n_classes
        features_all = torch.zeros(args.n_sample, args.embed_dim).to(device)
        labels_all = torch.zeros(args.n_sample, dtype=torch.long).to(device)
        
        for ids, samples, labels in data_loader_test:
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)               
            features = extracter(samples)
            features_all[ids] = features
            labels_all[ids] = labels

    features_cat = torch.nn.functional.normalize(features_all, dim=-1)

    if torch.isnan(features_cat).any():
        print("Warning: Features contain NaN!")
        features_cat = torch.nan_to_num(features_cat)

    features_np = features_cat.cpu().numpy()
    
    kmeans_model = KMeans(n_clusters=args.n_classes, n_init=1, random_state=args.seed).fit(features_np)
    kmeans_label = kmeans_model.labels_

    nmi, ari, f, acc = utils.evaluate(np.asarray(labels_all.cpu()), kmeans_label, n_cluster=int(args.n_classes))
    result = {'nmi': nmi, 'ari': ari, 'f': f, 'acc': acc}

    return result
