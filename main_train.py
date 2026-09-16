import argparse
import datetime
import os
import time
import warnings
from pathlib import Path
import numpy as np
from torchinfo import summary
import torch
import yaml
import model as md 
import utils
from engine_train import train_one_epoch
from dataset_loader import load_dataset
import Paramter as Pm
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)
    device = torch.device(args.device.lower())
    dataset_train, dataset_test = load_dataset(args)
    data_loader_train  = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,  
        pin_memory=False,
        drop_last=True  
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,          
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,                  
    )
    model = md.DIVIDE(
        layer_dims=args.encoder_dim,
        temperature=args.temperature,
        n_classes=args.n_classes,
        drop_rate=args.drop_rate,
        rwr_alpha=args.rwr_alpha,
        knn_k=args.knn_k,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), betas=(0.9, 0.999), weight_decay=float(args.weight_decay))
    if args.train_id == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        state_logger.write('Batch size: {}'.format(args.batch_size))
        state_logger.write('Start time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        state_logger.write(summary(model).__repr__())
        state_logger.write('\n>> Start training {}-th initial, seed: {},'.format(args.train_id, args.seed))

    best_acc = 0.0
    best_state = None
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs
        train_state, loss_value = train_one_epoch(
            model, data_loader_train, data_loader_test,
            optimizer,
            device, epoch,
            state_logger,
            args
        )
        if args.print_this_epoch:
            state_logger.write('Epoch {} K-means: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} F = {:.4f} Loss = {:.4f}'
                               .format(epoch, train_state['acc'], train_state['nmi'], train_state['ari'],
                                       train_state['f'], loss_value))
            if train_state['acc'] > best_acc:
                best_acc = train_state['acc']
                best_state = dict(train_state)
                best_epoch = epoch 
                if args.output_dir:
                    torch.save(model, os.path.join(args.output_dir, "best_checkpoint.pth"))
                    state_logger.write(f'---> New best model saved at epoch {epoch} with ACC: {best_acc:.4f} <---')

    if best_state is None:
        best_state = dict(train_state)
        best_epoch = epoch
    state_logger.write(
        f'Best Run Result [{args.train_id}] at epoch {best_epoch}: '
        f'ACC = {best_state["acc"]:.4f}, NMI = {best_state["nmi"]:.4f}, '
        f'ARI = {best_state["ari"]:.4f}, F = {best_state["f"]:.4f}')
    return best_state


def main(args):
    start_time = time.time()

    result_avr = {'nmi': [], 'ari': [], 'f': [], 'acc': []}
    batch_scale = args.batch_size / 256
    if args.lr is None:  
        args.lr = args.blr * batch_scale
        args.minlr = args.min_blr * batch_scale
    state_logger = utils.FileLogger(os.path.join(args.output_dir, 'log_train.txt'))
    for t in range(args.train_time):
        args.train_id = t
        train_state = train_one_time(args, state_logger)
        args.seed = args.seed + 1
        for k, v in train_state.items():
            result_avr[k].append(v)
    for k, v in result_avr.items():
        x = np.asarray(v) * 100
        result_avr[k] = [x.mean(), x.std()]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    state_logger.write('\nTraining time {}\n'.format(total_time_str))
    state_logger.write('Average Best Result: ACC = {:.2f}({:.2f}) NMI = {:.2f}({:.2f}) ARI = {:.2f}({:.2f}) F = {:.2f}({:.2f})'
                       .format(*result_avr['acc'], *result_avr['nmi'], *result_avr['ari'], *result_avr['f']))


if __name__ == '__main__':
    args = Pm.get_args_parameter()
    args = args.parse_args()  
    config_path = args.config_file
    if not os.path.isfile(config_path):
        raise ValueError(f"Config file does not exist: {config_path}")
    print(f"Loading config from: {config_path}")
    with open(args.config_file,encoding='utf-8') as f:
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.safe_load(f.read())
        else:
            configs = yaml.load(f.read(), Loader=yaml.BaseLoader)
    args = vars(args)
    args.update(configs)
    args = argparse.Namespace(**args)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    folder_name = '_'.join(
        [args.dataset, current_date])
    args.embed_dim = args.encoder_dim[0][-1]
    args.output_dir = os.path.join(args.output_dir, folder_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
