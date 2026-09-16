import argparse


def get_args_parameter():
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--config_file', type=str, default="config/AgNews.yaml",
                        help='path to config file')
    parser.add_argument('--encoder_dim', type=list, nargs='+', default=[])

    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--maxtemperature', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--rwr_alpha', type=float, default=0.4,
                        help='restart probability used in the RWR/PPR high-order graph')
    parser.add_argument('--knn_k', type=int, default=10,
                        help='fixed K for the KNN graph; default: 10')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=20, 
                        help='epochs to warmup learning rate')
    parser.add_argument('--train_time', type=int, default=5)

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Initial value of the weight decay. (default: 0)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--minlr', type=float, default=None,
                        help='learning rate (absolute minlr)')

    parser.add_argument('--dataset', type=str, default='AgNews',
                        choices=['AgNews', 'Biomedical', 'SearchSnippets',
                                 'StackOverflow'])
    parser.add_argument('--missing_rate', type=float, default=0)
    parser.add_argument('--data_path', type=str, default='./',
                        help='path to your folder of dataset')
    parser.add_argument('--device', default='cuda', type=str,
                        choices=['cpu', 'cuda'],help='device to use for training / testing')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='path where to save, empty for no saving')

    parser.add_argument('--print_freq', default=50, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=None, type=int)

    return parser
