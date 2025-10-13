import os
import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Basic paths and environment
    parser.add_argument('--home', type=str, help='Root directory of the project.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers.')

    # Model and data
    parser.add_argument('--model', type=str, default='ViT-B-32', help='Model name.')
    parser.add_argument("--openclip-cachedir", type=str, default=os.path.expanduser('~/.cache/open_clip'), help="Directory for caching models from OpenCLIP.")
    parser.add_argument('--data_location', type=str, default=os.path.expanduser('~/data'), help='Root directory for datasets.')
    parser.add_argument('--eval_datasets', type=lambda x: x.split(','), default=None, help='Comma-separated datasets for evaluation.')
    parser.add_argument('--train_dataset', type=lambda x: x.split(','), default=None, help='Comma-separated datasets for training.')

    # Experiment and results
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for logging.')
    parser.add_argument('--results_db', type=str, default=None, help='Optional results database path.')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0.1, help='Weight decay.')
    parser.add_argument('--ls', type=float, default=0.0, help='Label smoothing.')
    parser.add_argument('--warmup_length', type=int, default=500, help='Warmup steps.')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs.')

    # Classifier and caching
    parser.add_argument('--load', type=lambda x: x.split(','), default=None, help='Optionally load classifiers.')
    parser.add_argument('--save', type=str, default=None, help='Optionally save classifier outputs.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory for caching features and encoder.')

    # Sparsity and quantization settings
    parser.add_argument('--mask_rate', type=float, default=0.5, help='Sparsity rate.')
    parser.add_argument('--use_quant', type=str, default='True', help='Whether to use quantization.')
    parser.add_argument('--quant_bit', type=int, default=16, help='Quantization bit width.')
    parser.add_argument('--use_sparse', type=str, default='True', help='Whether to use sparsity.')
    parser.add_argument('--use_trace_norm', type=str, default='True', help='Whether to use trace norm rescaling.')
    parser.add_argument('--additional_factor', type=float, default=1.0, help='Manually applied additional scaling factor.')
    parser.add_argument('--step_size', type=float, default=0.02, help='Step size for sparsity adjustment.')

    args = parser.parse_args()

    # Auto-select device if not specified
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    # Simplify load input if single entry
    if args.load is not None and len(args.load) == 1:
        args.load = args.load[0]

    return args