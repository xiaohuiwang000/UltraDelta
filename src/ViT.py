import os
import time
import gc
import numpy as np
import torch
import logging
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("BASE_DIR:", BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from eval import eval_single_dataset
from src.modeling import ImageEncoder
from args import parse_arguments
from ultradelta_compression import *


def create_log_dir(path, filename='log.txt'):
    """Creates log directory and returns a logger."""
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def create_delta(exam_datasets, pretrained_model_sd, checkpoint_dir, device, enable_layers):
    """Create deltas for each dataset by comparing with pretrained model."""
    deltas = []
    for dataset_name in exam_datasets:
        finetuned_model = torch.load(os.path.join(checkpoint_dir, dataset_name, 'finetuned.pt'), map_location=device, weights_only=False)
        finetuned_model_sd = finetuned_model.state_dict()
        
        delta = {}
        for layer_name, layer_value in finetuned_model_sd.items():
            if any(key in layer_name for key in enable_layers):
                if "weight" in layer_name:
                    delta[layer_name] = finetuned_model_sd[layer_name] - pretrained_model_sd[layer_name]

        deltas.append(delta)
    return deltas


def apply_delta(delta, pretrained_checkpoint, device):
    """Apply delta weights to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in delta:
                continue
            new_state_dict[key] = pretrained_state_dict[key] + delta[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model


def main():
    args = parse_arguments()
    
    model = args.model
    logs_path = os.path.join(args.home, 'logs', model)
    checkpoint_dir = os.path.join(args.home, 'checkpoints', model)
    pretrained_checkpoint = os.path.join(checkpoint_dir, 'zeroshot.pt')
    args.save = checkpoint_dir
    args.data_location = os.path.join(args.home, 'data')

    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(logs_path, 'log_{}_ultradelta.txt'.format(str_time_))

    print("loading deltas....")
    enable_layers = ['attn.in_proj', 'attn.out_proj', 'mlp.c_fc', 'mlp.c_proj']
    pretrained_model = torch.load(pretrained_checkpoint, map_location=args.device, weights_only=False)
    pretrained_model_sd = pretrained_model.state_dict()
    deltas = create_delta(exam_datasets, pretrained_model_sd, checkpoint_dir, args.device, enable_layers)

    print("use_quant: ", args.use_quant)
    if args.use_quant == 'True':
        print("quantizing deltas....")
        for delta in deltas:
            for key in delta:
                delta[key] = quantize_delta(delta[key], args.quant_bit)

    layer_mask_rates_list = assign_sparsity_rates(deltas, args)
    print("assigning done")

    deltas = apply_sparsity_mask(deltas, layer_mask_rates_list, args)
    print("pruning done")

    print("use_trace_norm: ", args.use_trace_norm)
    if args.use_trace_norm == 'True':
        trace_norms = []
        for delta in deltas:
            trace_norm = 0
            for key, val in delta.items():
                if val.dim() > 1:
                    try:
                        U, S, V = torch.svd(val)
                        trace_norm += torch.sum(S).item()
                    except:
                        pass
            trace_norms.append(trace_norm)

        avg_sparsed_trace_norm = sum(trace_norms) / len(trace_norms)
        ratios = []
        for i in range(len(trace_norms)):
            ratio = avg_sparsed_trace_norm / trace_norms[i]
            ratios.append(ratio)
        max_ratio = max(ratios)
        for i in range(len(deltas)):
            ratio_norm = ratios[i] / max_ratio
            print(f"dataset: {exam_datasets[i]}, ratio: {ratio_norm}")
            for key in deltas[i]:
                deltas[i][key] = deltas[i][key] * ratio_norm

    # evaluate
    accs = []
    for i, dataset_name in enumerate(exam_datasets):
        image_encoder = apply_delta(deltas[i], pretrained_checkpoint, args.device)
        metrics = eval_single_dataset(image_encoder, dataset_name, args)
        log.info(str(dataset_name) + ':' + str(metrics.get('top1') * 100) + '%')
        accs.append(metrics.get('top1') * 100)

        del image_encoder
        gc.collect()
        torch.cuda.empty_cache()
    log.info('Avg ACC: ' + str(np.mean(accs)) + '%')


if __name__ == "__main__":
    main()
