import argparse
import copy
import os
import sys
import time

import torch
from thop import profile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scipt.train_model import _apply_ablation_preset_from_code, _build_model_by_structure


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def _format_count(value):
    value = float(value)
    units = ['', 'K', 'M', 'G', 'T']
    idx = 0
    while abs(value) >= 1000.0 and idx < len(units) - 1:
        value /= 1000.0
        idx += 1
    return f"{value:.3f}{units[idx]}"


@torch.no_grad()
def benchmark_inference(model, dummy_input, device, warmup_iters=20, measure_iters=100):
    model.eval()
    dummy_input = dummy_input.to(device)

    for _ in range(max(0, warmup_iters)):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(1, measure_iters)):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_samples = dummy_input.size(0) * max(1, measure_iters)
    samples_per_second = total_samples / max(elapsed, 1e-12)
    return samples_per_second, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMAPSS model parameter/FLOP statistics')

    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for FLOPs and throughput measurement input tensor')

    parser.add_argument('--use-spatial-gat', dest='use_spatial_gat', action='store_true', default=True)
    parser.add_argument('--disable-spatial-gat', dest='use_spatial_gat', action='store_false')
    parser.add_argument('--gat-hidden-dim', type=int, default=8)
    parser.add_argument('--gat-out-dim', type=int, default=16)
    parser.add_argument('--gat-num-layers', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--gat-dropout', type=float, default=0.0)
    parser.add_argument('--gat-alpha', type=float, default=0.1)
    parser.add_argument('--gat-embed-dim', type=int, default=8)
    parser.add_argument('--gat-topk', type=int, default=7)
    parser.add_argument('--graph-mode', type=str, default='dynamic_knn', choices=['dynamic_knn', 'path'])

    parser.add_argument('--decoder-attention-size', type=int, default=28)
    parser.add_argument('--use-aef', dest='use_aef', action='store_true', default=True)
    parser.add_argument('--disable-aef', dest='use_aef', action='store_false')
    parser.add_argument('--use-decoder', dest='use_decoder', action='store_true', default=True)
    parser.add_argument('--disable-decoder', dest='use_decoder', action='store_false')

    parser.add_argument('--apply-code-ablation', dest='apply_code_ablation', action='store_true', default=True)
    parser.add_argument('--disable-code-ablation', dest='apply_code_ablation', action='store_false')
    parser.add_argument('--model-code', type=str, default='A',
                        help='A/B/C/D/A_AEF_0FF etc. Used to auto-apply ablation preset.')
    parser.add_argument('--model-structure', type=str, default='encoderdecoder',
                        choices=['encoderdecoder', 'original'])
    parser.add_argument('--lstm-hidden-dim', type=int, default=8)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--warmup-iters', type=int, default=20,
                        help='Warmup iterations before throughput timing')
    parser.add_argument('--measure-iters', type=int, default=100,
                        help='Measured iterations for throughput timing')

    args = parser.parse_args()

    if args.apply_code_ablation:
        _apply_ablation_preset_from_code(args, args.model_code)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = _build_model_by_structure(args).to(device)
    model.eval()

    encoder_x = torch.ones((args.batch_size, args.sequence_len, args.feature_num), device=device)

    # THOP registers forward hooks; use a copy to avoid affecting later benchmark calls.
    profile_model = copy.deepcopy(model).to(device).eval()
    # THOP returns MACs by default. FLOPs ~= 2 * MACs for multiply-add operations.
    macs, _ = profile(profile_model, inputs=(encoder_x,), verbose=False)
    flops_per_batch = 2.0 * float(macs)
    flops_per_sample = flops_per_batch / float(args.batch_size)

    param_info = get_parameter_number(model)

    samples_per_second, elapsed = benchmark_inference(
        model=model,
        dummy_input=encoder_x,
        device=device,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
    )
    flops_per_second = flops_per_sample * samples_per_second

    print('================ Model Complexity Statistics ================')
    print(f"device: {device}")
    print(f"model_structure: {args.model_structure}")
    print(f"model_code: {args.model_code}")
    print(f"input_shape: ({args.batch_size}, {args.sequence_len}, {args.feature_num})")
    print('-------------------------------------------------------------')
    print(
        f"parameters(total/trainable): {param_info['Total']} / {param_info['Trainable']} "
        f"({_format_count(param_info['Total'])} / {_format_count(param_info['Trainable'])})"
    )
    print(f"MACs per forward (batch): {macs:.0f} ({_format_count(macs)})")
    print(f"FLOPs per forward (batch): {flops_per_batch:.0f} ({_format_count(flops_per_batch)})")
    print(f"FLOPs per sample: {flops_per_sample:.0f} ({_format_count(flops_per_sample)})")
    print('-------------------------------------------------------------')
    print(f"throughput: {samples_per_second:.2f} samples/s")
    print(f"estimated FLOPS: {flops_per_second:.2f} ({_format_count(flops_per_second)}FLOPS)")
    print(f"benchmark elapsed: {elapsed:.4f}s for {args.measure_iters} iters")
