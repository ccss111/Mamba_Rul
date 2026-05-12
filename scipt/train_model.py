import datetime
import argparse
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import *
from model import *
from utils import *
# 自动创建 logs 文件夹（不存在就新建）
default_log_dir = "/CMAPSS-release/logs"
if not os.path.exists(default_log_dir):
    os.makedirs(default_log_dir)


def _build_log_name(model_code, lr, embed_dim, topk, time_tag):
    model_code = str(model_code).strip()
    if not model_code:
        raise ValueError("--model-code cannot be empty.")
    lr_str = format(float(lr), "g")
    return (
        f"{model_code}_学习率+{lr_str}_嵌入模型维度embed+{embed_dim}_"
        f"topk+{topk}_{time_tag}.log"
    )

if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory
    parser = argparse.ArgumentParser(description='Cmapss Dataset With Pytorch')
    parent_dir = "/CMAPSS-release"
    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--dataset-root', type=str,
                        default=parent_dir + '/CMAPSSData/',
                        help='The dir of CMAPSS dataset1')
    parser.add_argument('--sub-dataset', type=str, default='FD004', help='FD001/2/3/4')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['cosine', 'step'],
                        help='learning rate scheduler type')
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
    parser.add_argument('--eta-min', type=float, default=1e-5, help='minimum lr for cosine annealing')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--use-exponential-smoothing', default=True)  
    parser.add_argument('--smooth-rate', type=int, default=40)
    parser.add_argument('--use-spatial-gat', dest='use_spatial_gat', action='store_true', default=True,
                        help='Enable sensor graph attention before LSTM encoder')
    parser.add_argument('--disable-spatial-gat', dest='use_spatial_gat', action='store_false',
                        help='Disable sensor graph attention and use raw sensor sequence')
    parser.add_argument('--gat-hidden-dim', type=int, default=8,
                        help='Hidden output size of first GAT layer')
    parser.add_argument('--gat-out-dim', type=int, default=16,
                        help='Output embedding size of second GAT layer')
    parser.add_argument('--gat-dropout', type=float, default=0.0,
                        help='Dropout used inside graph attention layers')
    parser.add_argument('--gat-alpha', type=float, default=0.1,
                        help='Negative slope for graph attention LeakyReLU')
    parser.add_argument('--gat-embed-dim', type=int, default=16,
                        help='Sensor node embedding size used to build cosine-similarity graph')
    parser.add_argument('--gat-topk', type=int, default=5,
                        help='Top-k neighbors per sensor selected by cosine similarity')
    parser.add_argument('--graph-mode', type=str, default='dynamic_knn',
                        choices=['dynamic_knn', 'static_knn', 'static_full'],
                        help='Graph construction mode for spatial GAT')
    parser.add_argument('--feature-attention-size', type=int, default=4,
                        help='Attention size for self-concat feature attention')
    parser.add_argument('--decoder-fusion', type=str, default='concat', choices=['concat', 'gate'],
                        help='Fusion mode for AEF/AOF in decoder: concat or learned gate')
    parser.add_argument('--use-temporal-attention', dest='use_temporal_attention', action='store_true', default=True,
                        help='Enable time-step attention over encoder outputs')
    parser.add_argument('--disable-temporal-attention', dest='use_temporal_attention', action='store_false',
                        help='Disable time-step attention and fallback to additive attention')
    parser.add_argument('--temporal-attention-size', type=int, default=28,
                        help='Hidden size for temporal attention score network')
    parser.add_argument('--temporal-attention-dropout', type=float, default=0.0,
                        help='Dropout ratio used in temporal attention scoring')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', dest='save_model', action='store_true', default=True,
                        help='save trained models')
    parser.add_argument('--no-save-model', dest='save_model', action='store_false',
                        help='do not save trained models')
    parser.add_argument('--model-code', type=str, default='A',
                        help='模型代号，用于日志命名，如 A/B/C/D/E/F')
    args = parser.parse_args()

    num = 999#28 23 1
    torch.manual_seed(num)
    #2 17 27 30 33 51 62 80 88 97 


    train_loader, valid_loader, test_loader, test_loader_last, \
        num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path=args.dataset_root,
            sub_dataset=args.sub_dataset,
            max_rul=args.max_rul,
            seq_length=args.sequence_len,
            batch_size=args.batch_size,
            use_exponential_smoothing=args.use_exponential_smoothing,
            smooth_rate=args.smooth_rate)

    encoder_input_size = args.gat_out_dim if args.use_spatial_gat else args.feature_num
    encoder = Seq2SeqEncoder(input_size=encoder_input_size, num_layers=2, num_hidden=8)
    decoder = Seq2SeqDecoder(input_size=encoder_input_size, num_layers=2, num_hidden=8,
                             seq_len=args.sequence_len,
                             attention_size=args.temporal_attention_size,
                             fusion_mode=args.decoder_fusion,
                             use_temporal_attention=args.use_temporal_attention,
                             temporal_attention_dropout=args.temporal_attention_dropout)
    model = EncoderDecoder(encoder=encoder, decoder=decoder,
                           feature_attention_size=args.feature_attention_size,
                           use_spatial_gat=args.use_spatial_gat,
                           graph_mode=args.graph_mode,
                           num_sensors=args.feature_num,
                           gat_hidden_dim=args.gat_hidden_dim,
                           gat_out_dim=args.gat_out_dim,
                           gat_embed_dim=args.gat_embed_dim,
                           gat_topk=args.gat_topk,
                           gat_dropout=args.gat_dropout,
                           gat_alpha=args.gat_alpha)

    model_type = type(model).__name__

    criterion_train = torch.nn.MSELoss()
    criterion_eval = RMSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.max_epochs,
            eta_min=args.eta_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )

    time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs(default_log_dir, exist_ok=True)
    log_file_name = _build_log_name(
        model_code=args.model_code,
        lr=args.lr,
        embed_dim=args.gat_embed_dim,
        topk=args.gat_topk,
        time_tag=time,
    )
    log_path = os.path.join(default_log_dir, log_file_name)

    with open(log_path, "a", encoding="utf-8") as f:

        f.write("-----"+ args.sub_dataset + "-----\n")
        f.write("关键参数:\n")
        f.write(f"序列长度: {args.sequence_len}\n")
        f.write(f"批大小(batch_size): {args.batch_size}\n")
        f.write(f"最大训练轮数(max_epochs): {args.max_epochs}\n")
        f.write(f"随机数种子: {num}\n")
        f.write(f"模型代号(model_code): {args.model_code}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"学习率调度器: {args.lr_scheduler}\n")
        if args.lr_scheduler == 'cosine':
            f.write(f"余弦退火最小学习率(eta_min): {args.eta_min}\n")
        else:
            f.write(f"Step步长(step_size): {args.step_size}\n")
            f.write(f"Step衰减系数(gamma): {args.gamma}\n")
        
        f.write(f"空间GAT: {'开启' if args.use_spatial_gat else '关闭'}\n")
        f.write(f"时间注意力: {'开启' if args.use_temporal_attention else '关闭'}\n")
        if args.use_temporal_attention:
            f.write(f"时间注意力隐层维度: {args.temporal_attention_size}\n")
            f.write(f"时间注意力dropout: {args.temporal_attention_dropout}\n")
        f.write(f"Decoder融合方式: {args.decoder_fusion}\n")
        if args.use_spatial_gat:
            f.write(f"图构建模式(graph_mode): {args.graph_mode}\n")
            f.write(f"GAT邻居数(topk): {args.gat_topk}\n")
            f.write(f"GAT嵌入维度: {args.gat_embed_dim}\n")
        f.write("------------------------------\n")


    train(
        model, train_loader, valid_loader,
        test_loader, args.max_epochs, optimizer,
        scheduler, criterion_train, criterion_eval,
        lines_list=[], patience=args.patience, max_rul=args.max_rul, num_test_windows=num_test_windows,
        device=torch.device('cuda') if not args.no_cuda else torch.device('cpu'), time=time, log_path=log_path,
        checkpoint_dir=os.path.join(parent_dir, 'trials'),
        checkpoint_prefix='model_' + args.sub_dataset + '_' + time)

    model_output_dir = os.path.join(parent_dir, 'trials')
    os.makedirs(model_output_dir, exist_ok=True)

    saved_model_path = os.path.join(model_output_dir, 'model_' + args.sub_dataset + '_' + time + '.pkl')

    if args.save_model:
        torch.save(model, saved_model_path)
    else:
        # train() 会在验证最优时写入 best_*.pkl
        saved_model_path = os.path.join(model_output_dir, 'best_model_' + args.sub_dataset + '_' + time + '.pkl')

    auto_test_cmd = [
        sys.executable,
        os.path.join(parent_dir, 'scipt', 'test_model.py'),
        '--sub-dataset', args.sub_dataset,
        '--smooth-rate', str(args.smooth_rate),
        '--sequence-len', str(args.sequence_len),
        '--feature-num', str(args.feature_num),
        '--dataset-root', args.dataset_root,
        '--max-rul', str(args.max_rul),
        '--batch-size', str(args.batch_size),
        '--model-path', saved_model_path,
    ]
    if args.no_cuda:
        auto_test_cmd.append('--no-cuda')

    test_result = subprocess.run(
        auto_test_cmd,
        cwd=parent_dir,
        capture_output=True,
        text=True,
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("[Auto Test] command: " + " ".join(auto_test_cmd) + "\n")
        if test_result.stdout:
            f.write(test_result.stdout.strip() + "\n")
        if test_result.returncode != 0:
            f.write("[Auto Test][ERROR] return_code={}\n".format(test_result.returncode))
            if test_result.stderr:
                f.write(test_result.stderr.strip() + "\n")

    if test_result.returncode == 0:
        print("[Auto Test] completed. Result appended to log file.")
        if test_result.stdout:
            print(test_result.stdout.strip())
    else:
        print("[Auto Test][ERROR] Failed to run test_model.py. See log for details.")
