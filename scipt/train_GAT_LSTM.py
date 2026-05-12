import argparse
import datetime
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import *
from model import *
from utils import *

DEFAULT_LOG_DIR = "/CMAPSS-release/logs_GAT_LSTM"
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)


def _parse_int_list(raw_value, arg_name):
    values = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{arg_name} contains non-integer value: {token}") from exc
        if value <= 0:
            raise ValueError(f"{arg_name} expects positive integers, got: {value}")
        values.append(value)

    if not values:
        raise ValueError(f"{arg_name} cannot be empty")
    return values


def _build_log_name(sub_dataset, model_code, lr, gat_hidden_dims, lstm_hidden_dims, fusion_mode,
                    graph_mode, embed_dim, topk, use_aef, use_aof, time_tag):
    model_code = str(model_code).strip()
    if not model_code:
        raise ValueError("--model-code cannot be empty")
    lr_str = format(float(lr), "g")
    gat_tag = "-".join(str(v) for v in gat_hidden_dims)
    lstm_tag = "-".join(str(v) for v in lstm_hidden_dims)
    attention_tag = f"AEF{int(bool(use_aef))}_AOF{int(bool(use_aof))}"
    return (
        f"{sub_dataset}_{model_code}_lr+{lr_str}_GAT_hdim+{gat_tag}_"
        f"LSTM_hdim+{lstm_tag}_+{graph_mode}_embed_dim+{embed_dim}_"
        f"Topk+{topk}_+{fusion_mode}_{attention_tag}_{time_tag}.log"
    )


def _translate_auto_test_output(text):
    translated = str(text)
    translated = translated.replace("model_path:", "模型路径:")
    translated = translated.replace("rmse_final:", "最终RMSE:")
    translated = translated.replace("score:", "评分:")
    return translated


def _resolve_smooth_rate(sub_dataset, smooth_rate):
    if smooth_rate is not None:
        return int(smooth_rate)

    dataset_key = str(sub_dataset).strip().upper()
    if dataset_key in ("FD001", "FD003"):
        return 30
    if dataset_key in ("FD002", "FD004"):
        return 40
    raise ValueError(f"不支持的子数据集: {sub_dataset}，无法自动选择 smooth-rate")


if __name__ == "__main__":
    parent_dir = "/CMAPSS-release"

    parser = argparse.ArgumentParser(
        description="在 CMAPSS 上训练 GAT_LSTM"
    )
    parser.add_argument("--sequence-len", type=int, default=30)
    parser.add_argument("--feature-num", type=int, default=14)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=parent_dir + "/CMAPSSData/",
        help="CMAPSS 数据集目录",
    )
    parser.add_argument("--sub-dataset", type=str, default="FD001", help="FD001/FD002/FD003/FD004")
    parser.add_argument("--max-rul", type=int, default=125, help="分段 RUL 截断上限")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="step",
        choices=["step"],
        help="学习率调度器类型（仅支持step）",
    )
    parser.add_argument("--step-size", type=int, default=10, help="StepLR 衰减步长")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR 衰减系数")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8, help="早停耐心轮数")
    parser.add_argument("--max-epochs", type=int, default=30)

    parser.add_argument(
        "--use-exponential-smoothing",
        dest="use_exponential_smoothing",
        action="store_true",
        default=True,
        help="预处理中启用指数平滑",
    )
    parser.add_argument(
        "--disable-exponential-smoothing",
        dest="use_exponential_smoothing",
        action="store_false",
        help="预处理中关闭指数平滑",
    )
    parser.add_argument(
        "--smooth-rate",
        type=int,
        default=None,
        help="指数平滑系数；默认自动选择：FD001/FD003=30，FD002/FD004=40",
    )

    parser.add_argument(
        "--gat-hidden-dims",
        type=str,
        default="8,8,8",
        help="GAT 隐藏维度，逗号分隔",
    )
    parser.add_argument(
        "--lstm-hidden-dims",
        type=str,
        default="8,8",
        help="LSTM 隐藏维度，逗号分隔",
    )
    parser.add_argument("--gat-dropout", type=float, default=0.1, help="GAT 层中的 Dropout")
    parser.add_argument("--gat-alpha", type=float, default=0.1, help="GAT 中 LeakyReLU 的 alpha")
    parser.add_argument(
        "--graph-mode",
        type=str,
        default="dynamic_topk",
        choices=["dynamic_topk", "path"],
        help="时序图构建方式：动态Topk图/路径图",
    )
    parser.add_argument("--gat-embed-dim", type=int, default=8, help="动态Topk图的节点嵌入维度")
    parser.add_argument("--gat-topk", type=int, default=7, help="动态Topk图的邻居数量")

    parser.add_argument(
        "--feature-attention-size",
        type=int,
        default=4,
        help="AOF 拼接自注意力的隐藏维度",
    )
    parser.add_argument(
        "--decoder-attention-size",
        type=int,
        default=28,
        help="AEF 加性注意力的隐藏维度",
    )
    parser.add_argument(
        "--decoder-fusion",
        type=str,
        default="concat",
        choices=["concat", "gate"],
        help="AEF/AOF 融合方式",
    )
    
    parser.add_argument("--seeds", type=str, default="2,17,27,30,33,51,62,80,88,97")#2,
    parser.add_argument("--no-cuda", action="store_true", default=False, help="禁用 CUDA 训练")

    parser.add_argument("--save-model",dest="save_model",action="store_true",default=True,help="保存训练后的模型",)
    parser.add_argument("--no-save-model",dest="save_model",action="store_false",help="不额外保存最终模型，自动测试使用 best_ checkpoint",)
    parser.add_argument("--model-code",type=str,default="GAT_LSTM_AEFAOF",help="模型代号，用于日志命名",)

    args = parser.parse_args()

    model_code = str(args.model_code).strip()
    if not model_code:
        raise ValueError("--model-code cannot be empty")

    gat_hidden_dims = _parse_int_list(args.gat_hidden_dims, "--gat-hidden-dims")
    lstm_hidden_dims = _parse_int_list(args.lstm_hidden_dims, "--lstm-hidden-dims")
    seed_list = _parse_int_list(args.seeds, "--seeds")
    smooth_rate = _resolve_smooth_rate(args.sub_dataset, args.smooth_rate)

    model_output_root = os.path.join(DEFAULT_LOG_DIR, f"{args.sub_dataset}_{model_code}")
    os.makedirs(model_output_root, exist_ok=True)

    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")

    for seed in seed_list:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        seed_output_dir = os.path.join(model_output_root, f"seed{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)

        train_loader, valid_loader, test_loader, test_loader_last, num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path=args.dataset_root,
            sub_dataset=args.sub_dataset,
            max_rul=args.max_rul,
            seq_length=args.sequence_len,
            batch_size=args.batch_size,
            use_exponential_smoothing=args.use_exponential_smoothing,
            smooth_rate=smooth_rate,
        )

        model = GAT_LSTM_model(
            num_patch=args.sequence_len,
            patch_size=args.feature_num,
            hidden_dim=gat_hidden_dims,
            lstm_hidden_dim=lstm_hidden_dims,
            graph_mode=args.graph_mode,
            embed_dim=args.gat_embed_dim,
            topk=args.gat_topk,
            dropout=args.gat_dropout,
            alpha=args.gat_alpha,
            return_attention=True,
        )

        criterion_train = torch.nn.MSELoss()
        criterion_eval = RMSELoss()
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )

        now_tag = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_tag = f"{now_tag}_seed{seed}"

        log_file_name = _build_log_name(
            sub_dataset=args.sub_dataset,
            model_code=model_code,
            lr=args.lr,
            gat_hidden_dims=gat_hidden_dims,
            lstm_hidden_dims=lstm_hidden_dims,
            fusion_mode=args.decoder_fusion,
            graph_mode=args.graph_mode,
            embed_dim=args.gat_embed_dim,
            topk=args.gat_topk,
            use_aef=False,
            use_aof=False,
            time_tag=run_tag,
        )
        log_path = os.path.join(seed_output_dir, log_file_name)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("-----" + args.sub_dataset + "-----\n")
            f.write("模型: GAT_LSTM_model\n")
            f.write(f"序列长度: {args.sequence_len}\n")
            f.write(f"特征维度: {args.feature_num}\n")
            f.write(f"批大小: {args.batch_size}\n")
            f.write(f"最大训练轮数: {args.max_epochs}\n")
            f.write(f"随机种子: {seed}\n")
            f.write(f"学习率: {args.lr}\n")
            f.write("学习率调度器: step\n")
            f.write(f"Step步长(step_size): {args.step_size}\n")
            f.write(f"Step衰减系数(gamma): {args.gamma}\n")
            f.write(f"GAT隐藏层维度: {gat_hidden_dims}\n")
            f.write(f"LSTM隐藏层维度: {lstm_hidden_dims}\n")
            f.write(f"GAT Dropout系数: {args.gat_dropout}\n")
            f.write(f"GAT LeakyReLU系数(alpha): {args.gat_alpha}\n")
            f.write(f"图构建模式: {args.graph_mode}\n")
            f.write(f"节点嵌入维度(embed_dim): {args.gat_embed_dim}\n")
            f.write(f"Topk邻居数: {args.gat_topk}\n")
            f.write(f"是否启用指数平滑: {args.use_exponential_smoothing}\n")
            f.write(f"平滑系数(smooth_rate): {smooth_rate}\n")
            f.write("------------------------------\n")

        train(
            model,
            train_loader,
            valid_loader,
            test_loader,
            args.max_epochs,
            optimizer,
            scheduler,
            criterion_train,
            criterion_eval,
            lines_list=[],
            patience=args.patience,
            max_rul=args.max_rul,
            num_test_windows=num_test_windows,
            device=device,
            time=run_tag,
            log_path=log_path,
            checkpoint_dir=seed_output_dir,
            checkpoint_prefix="model_" + args.sub_dataset + "_" + run_tag,
            log_language="zh",
        )

        model_output_dir = seed_output_dir
        os.makedirs(model_output_dir, exist_ok=True)

        saved_model_path = os.path.join(
            model_output_dir,
            "model_" + args.sub_dataset + "_" + run_tag + ".pkl",
        )

        if args.save_model:
            torch.save(model, saved_model_path)
        else:
            saved_model_path = os.path.join(
                model_output_dir,
                "best_model_" + args.sub_dataset + "_" + run_tag + ".pkl",
            )

        auto_test_cmd = [
            sys.executable,
            os.path.join(parent_dir, "scipt", "test_model.py"),
            "--sub-dataset",
            args.sub_dataset,
            "--smooth-rate",
            str(smooth_rate),
            "--sequence-len",
            str(args.sequence_len),
            "--feature-num",
            str(args.feature_num),
            "--dataset-root",
            args.dataset_root,
            "--max-rul",
            str(args.max_rul),
            "--batch-size",
            str(args.batch_size),
            "--model-path",
            saved_model_path,
        ]
        if args.no_cuda:
            auto_test_cmd.append("--no-cuda")

        test_result = subprocess.run(
            auto_test_cmd,
            cwd=parent_dir,
            capture_output=True,
            text=True,
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[自动测试] 命令: " + " ".join(auto_test_cmd) + "\n")
            if test_result.stdout:
                f.write(_translate_auto_test_output(test_result.stdout.strip()) + "\n")
            if test_result.returncode != 0:
                f.write("[自动测试][错误] 返回码={}\n".format(test_result.returncode))
                if test_result.stderr:
                    f.write(test_result.stderr.strip() + "\n")

        if test_result.returncode == 0:
            print("[自动测试] 已完成，结果已追加到日志。")
            if test_result.stdout:
                print(_translate_auto_test_output(test_result.stdout.strip()))
        else:
            print("[自动测试][错误] 运行 test_model.py 失败，请查看日志。")
