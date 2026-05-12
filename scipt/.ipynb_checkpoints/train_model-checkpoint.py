import datetime
import argparse
import os
from dataset import *
from model import *
from utils import *
# 自动创建 logs 文件夹（不存在就新建）
log_dir = "/CMAPSS-release/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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
    parser.add_argument('--sub-dataset', type=str, default='FD001', help='FD001/2/3/4')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
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
    parser.add_argument('--feature-attention-size', type=int, default=4,
                        help='Attention size for self-concat feature attention')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', type=str, default=True, help='save trained models')
    args = parser.parse_args()

    torch.manual_seed(28)

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
                             seq_len=args.sequence_len, attention_size=28)
    model = EncoderDecoder(encoder=encoder, decoder=decoder,
                           feature_attention_size=args.feature_attention_size,
                           use_spatial_gat=args.use_spatial_gat,
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(log_dir, "train_log_"+args.sub_dataset+'_'+time+".txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-----"+ model_type + "-----\n")
        f.write("-----"+ args.sub_dataset + "-----\n")


    train(
        model, train_loader, valid_loader,
        test_loader, args.max_epochs, optimizer,
        scheduler, criterion_train, criterion_eval,
        lines_list=[], patience=args.patience, max_rul=args.max_rul, num_test_windows=num_test_windows,
        device=torch.device('cuda') if not args.no_cuda else torch.device('cpu'), time=time, log_path=log_path)
    


    if args.save_model:
        torch.save(model, parent_dir+'/trials/'+'model_'+args.sub_dataset+'_'+time+'.pkl')
