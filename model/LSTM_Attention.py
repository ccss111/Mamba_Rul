from .Attention_modules import *
from .GAT import SensorSpatialGAT


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, num_layers, num_hidden):
        super(Seq2SeqEncoder, self).__init__()
        self.nn_lstm = nn.LSTM(num_layers=num_layers, input_size=input_size, hidden_size=num_hidden, batch_first=True)
        self.lstm = self.nn_lstm

    def forward(self, x):
        output = self.lstm(x)
        return output


class Seq2SeqDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 num_layers,
                 num_hidden,
                 seq_len,
                 attention_size,
                 use_aef=True):
        super(Seq2SeqDecoder, self).__init__()
        self.input_size = input_size
        self.use_aef = bool(use_aef)
        self.lstm = torch.nn.LSTM(input_size=num_hidden,
                                  num_layers=num_layers,
                                  hidden_size=num_hidden,
                                  batch_first=True)
        self.attention = AdditiveAttentionForSeq(num_hidden=num_hidden,
                                                 attention_size=attention_size,
                                                 seq_len=seq_len)
        self.Linear = nn.Linear(num_hidden, 1)

    # 仅保留可选AEF: 开启时使用加性注意力聚合, 关闭时回退到最后时刻编码特征
    def forward(self, encoder_output, encoder_state):
        use_aef = bool(getattr(self, 'use_aef', True))

        if use_aef:
            decoder_input, attention_weights = self.attention(encoder_output, encoder_state[0])
        else:
            decoder_input = encoder_output[:, -1:, :]
            attention_weights = None

        output, _ = self.lstm(decoder_input, encoder_state)
        output = output[:, -1, :]
        output = self.Linear(output)
        return output, attention_weights


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 use_spatial_gat=True,
                 graph_mode='dynamic_knn',
                 num_sensors=14,
                 gat_hidden_dim=8,
                 gat_out_dim=16,
                 gat_num_layers=2,
                 gat_embed_dim=16,
                 gat_topk=5,
                 gat_dropout=0.0,
                 gat_alpha=0.1,
                 use_decoder=True):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_spatial_gat = use_spatial_gat
        self.graph_mode = graph_mode
        self.use_decoder = bool(use_decoder)

        if self.use_decoder and self.decoder is None:
            raise ValueError("decoder cannot be None when use_decoder=True")

        if use_spatial_gat:
            self.spatial_gat = SensorSpatialGAT(
                num_sensors=num_sensors,
                in_features=1,
                hidden_features=gat_hidden_dim,
                out_features=gat_out_dim,
                num_layers=gat_num_layers,
                embed_dim=gat_embed_dim,
                topk=gat_topk,
                graph_mode=graph_mode,
                dropout=gat_dropout,
                alpha=gat_alpha,
            )
        else:
            self.spatial_gat = None

        if not self.use_decoder:
            encoder_hidden_size = getattr(self.encoder.lstm, 'hidden_size', 0)
            if int(encoder_hidden_size) <= 0:
                raise ValueError("failed to infer encoder hidden size for encoder-only regression")
            self.encoder_regressor = nn.Linear(int(encoder_hidden_size), 1)
        else:
            self.encoder_regressor = None

    def forward(self, encoder_x):
        use_decoder = bool(getattr(self, 'use_decoder', True))

        # GAT空间建模
        if self.use_spatial_gat:
            encoder_x = self.spatial_gat(encoder_x)
        # encoder-lstm时序建模
        encoder_output, encoder_state = self.encoder(encoder_x)

        if use_decoder:
            # decoder-lstm时序建模+可选AEF注意力
            output, attention_weights = self.decoder(encoder_output, encoder_state)
            return output, attention_weights

        # 编码器消融: 去掉解码器, 直接用编码器最后时刻隐藏状态回归
        last_hidden = encoder_output[:, -1, :]
        output = self.encoder_regressor(last_hidden)
        attention_weights = torch.zeros(
            encoder_output.size(0),
            encoder_output.size(1),
            device=encoder_output.device,
            dtype=encoder_output.dtype,
        )
        return output, attention_weights
