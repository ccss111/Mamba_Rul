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
    def __init__(self, input_size, num_layers, num_hidden, seq_len, attention_size):
        super(Seq2SeqDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size+num_hidden,
                                  num_layers=num_layers,
                                  hidden_size=num_hidden,
                                  batch_first=True)
        self.attention = AdditiveAttentionForSeq(num_hidden=num_hidden,
                                                 attention_size=attention_size,
                                                 seq_len=seq_len)
        self.Linear = nn.Linear(num_hidden, 1)

    def forward(self, decoder_x, encoder_output, encoder_state):
        output, attention_weights = self.attention(encoder_output, encoder_state[0])  # encoder_state[0]表示h
        output = torch.cat((output, decoder_x), dim=-1)
        output, _ = self.lstm(output, encoder_state)
        output = output.squeeze(1)
        output = self.Linear(output)
        return output, attention_weights


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 feature_attention_size,
                 use_spatial_gat=True,
                 num_sensors=14,
                 gat_hidden_dim=8,
                 gat_out_dim=16,
                 gat_embed_dim=16,
                 gat_topk=5,
                 gat_dropout=0.0,
                 gat_alpha=0.1):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_spatial_gat = use_spatial_gat

        if use_spatial_gat:
            self.spatial_gat = SensorSpatialGAT(
                num_sensors=num_sensors,
                in_features=1,
                hidden_features=gat_hidden_dim,
                out_features=gat_out_dim,
                embed_dim=gat_embed_dim,
                topk=gat_topk,
                dropout=gat_dropout,
                alpha=gat_alpha,
            )
            attention_input_size = gat_out_dim
        else:
            self.spatial_gat = None
            attention_input_size = num_sensors

        self.feature_attention = SelfConcatAttentionForSeq(
            input_size=attention_input_size,
            attention_size=feature_attention_size,
        )

    def forward(self, encoder_x):
        if self.use_spatial_gat:
            encoder_x = self.spatial_gat(encoder_x)

        encoder_output, encoder_state = self.encoder(encoder_x)
        decoder_x, attention_weight_feature = self.feature_attention(encoder_x, encoder_x, encoder_x)
        output, attention_weights = self.decoder(decoder_x, encoder_output, encoder_state)
        return output, attention_weights
