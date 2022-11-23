import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math
from torchlibrosa.augmentation import SpecAugmentation
import random
from warnings import filterwarnings
filterwarnings("ignore")

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        # print('att1', x.shape) torch.Size([32, 1536, 7])
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        # print('att2', norm_att.shape) torch.Size([32, 152, 7])
        cla = self.nonlinear_transform(self.cla(x))
        # print('att3', cla.shape) torch.Size([32, 152, 7])
        x = torch.sum(norm_att * cla, dim=2)
        # print('att4', x.shape) torch.Size([32, 152])
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.SiLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h

class SeqEncoder2(nn.Module):
    def __init__(self, in_dim: int, num_feat: int):
        super(SeqEncoder2, self).__init__()
        self.conv0 = Conv1dStack(in_dim, num_feat, 3, padding=1)
        self.conv1 = Conv1dStack(num_feat, num_feat//2, 6, padding=5, dilation=2)
        self.conv2 = Conv1dStack(num_feat//2, num_feat//4, 15, padding=7, dilation=1)
        self.conv3 = Conv1dStack(num_feat//4, num_feat//4, 30, padding=29, dilation=2)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerWrapper(nn.Module):
    def __init__(self, dmodel=256, nhead=8, num_layers=2):
        super(TransformerWrapper, self).__init__()
        self.pos_encoder = PositionalEncoding(256)
        encoder_layer = TransformerEncoderLayer(d_model=dmodel, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.pos_emb = PositionalEncoding(dmodel)

    def flatten_parameters(self):
        pass

    def forward(self, x):
        x = x.permute((1, 0, 2)).contiguous()
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        x = x.permute((1, 0, 2)).contiguous()
        return x, None


class ConvWithRes(nn.Module):
    def __init__(self, in_f, out_f, k):
        super(ConvWithRes, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_f, out_f, k, 1, padding = k//2),
            nn.InstanceNorm1d(out_f),
            nn.Dropout(0.1),
            nn.SiLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_f, out_f, 1, 1),
            nn.InstanceNorm1d(out_f),
            nn.Dropout(0.1),
            nn.SiLU(),
        )
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = F.relu(x1 + x2)
        return x

class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM=nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1=nn.Linear(d_model*2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)


    def forward(self, x):
        res=x
        x, _ = self.LSTM(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=res+x
        return x
    
class SAKTModel(nn.Module):
    def __init__(self, n_skill, embed_dim=128, pos_encode='LSTM', nlayers=2, rnnlayers=3,
    dropout=0.1, nheads=8):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        #self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        if pos_encode=='LSTM':
            self.pos_encoder = nn.ModuleList([ResidualLSTM(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU':
            self.pos_encoder = nn.ModuleList([ResidualGRU(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU2':
            self.pos_encoder = nn.GRU(embed_dim,embed_dim, num_layers=2,dropout=dropout)
        elif pos_encode=='RNN':
            self.pos_encoder = nn.RNN(embed_dim,embed_dim,num_layers=2,dropout=dropout)
        self.pos_encoder_dropout = nn.Dropout(dropout)

        # print(n_skill, embed_dim)
        self.embedding = nn.Linear(n_skill, embed_dim)
        # self.cat_embedding = nn.Embedding(n_cat, embed_dim, padding_idx=0)
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim*4, dropout) for i in range(nlayers)]
        conv_layers = [nn.Conv1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        deconv_layers = [nn.ConvTranspose1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        layer_norm_layers = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        layer_norm_layers2 = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.nheads = nheads
        # self.pred = nn.Linear(embed_dim, nout)
        # self.downsample = nn.Linear(embed_dim*2,embed_dim)

    def forward(self, numerical_features):
        device = numerical_features.device
        numerical_features = numerical_features.permute(0, 2, 1)
        # print(numerical_features.shape)
        numerical_features=self.embedding(numerical_features)
        x = numerical_features#+categorical_features
        x = x.permute(1, 0, 2)
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x=lstm(x)
        # x = self.pos_encoder(x)
        x = self.pos_encoder_dropout(x)
        x = self.layer_normal(x)

        # print(x.shape)

        feats = []
        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(self.conv_layers,
                                                               self.transformer_encoder,
                                                               self.layer_norm_layers,
                                                               self.layer_norm_layers2,
                                                               self.deconv_layers):
            #LXBXC to BXCXL
            res=x
            x=F.relu(conv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm1(x)
            x=transformer_layer(x)
            x=F.relu(deconv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm2(x)
            x=res+x
            # print(x.shape)

            feats.append(x)

        # x = torch.cat(feats, dim=2)
        x = x.permute(1, 0, 2)
        # print(x.shape)

        return x
        # output = self.pred(x)

        # return output.squeeze(-1)
    

class NModel(nn.Module):
    def __init__(self, cfg, in_feat):
        super(NModel, self).__init__()
        # self.encoder =  SeqEncoder2(in_feat, 768)
        self.cfg = cfg
        # feat_num = 1024
        # self.encoder =  SeqEncoder2(in_feat, feat_num)
        # # self.encoder2 =  SeqEncoder2(2*feat_num, feat_num)
        # self.encoder2 = ConvWithRes(2*feat_num, 2*feat_num, 5)
        # # self.encoder3 =  SeqEncoder2(2*feat_num, feat_num)

        self.encoder = SAKTModel(in_feat, embed_dim=1024, pos_encode='LSTM',
                    nlayers=3, rnnlayers=3, dropout=0.2,nheads=16) 

        num_features = 1024


        self.bn0 = nn.BatchNorm1d(in_feat)

        # self.conv1 = ConvWithRes(in_feat, 768, 5)
        # self.conv2 = ConvWithRes(768, 384, 3)
        # self.conv3 = ConvWithRes(384, 192, 5)
        # num_features = 3*448 + 3*320

        # num_features = 2*feat_num
        self.rnn0 = TransformerWrapper(num_features, nhead=8, num_layers=2)

        self.rnn1 = nn.LSTM(num_features, num_features // 2, batch_first=True, num_layers=1, bidirectional=True)
        self.rnn2 = nn.GRU(num_features, num_features // 2, batch_first=True, num_layers=1, bidirectional=True)

        # self.rnn1 = TransformerWrapper(num_features, nhead=4, num_layers=1)
        # self.rnn2 = TransformerWrapper(num_features, nhead=4, num_layers=2)

        self.fc1 = nn.Linear(num_features, num_features, bias=True)
        self.att_block = AttBlockV2(
            num_features, len(cfg.target_cols), activation="sigmoid")

        self.init_weight()

        self.round = 1
        if '_r3' in self.cfg.name:
            self.round = 3

        if '_r4' in self.cfg.name:
            self.round = 4

        if '_r5' in self.cfg.name:
            self.round = 5

        if '_r6' in self.cfg.name:
            self.round = 6

        if '_r7' in self.cfg.name:
            self.round = 7

        if '_r8' in self.cfg.name:
            self.round = 8


    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)


    def forward(self, x, mask=None):
        # x3 = self.encoder(x)

        # x = self.conv1(x)
        # x1 = self.conv2(x)
        # x2 = self.conv3(x1)
        # x = torch.cat([x, x1, x2, x3], dim=1)

        x = self.bn0(x)


        if self.training:
            if self.round == 4:
                aug_prob = 0.25
            elif self.round == 5:
                aug_prob = 0.4
            elif self.round == 8:
                aug_prob = 0.25
            else:
                aug_prob = 0.0

            if random.random() < aug_prob:
                spec_augmenter =  SpecAugmentation(time_drop_width=2, time_stripes_num=4,
   freq_drop_width=2, freq_stripes_num=4)
                x = x.unsqueeze(1)
                x = spec_augmenter(x)
                x = x.squeeze(1)
                # print(x.shape)


        x = self.encoder(x)
        # x = self.encoder2(x)
        # x = self.encoder3(x)

        # features = x.permute(0, 2, 1).contiguous()
        features = x
        # self.rnn0.flatten_parameters()
        x, _ = self.rnn0(features)

        x = F.dropout(x, p=0.25)
        x, _ = self.rnn1(x)

        x = F.dropout(x, p=0.25)
        x, _ = self.rnn2(x)

        frames_num = x.shape[1]

        # print('1',x.shape)
        # x = F.dropout(x, p=0.3)
        x = x.transpose(1, 2)
        # (seq_output, norm_att, row_output) = self.att_block(x)
        # print('2', seq_output.shape, row_output.shape) 

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)

        # x1_m = F.max_pool1d(x_m, kernel_size=3, stride=1, padding=1)
        # x2_m = F.avg_pool1d(x_m, kernel_size=3, stride=1, padding=1)

        x = x1 + x2
        # x = x1 + x2 + x1_m + x2_m

        # print('2',x.shape)

        x = F.dropout(x, p=self.cfg.drop_rate, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=self.cfg.drop_rate, training=self.training)

        # print(x.shape) #torch.Size([4, 2048, 20])

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # print('1', segmentwise_logit.shape, segmentwise_output.shape) 
        #1 torch.Size([4, 20, 152]) torch.Size([4, 20, 152])

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        # framewise_output = interpolate(segmentwise_output,
        #          interpolate_ratio)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)
        # print(logit.shape, framewise_logit.shape)
        return {
        'out1':logit,
        'out2':framewise_logit
        }