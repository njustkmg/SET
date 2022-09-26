import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from caption.modules.attention import MultiHeadAttention
from caption.modules.utils import sinusoid_encoding_table, PositionWiseFeedForward
from caption.modules.containers import ConModule, ModuleList


class MeshedDecoderLayer(ConModule):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)


        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        enc_att3 = self.enc_att(self_att, enc_output, enc_output, mask_enc_att) * mask_pad

        enc_att = enc_att3
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class Decoder(ConModule):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=768, d_k=96, d_v=96, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        # vocab_size = vocab_size + 1
        # print("vocab_size ============ ", vocab_size)
        # self.vl_decoder = nn.Linear(768, d_model)
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        # input = input[:, :200]
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1) # torch.triu上三角矩阵
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        # print(input.size(), seq.size()) # torch.Size([50, 275]) torch.Size([50, 275])
        # for i in input:
        #     for j in i:
        #         if j.data > 1810:
        #             print("!!!!!!: ", j.data)
        #         print(j)
        # print("input: ", input.max(), "-----", input.min())
        # print("seq: ", seq.max(), "-----", seq[0])
        # print(input.shape) #torch.Size([50, 98])
        # print(seq.shape)

        # seq = seq[:, :200]
        # encoder_output = self.vl_decoder(encoder_output)
        out = self.word_emb(input) + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        # print(out.size()) # torch.Size([10, 144, 512])
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
