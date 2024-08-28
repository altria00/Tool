# import torch
# from torch import nn
# import torch.nn.functional as F
# import math
#
#
# class TokenEmbedding(nn.Embedding):
#     def __init__(self, vocab_size, d_model):
#         super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
#
#
# class PositionEmbedding(nn.Module):
#     def __init__(self, d_model, maxlen, device):
#         super().__init__()
#         self.P = torch.zeros(1, maxlen, d_model, device)
#         self.P.requires_grad(False)
#
#         position = torch.arange(0, maxlen, dtype=torch.float, device=device).unsqueeze(1)
#         _2i = torch.arange(0, d_model, 2)
#
#         self.P[:, :, 0:2] = torch.sin(position / (torch.pow(10000, _2i / d_model)))
#         self.P[:, :, 1:2] = torch.cos(position / (torch.pow(10000, _2i / d_model)))
#
#     def forward(self, x):
#         return self.P[:, :x.shape(1), :].to(x.device)
#
#
# class TransformEmbedding(nn.Module):
#     def __init__(self, vocab_size, d_model, maxlen, device, dropout=0.1):
#         super().__init__()
#         self.token_embedding = TokenEmbedding(vocab_size, d_model)
#         self.position_embedding = PositionEmbedding(d_model, maxlen, device)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         token_embedded = self.token_embedding(x)
#         position_embedded = self.position_embedding(x)
#         return self.dropout(token_embedded + position_embedded)
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, d_model, eps=1e-6):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(1, d_model, 1))
#         self.beta = nn.Parameter(torch.zeros(1, d_model, 1))
#         self.eps = eps
#
#     def foward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, unbiased=False, keepdim=True)
#         normalized_x = (x - mean) / torch.sqrt(var + self.eps)
#         return self.alpha * normalized_x + self.beta
#
#
# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return self.dropout(x)
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
#         super().__init__()
#         self.norm1 = LayerNorm(d_model)
#         self.drop1 = nn.Dropout(dropout)
#         self.norm2 = LayerNorm(d_model)
#         self.drop2 = nn.Dropout(dropout)
#         self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
#
#     def forward(self, x, mask=None):
#         _x = x
#         x = x + self.drop1(self.self_attn(x, x, x, mask))
#         x = self.norm1(x + _x)
#         _x = x
#         x = x + self.drop2(self.feed_forward(x))
#         x = self.norm2(x + _x)
#
#         return x
#
#
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
#         self.norm1 = LayerNorm(d_model)
#         self.drop1 = nn.Dropout(dropout)
#
#         self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
#         self.norm2 = LayerNorm(d_model)
#         self.drop2 = nn.Dropout(dropout)
#
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
#         self.norm3 = LayerNorm(d_model)
#         self.drop3 = nn.Dropout(dropout)
#
#     def forward(self, dec, enc, t_mask, s_mask):
#         _dec = dec
#         q = dec
#         k = dec
#         v = dec
#
#         q = self.self_attn(q, k, v, t_mask)  # 下三角掩码 使其看不到未来的信息
#         q = self.drop1(q)
#         q = self.norm1(q + _dec)
#         _dec = q
#
#         if enc is not None:
#             q = q + self.cross_attn(q, enc, enc, s_mask)  # 位置的掩码，不用关注padding的信息
#             q = self.drop2(q)
#             q = self.norm2(q + _dec)
#             _dec = q
#
#         q = q + self.feed_forward(q)
#         q = self.drop3(q)
#         q = self.norm3(q + _dec)
#
#         return q
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_heads, dropout=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.dropout = nn.Dropout(dropout)
#         self.linear_Q = nn.Linear(d_model, d_model)
#         self.linear_K = nn.Linear(d_model, d_model)
#         self.linear_V = nn.Linear(d_model, d_model)
#         self.linear_out = nn.Linear(d_model, d_model)
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, q, k, v, mask=None):
#         batch_size, seq_len, dim = q.shape
#         d_k = self.d_model // self.n_heads
#         Q = self.linear_Q(q).view(batch_size, seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
#         K = self.linear_K(k).view(batch_size, seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
#         V = self.linear_V(v).view(batch_size, seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
#
#         attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
#
#         if mask is not None:
#             attention_weights = attention_weights.masked_fill(mask == 0, 1e-9)
#         attention_weights = self.softmax(attention_weights) @ V
#         attention_weights = attention_weights.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, dim)
#         return self.linear_out(attention_weights)
#
#
# class Encoder(nn.Module):
#     def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
#         self.embedding = TransformEmbedding(vocab_size, d_model, maxlen, device)
#
#     def forward(self, x, mask=None):
#         x = self.embedding(x)
#         for layer in self.layers:
#             x = layer(x, mask)
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
#         self.embedding = TransformEmbedding(vocab_size, d_model, maxlen, device)
#         self.linear_out = nn.Linear(d_model, vocab_size)
#
#     def forward(self, dec, enc, t_mask, s_mask):
#         x = self.embedding(dec)
#         for layer in self.layers:
#             x = layer(x, enc, t_mask, s_mask)
#         x = self.linear_out(x)
#         return x
#
#
# class Transformer(nn.Module):
#     def __init__(self, src_pad_idx, trg_pad_idx, enc_vocab_size, dec_vocab_size, maxlen, d_model, n_layers, n_heads,
#                  d_ff, device, dropout=0.1):
#         super().__init__()
#         self.src_pad_idx = src_pad_idx
#         self.trg_pad_idx = trg_pad_idx
#         self.encoder = Encoder(enc_vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout)
#         self.decoder = Decoder(dec_vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout)
#
#     def make_t_mask(self, q, k):
#         len_q, len_k = q.size(1), k.size(1)
#         mask = torch.tril(torch.ones((len_q, len_k), device=self.device)).bool()
#         return mask
#
#     def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
#         len_q, len_k = q.size(1), k.size(1)
#         mask_q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
#         mask_q.repeat(1, 1, 1, len_k)
#         mask_k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
#         mask_k.repeat(1, 1, len_q, 1)
#
#         mask = mask_q & mask_k
#         return mask
#
#     def forward(self, src, trg):
#         src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
#         trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_t_mask(trg, trg)
#         src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
#         enc_output = self.encoder(src, src_mask)
#         dec_output = self.decoder(trg, enc_output, trg_mask, src_trg_mask)
#         return dec_output

# def initialize_weights(m):
#     if hasattr(m, "weight") and m.weight.dim() > 1:
#         nn.init.kaiming_uniform(m.weight.data)
#
#
# if __name__ == "__main__":
#     enc_vocab_size = 5893
#     dec_vocab_size = 7853
#     src_pad_idx = 1
#     trg_pad_idx = 1
#     trg_sos_idx = 2
#     batch_size = 128
#     max_len = 1024
#     d_model = 512
#     n_layers = 3
#     n_heads = 2
#     ffn_hidden = 1024
#     drop_prob = 0.1
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model = Transformer(
#         src_pad_idx=src_pad_idx,
#         trg_pad_idx=trg_pad_idx,
#         d_model=d_model,
#         enc_vocab_size=enc_vocab_size,
#         dec_vocab_size=dec_vocab_size,
#         maxlen=max_len,
#         d_ff=ffn_hidden,
#         n_heads=n_heads,
#         n_layers=n_layers,
#         device=device,
#     ).to(device)
#
#     model.apply(initialize_weights)
#     src = torch.load("tensor_src.pt")
#     src = torch.cat((src, torch.ones(src.shape[0], 2, dtype=torch.int)), dim=-1)
#     trg = torch.load("tensor_trg.pt")
#
#     result = model(src, trg)
#     print(result, result.shape)


import torch
from torch import nn
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        self.encoding.requires_grad_(False)

        pos = torch.arange(0, maxlen, device=device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_groups):
        super(GroupQueryAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups

        assert d_model % n_heads == 0
        self.n_heads_groups = self.n_heads // self.n_groups
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_v = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def expand(self, data):
        batch, time = data.shape[0], data.shape[2]
        data = data[:, :, None, :, :].expand(batch, self.n_groups, self.n_heads_groups, time,
                                             self.head_dim).contiguous()
        data = data.view(batch, self.n_groups * self.n_heads_groups, time, self.head_dim)

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch = q.shape[0]
        q = q.view(batch, -1, self.n_groups * self.n_heads_groups, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, -1, self.n_groups, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, -1, self.n_groups, self.head_dim).permute(0, 2, 1, 3)

        k = self.expand(k)
        v = self.expand(v)
        score = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.d_model)
        output = self.w_combine(score)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask is not None:
            # mask = torch.tril(torch.ones(time, time, dtype=bool))
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)  # 下三角掩码

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)

            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            env_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_head,
            n_layer,
            drop_prob,
            device,
    ):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(
            env_voc_size, d_model, max_len, drop_prob, device
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            dec_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_head,
            n_layer,
            drop_prob,
            device,
    ):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(
            dec_voc_size, d_model, max_len, drop_prob, device
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)

        dec = self.fc(dec)

        return dec


class Transformer(nn.Module):
    def __init__(
            self,
            src_pad_idx,
            trg_pad_idx,
            enc_voc_size,
            dec_voc_size,
            max_len,
            d_model,
            n_heads,
            ffn_hidden,
            n_layers,
            drop_prob,
            device,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            enc_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device,
        )
        self.decoder = Decoder(
            dec_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)

        # (Batch, Time, len_q, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = (
            torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        )
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(
            trg, trg, self.trg_pad_idx, self.trg_pad_idx
        ) * self.make_casual_mask(trg, trg)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        ouput = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return ouput


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


if __name__ == "__main__":
    enc_voc_size = 5893
    dec_voc_size = 7853
    src_pad_idx = 1
    trg_pad_idx = 1
    trg_sos_idx = 2
    batch_size = 128
    max_len = 1024
    d_model = 512
    n_layers = 3
    n_heads = 2
    ffn_hidden = 1024
    drop_prob = 0.1
    device = torch.device("cpu")

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        d_model=d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_heads=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
    ).to(device)

    model.apply(initialize_weights)
    src = torch.load("tensor_src.pt")
    src = torch.cat((src, torch.ones(src.shape[0], 2, dtype=torch.int)), dim=-1)
    trg = torch.load("tensor_trg.pt")

    result = model(src, trg)
    print(result, result.shape)