import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from torchvision.ops.misc import FrozenBatchNorm2d

from nn.backbone import Backbone


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


class Encoder(nn.Module):
    def __init__(self, encoded_image_size):
        super(Encoder, self).__init__()

        self.backbone = Backbone(name='resnet101',
                                 layers_to_train=[],
                                 return_interm_layers={'layer4': "0"},
                                 pretrained=True,
                                 norm_layer=FrozenBatchNorm2d)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, x):
        x = self.backbone(x)['0']

        encoder_out = self.adaptive_pool(x)
        encoder_out = encoder_out.permute(0, 2, 3, 1)

        return encoder_out


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # fatt
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # MLP
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        # 位置权重
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        # z: context vector
        z = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return z, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.lh = nn.Linear(decoder_dim, embed_dim)
        self.lz = nn.Linear(encoder_dim, embed_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        # MLP
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        # MLP
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        device = encoder_out.device

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        # h-1, c-1
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        # 剔除 <end>
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            z, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            # beta
            beta = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            z = beta * z

            # h0
            # cat(E_y(t-1), h_(t-1), z(t))
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], z], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # h1
            # p = Lo(E_y(t-1) + Lhh_(t) + Lzz(t))
            preds = self.fc(embeddings[:batch_size_t, t, :] + self.lh(h) + self.lz(z))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class Nica(nn.Module):
    def __init__(self, cfg):
        super(Nica, self).__init__()

        attention_dim = cfg['attention_dim']
        embed_dim = cfg['embed_dim']
        encoder_dim = cfg['encoder_dim']
        decoder_dim = cfg['decoder_dim']
        vocab_size = cfg['vocab_size']
        encoded_image_size = cfg['encoded_image_size']

        self.encoder = Encoder(encoded_image_size=encoded_image_size)

        self.decoder = DecoderWithAttention(attention_dim=attention_dim,
                                            embed_dim=embed_dim,
                                            decoder_dim=decoder_dim,
                                            encoder_dim=encoder_dim,
                                            vocab_size=vocab_size)

    def forward(self, batch):
        x = batch[0]
        caps = batch[1]
        cap_lens = batch[2]
        cap_ids = batch[3]

        encoder_out = self.encoder(x)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(encoder_out, caps, cap_lens)

        if self.training:
            # 剔除 <start>
            # 推理的时候输入 <start> ，需要预测下一个词，且最后第二个词输入时，需要预测 <end>
            # scores :   a   b   c  <end>
            # target :   a   b   c  <end>
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data.to(self.device)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data.to(self.device)
            loss = self.loss(scores, targets)
            loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()
            top5 = accuracy(scores, targets, 5)
            return loss, {'ce_loss': loss.item(), 'Top-5 Accuracy': top5}

        cap_ids = cap_ids[sort_ind]

        id_store = [torch.where(cap_ids == k)[0] for k in torch.unique(cap_ids)]

        tg_caps = []
        for i in cap_ids:
            ind = id_store[i]
            tg_caps.append(caps_sorted[ind])

        return scores, tg_caps

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            self.criterion = nn.CrossEntropyLoss()

        return self.criterion(preds, targets)
