import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from nn.backbone import Backbone


class DecoderWithAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, vocab_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_state = nn.LSTMCell(encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.reset_parameters()  # initialize some layers with the uniform distribution

    def reset_parameters(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        zeros = torch.zeros(self.encoder_dim, self.decoder_dim,
                            dtype=encoder_out.dtype,
                            device=encoder_out.device)
        h = zeros  # (batch_size, decoder_dim)
        c = zeros
        return h, c

    def forward(self, encoder_out, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        encoder_out = encoder_out.view(batch_size, -1)

        # 根据解析排序，长->短
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]

        decode_lengths = (caption_lengths - 1).tolist()

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        for t in range(max(caption_lengths)):
            # 由于排序后，长的在前短的在后，由于batch输入需要padding，找出截止到目前没padding的batch
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.hidden_state(encoder_out[:batch_size_t, t], (h, c))

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
        return


class Sat(nn.Module):
    def __init__(self, cfg):
        super(Sat, self).__init__()

        self.encoder = Backbone(name='resnet50',
                                layers_to_train=[],
                                return_interm_layers={'layer4': "0"},
                                pretrained=True,
                                norm_layer=FrozenBatchNorm2d)

        self.decoder = DecoderWithAttention()
