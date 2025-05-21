import torch
from torch import nn
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


class LSTMOtm(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, vocab_size):
        super(LSTMOtm, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_state = nn.LSTMCell(encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.reset_parameters()  # initialize some layers with the uniform distribution

    def reset_parameters(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
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

        encoder_out = encoder_out.view(batch_size, -1)

        # 根据解析排序，长->短
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]

        decode_lengths = (caption_lengths - 1).tolist()

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        for t in range(max(caption_lengths)):
            # 由于排序后，长的在前短的在后，由于batch输入需要padding，找出截止到目前没padding的batch
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.hidden_state(encoder_out[:batch_size_t, t], (h, c))

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
        return predictions, sort_ind


class Sat(nn.Module):
    def __init__(self, cfg):
        super(Sat, self).__init__()

        encoder_dim = cfg['encoder_dim']
        decoder_dim = cfg['encoder_dim']
        vocab_size = cfg['vocab_size']

        self.encoder = Backbone(name='resnet50',
                                layers_to_train=[],
                                return_interm_layers={'layer4': "0"},
                                pretrained=True,
                                norm_layer=FrozenBatchNorm2d)

        self.decoder = LSTMOtm(encoder_dim, decoder_dim, vocab_size)

    def forward(self, batch):
        x = batch[0]
        cap_lens = batch[2]

        encoder_out = self.encoder(x)['0']
        scores, sort_ind = self.decoder(encoder_out, cap_lens)

        if self.training:
            targets = batch[1]
            targets = targets[sort_ind, 1:]
            loss = self.criterion(scores, targets)
            top5 = accuracy(scores, targets, 5)
            return loss, {'ce_loss': loss.item(), 'Top-5 Accuracy': top5}

        return scores

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            self.criterion = nn.CrossEntropyLoss()

        return self.criterion(preds, targets)
