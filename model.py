import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # outputs = [batch_size, seq_len, hid_dim * n_directions]
        # hidden = [batch_size, n_layers * n_direction, hid_dim]
        # cell = [batch_size, n_layers * n_direction, hid_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        # x: (N)
        assert len(x.shape) == 1
        x = x.unsqueeze(0, dim=1)  # (N, 1)
        assert len(hidden.shape) == 2
        x = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(x, (hidden, cell))
        # lstm_out: (N, 1, hidden_dim)
        prediction = self.fc_out(lstm_out.squeeze(1))  # (N, hidden_dim)
        assert prediction.shape[1] == self.hidden_dim

        return prediction, (hidden, cell)


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
        self.device = device
        self.vocab_size = vocab_size

    def forward(self, x, y):
        # x: (N, L)
        hidden, cell = self.encoder(x)
        sequence_length = x.shape[1]
        predictions = torch.zeros(x.shape[0], sequence_length, self.vocab_size).to(self.device)

        target = y[:, 0]
        for i in range(1, sequence_length):
            output, (hidden, cell) = self.decoder(target, hidden, cell)
            predictions[:, i] = output
            target = y[:, i]

        return predictions
