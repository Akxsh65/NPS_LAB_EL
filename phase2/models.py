import math
import torch
import torch.nn as nn


class CNNBiLSTMClassifier(nn.Module):
    """
    Input:  (B, 3, 30)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        lstm_input = conv_channels * 2
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 30)
        x = self.conv(x)  # (B, C, L')
        x = x.transpose(1, 2)  # (B, L', C)
        out, _ = self.lstm(x)  # (B, L', 2*hidden)
        pooled = out.mean(dim=1)  # temporal global average pooling
        return self.fc(self.dropout(pooled))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """
    Input:  (B, 3, 30)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        seq_len: int = 30,
        in_channels: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 30) -> (B, 30, 3)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.fc(self.dropout(pooled))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "cnn_bilstm":
        return CNNBiLSTMClassifier(num_classes=num_classes)
    if model_name == "transformer":
        return TransformerClassifier(num_classes=num_classes)
    raise ValueError(f"Unsupported model '{model_name}'. Use one of: cnn_bilstm, transformer")
