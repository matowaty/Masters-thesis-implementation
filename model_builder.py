"""
model_builder.py — PyTorch Deep Learning Model Architectures

Defines the core forecasting models:
    1. BiLSTMModel: Standard Bidirectional LSTM.
    2. BiLSTMAttentionModel: BiLSTM with an additive attention mechanism.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect and return the best available PyTorch device.

    Checks for CUDA (NVIDIA), MPS (Apple Silicon), and falls back to CPU.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info("Auto-detected PyTorch device: %s", device)
    return device


class BiLSTMModel(nn.Module):
    """Standard Bidirectional LSTM for time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the BiLSTM model.

        Args:
            input_size: Number of features in the input data.
            hidden_size: Number of features in the hidden state.
            num_layers: Number of recurrent layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout for the output of the LSTM
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        # Multiply by 2 because it's bidirectional
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size].

        Returns:
            Output tensor of shape [batch_size, 1].
        """
        # lstm_out: [batch_size, sequence_length, hidden_size * 2]
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time step
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size * 2]

        last_hidden = self.dropout(last_hidden)

        # out: [batch_size, 1]
        out = self.fc(last_hidden)
        
        return out


class BiLSTMAttentionModel(nn.Module):
    """Bidirectional LSTM with Additive Attention for time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the BiLSTM with Attention model.

        Args:
            input_size: Number of features in the input data.
            hidden_size: Number of features in the hidden state.
            num_layers: Number of recurrent layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention layer (additive attention)
        # We learn a linear transformation to a context vector
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size].

        Returns:
            Output tensor of shape [batch_size, 1].
        """
        # lstm_out: [batch_size, sequence_length, hidden_size * 2]
        lstm_out, _ = self.lstm(x)

        # Attention mechanism
        # Calculate attention scores
        # scores: [batch_size, sequence_length, 1]
        scores = self.attention_weights(lstm_out)
        
        # Normalize scores to probabilities over the sequence length
        # alpha: [batch_size, sequence_length, 1]
        alpha = F.softmax(scores, dim=1)
        
        # Compute context vector as weighted sum of LSTM outputs
        # context: [batch_size, hidden_size * 2]
        context = torch.sum(alpha * lstm_out, dim=1)

        context = self.dropout(context)

        # out: [batch_size, 1]
        out = self.fc(context)

        return out


class ClassificationBiLSTMModel(nn.Module):
    """Bidirectional LSTM for 3-class time-series classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 3)  # 3-class logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out


class ClassificationBiLSTMAttentionModel(nn.Module):
    """Bidirectional LSTM with Additive Attention for 3-class classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 3)  # 3-class logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        scores = self.attention_weights(lstm_out)
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(alpha * lstm_out, dim=1)
        context = self.dropout(context)
        out = self.fc(context)
        return out
