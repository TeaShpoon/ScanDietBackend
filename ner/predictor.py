import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List, Tuple
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize  # noqa: E402


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim,
        hidden_dim,
        dropout,
        pad_idx,
        embedding_matrix=None,
        freeze_embeddings=False,
    ):
        super(BiLSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, mask):
        embeddings = self.embedding(x)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def predict(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask.bool())


class NERPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Load the trained model and vocabularies from a checkpoint.

        Args:
            checkpoint_path: Path to the .pt file saved by the training script.
            device: 'cpu' or 'cuda'. If None, auto-detect.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.word2idx = checkpoint["word2idx"]
        self.idx2tag = checkpoint["idx2tag"]
        self.tag2idx = checkpoint["tag2idx"]
        self.embedding_matrix = checkpoint["embedding_matrix"]
        config = checkpoint["config"]

        self.pad_idx = self.word2idx.get("<PAD>", 0)
        self.unk_idx = self.word2idx.get("<UNK>", 1)

        # Recreate model
        self.model = BiLSTMCRF(
            vocab_size=len(self.word2idx),
            tagset_size=len(self.idx2tag),
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            pad_idx=self.pad_idx,
            embedding_matrix=self.embedding_matrix,
            freeze_embeddings=False,  # doesn't matter for inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_sentence(self, sentence: str) -> List[Tuple[str, str]]:
        words = word_tokenize(sentence)
        # Convert words to indices
        word_ids = [self.word2idx.get(w, self.unk_idx) for w in words]
        x = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        mask = torch.ones_like(x).float().to(self.device)

        with torch.no_grad():
            preds = self.model.predict(x, mask)[0]  # list of tag indices

        tags = [self.idx2tag[p] for p in preds]
        return list(zip(words, tags))

    def predict_batch(self, sentences: List[str]) -> List[List[Tuple[str, str]]]:
        return [self.predict_sentence(sent) for sent in sentences]
