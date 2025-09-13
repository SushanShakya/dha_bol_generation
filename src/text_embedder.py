import os
import sys
import re


class TextEmbedder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = self._stoi()
        self.itos = self._itos()

    def _stoi(self):
        return {v: i for i, v in enumerate(self.vocab)}

    def _itos(self):
        return {i: v for i, v in enumerate(self.vocab)}

    def embed(self, text):
        encoding = self.stoi
        tmp = list(map(lambda a: encoding[a], text.split(" ")))
        return [-1, *tmp, -1]

    def unembed(self, embeddings):
        return list(map(lambda a: self.itos.get(a, ""), embeddings))
