import torch


class TextGenerator:

    def __init__(self, probability, vocabulary):
        self.probability = probability
        self.vocabulary = self._rev_vocab(vocabulary)
        self.start = -1

    def _rev_vocab(self, vocab):
        m = {}
        for i in vocab.items():
            m[i[1]] = i[0]

        return m

    def generate_next(self, current):
        return torch.multinomial(
            self.probability[current],
            num_samples=1,
            replacement=True,
        ).item()

    def generate(self):
        generated = []
        c = self.start
        l = len(self.probability) - 1

        while True:
            nxt = self.generate_next(c)
            if nxt == l:
                break
            generated.append(self.vocabulary[nxt])
            c = nxt
        return " ".join(generated)
