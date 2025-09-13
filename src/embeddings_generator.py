import torch


class EmbeddingsGenerator:

    def __init__(self, probability):
        self.probability = probability
        self.start = -1

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
            generated.append(nxt)
            c = nxt

        return generated
