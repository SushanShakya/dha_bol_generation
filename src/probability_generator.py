from functools import reduce
import torch


class ProbabilityGenerator:

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def bigram(self):
        tmp = self.embeddings
        a = tmp[:-1]
        b = tmp[1:]
        return list(zip(a, b))

    def bigram_count(self):
        bigraph = self.bigram()

        def calc(a, b):
            if b in a:
                a[b] += 1
            else:
                a[b] = 1

            return a

        return reduce(calc, bigraph, dict())

    def probability_matrix(self):
        p = self.bigram_count().items()
        l = len(p)

        result = torch.zeros(l, l)

        for a, b in p:
            result[a[0], a[1]] = b

        for i in range(len(result)):
            result[i] /= result[i].sum()

        return result
