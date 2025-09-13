from functools import reduce
import torch


class ProbabilityGenerator:

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def pad(self):
        return [-1, *self.embeddings, -1]

    def bigraph(self):
        tmp = self.pad()
        a = tmp[:-1]
        b = tmp[1:]
        return list(zip(a, b))

    def bigraph_count(self):
        bigraph = self.bigraph()

        def calc(a, b):
            if b in a:
                a[b] += 1
            else:
                a[b] = 1

            return a

        return reduce(calc, bigraph, dict())

    def probability_matrix(self):
        p = self.bigraph_count().items()
        l = len(p)

        result = torch.zeros(l, l)

        for a, b in p:
            result[a[0], a[1]] = b

        for i in range(len(result)):
            result[i] /= result[i].sum()

        return result
