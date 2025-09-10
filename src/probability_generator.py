from functools import reduce


class ProbabilityGenerator:

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def pad(self):
        return [-1, *self.embeddings, -2]

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

    def probability(self):
        bc = self.bigraph_count()
        count = len(self.bigraph())

        def calc(a, b):
            a[b] = bc[b] / count
            return a

        return reduce(calc, bc.keys(), dict())
