import os
import sys
import re


class TextEmbedder:
    def __init__(self, vocab_path=None):
        self.vocab_path = (
            vocab_path if vocab_path is not None else "datasets/sample.dataset"
        )
        self.content = None

    def clean_text(self, text):
        return re.sub(r"\n+", " ", text)

    def extract_content(self):
        if self.content is not None:
            return self.content

        basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = basedir + "/" + self.vocab_path

        with open(path) as f:
            self.content = f.read()

        return self.clean_text(self.content)

    def vocabulary(self):
        content = self.extract_content()
        result = {}

        for i, word in enumerate(set(content.split(" "))):
            if word in result:
                continue
            result[word] = i

        return result

    def create_embeddings(self, text=None):
        encoding = self.vocabulary()
        text = text if text is not None else self.extract_content()
        text = self.clean_text(text)
        tmp = list(map(lambda a: encoding[a], text.split(" ")))

        return [-1, *tmp, -1]
