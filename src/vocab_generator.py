import re


class VocabGenerator:
    def clean_text(self, text):
        return re.sub(r"\n+", " ", text)

    def generate(self, content):
        return list(set(content.split(" ")))
