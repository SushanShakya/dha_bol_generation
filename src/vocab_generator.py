import re


class VocabGenerator:
    def clean_text(self, text):
        return re.sub(r"\n+", " ", text)

    def generate(self, content):
        cleaned = self.clean_text(content)
        return list(set(cleaned.split(" ")))
