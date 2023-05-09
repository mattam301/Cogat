from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
sbert_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")

#sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")


class Sample:
    def __init__(self, vid, speaker, label, text, audio, visual, sentence):
        self.vid = vid
        self.speaker = speaker
        self.label = label
        self.text = text
        self.audio = audio
        self.visual = visual
        self.sentence = sentence
        self.sbert_sentence_embeddings = sbert_model.encode(sentence)
