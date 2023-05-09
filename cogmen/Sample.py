from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
sbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

#sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v2")


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
