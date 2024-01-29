from sentence_transformers import SentenceTransformer
#model = SentenceTransformer("intfloat/multilingual-e5-large")
#model.save("./multilingual-e5-large")
#
#
#model = SentenceTransformer("thenlper/gte-large-zh")
#model.save("./gte-large-zh")
#
#
#model = SentenceTransformer("uer/sbert-base-chinese-nli")
#model.save("./sbert-base-chinese-nli")

#model = SentenceTransformer("moka-ai/m3e-base")
#model.save("./m3e-base")
#
#model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#model.save("./paraphrase-multilingual-MiniLM-L12-v2")

model = SentenceTransformer("sentence-transformers/LaBSE")
model.save("./LaBSE")


model = SentenceTransformer("moka-ai/m3e-large")
model.save("./m3e-large")

model = SentenceTransformer("symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli")
model.save("./sn-xlm-roberta-base-snli-mnli-anli-xnli")
