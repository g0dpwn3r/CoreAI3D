import gensim.downloader as api

# Download a pre-trained model (e.g., GloVe 50-dimensional)
# model_name = "glove-wiki-gigaword-50"
# model = api.load(model_name)

# For a larger model, this might take time and memory
# You can also load a local model if you have one
# model = gensim.models.KeyedVectors.load_word2vec_format('path/to/your/model.bin', binary=True)

# For demonstration, let's create a dummy model
from gensim.models import Word2Vec
sentences = [["hello", "world"], ["this", "is", "a", "test"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save to CSV format
output_filepath = "en_embeddings.csv"
with open(output_filepath, 'w', encoding='utf-8') as f:
    for word in model.wv.index_to_key:
        vector = model.wv[word]
        f.write(f"{word} {' '.join(map(str, vector))}\n")
print(f"Saved embeddings to {output_filepath}")