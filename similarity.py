import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.metrics import pairwise_distances
import numpy as np
from thefuzz import process

def generate_embeddings(movies_df_cleaned):
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer('all-MiniLM-L6-v2')

    combined_features = movies_df_cleaned['genres'].astype(str) + " " + \
                        movies_df_cleaned['keywords'].astype(str)
    
    movie_vectors = model.encode(combined_features.tolist(), show_progress_bar=True)

    data_enc = np.array(movie_vectors).astype('float32')
    dimension = data_enc.shape[1]

    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(data_enc)
    index.add(data_enc)

    faiss.write_index(index, 'data/movie_index.faiss')
    np.save('data/encoded_data.npy', data_enc)

def key_similarity(data_enc, index, movies_df_cleaned, title, k=10):
    try:
        title_idx = (movies_df_cleaned['title'] == title).idxmax()
    except ValueError:
        print("key_similarity error: Title not found")
    vector = data_enc[title_idx:title_idx+1]

    distances,indices = index.search(vector, k+1)

    return movies_df_cleaned.iloc[indices[0][1:]]

def titleMatching(df, title):
    if(not title):
        return None
    if(df['title'].isin([title]).any()):
        return title
    result = process.extractOne(title, df['title'])
    return result[0]
    



