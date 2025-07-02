# ENCRYPTIX-TASK-MOVIE-RECOMMENDOR
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'title': [
        'The Matrix', 'Inception', 'Interstellar',
        'The Notebook', 'Titanic', 'The Conjuring',
        'Annabelle', 'John Wick', 'Avengers: Endgame'
    ],
    'description': [
        'sci-fi action hacker virtual reality',
        'sci-fi thriller dream reality time',
        'sci-fi space time gravity drama',
        'romance drama love story',
        'romance tragedy ship iceberg',
        'horror haunted house demons exorcism',
        'horror doll evil haunted',
        'action revenge assassin dog',
        'action superhero marvel infinity war'
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description'])

similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend(title, df, similarity_matrix):
    if title not in df['title'].values:
        print("Movie not found!")
        return

    idx = df[df['title'] == title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Because you liked '{title}', you might also like:")
    for i, score in similarity_scores[1:4]:
        print(f" - {df.iloc[i]['title']}")

if __name__ == "__main__":
    print("Simple Movie Recommender System")
    user_movie = input("Enter a movie you like: ")
    recommend(user_movie.strip(), df, similarity_matrix)
