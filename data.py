import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Data rating kuliner dari beberapa pengguna
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    'kuliner': [
        'Tiga Ceret', 'RM Oh La Vita', 'Mang Engking Solo',
        'Tiga Ceret', 'Level One Solo', 'Par Four Cafe',
        'Epice Restaurant', 'Tirai Bamboe Restaurant', 'Palm Ethnic Resto',
        'Alama Resto', 'Tiga Ceret', 'Canting Londo Kitchen',
        'RM Oh La Vita', 'Tirai Bamboe Restaurant', 'Alama Resto'
    ],
    'rating': [5, 4, 3, 4, 5, 2, 4, 5, 4, 3, 4, 5, 5, 4, 3]
} 

# Membuat DataFrame dari data rating
df = pd.DataFrame(data)

# Membuat pivot table
pivot_table = df.pivot_table(index='user_id', columns='kuliner', values='rating').fillna(0)

# Menghitung kesamaan antar pengguna
cosine_sim = cosine_similarity(pivot_table)

# Membuat model Nearest Neighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(pivot_table)

# Fungsi untuk memberikan rekomendasi berdasarkan nama restoran
def recommend_by_restaurant(user_id, restaurant_name, n_recommendations):
    if restaurant_name not in pivot_table.columns:
        return "Restoran tidak ditemukan dalam data"
    
    distances, indices = model_knn.kneighbors(pivot_table.loc[user_id, :].values.reshape(1, -1), n_neighbors=len(pivot_table))
    similar_users = indices.flatten()[1:]
    
    restaurant_ratings = pivot_table[restaurant_name]
    
    recommendations = []
    for idx in similar_users:
        similar_user_id = pivot_table.index[idx]
        if pivot_table.loc[similar_user_id, restaurant_name] > 0:
            recommendations.append(similar_user_id)
            if len(recommendations) == n_recommendations:
                break
    
    return recommendations

# Mendapatkan input dari pengguna
user_id = int(input("Masukkan user_id: "))
restaurant_name = input("Masukkan nama restoran: ")
n_recommendations = int(input("Masukkan jumlah rekomendasi yang diinginkan: "))

# Mendapatkan rekomendasi
recommendations = recommend_by_restaurant(user_id, restaurant_name, n_recommendations)
print(f"Rekomendasi pengguna untuk restoran '{restaurant_name}': {recommendations}")
