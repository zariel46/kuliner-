from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load data
df = pd.read_csv('data.csv')

# Membuat pivot table
pivot_table = df.pivot_table(index='user_id', columns='kuliner', values='rating').fillna(0)

# Membuat model Nearest Neighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(pivot_table)

# Fungsi untuk memberikan rekomendasi berdasarkan nama restoran
def recommend_by_restaurant(user_id, restaurant_name, n_recommendations):
    if restaurant_name not in pivot_table.columns:
        return ["Restoran tidak ditemukan dalam data"]
    
    distances, indices = model_knn.kneighbors(pivot_table.loc[user_id, :].values.reshape(1, -1), n_neighbors=len(pivot_table))
    similar_users = indices.flatten()[1:]
    
    recommendations = []
    for idx in similar_users:
        similar_user_id = pivot_table.index[idx]
        if pivot_table.loc[similar_user_id, restaurant_name] > 0:
            recommendations.append(similar_user_id)
            if len(recommendations) == n_recommendations:
                break
    
    if not recommendations:
        return ["Tidak ada rekomendasi pengguna yang ditemukan"]
    
    return recommendations

@app.route('/')
def index():
    return render_template('Rekomendasi-tambah.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    restaurant_name = request.form['restaurant_name']
    n_recommendations = int(request.form['n_recommendations'])
    
    recommendations = recommend_by_restaurant(user_id, restaurant_name, n_recommendations)
    return render_template('Rekomendasi-tambah.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
