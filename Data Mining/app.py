import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from pyexpat import model
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib


app = Flask(__name__)
df = pd.read_csv('user_behavior_dataset.csv')

# Data Preprocessing for Association Route
def preprocess_data(data):
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace(r'\(.*?\)', '', regex=True).str.strip('_')
    data = data.drop(columns=['user_id', 'user_behavior_class'], errors='ignore')
    return data

# data = preprocess_data(df)
# Code ini kalau dijalanin bakal ngerusak codeku

# Preprocessing data and training model
def train_model():
    # Select relevant columns
    columns_to_use = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                      'Battery Drain (mAh/day)', 'Data Usage (MB/day)',
                      'Number of Apps Installed', 'Age']  # You can change or add more columns as needed

    # Features (X) and target (y)
    X = df[columns_to_use]
    y = df['Battery Drain (mAh/day)']  # Assuming we're predicting battery drain

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'model.pkl')
    print("Model has been saved as 'model.pkl'")

# Train the model when the app starts
train_model()

@app.route('/layout')
def layout():
    return render_template('layout.html')

@app.route('/')
def home():
    print("Flask app running...")
    return render_template('index.html', title="Home", header="Welcome to Flask")

@app.route('/about')
def about():
    print("Flask app running...")
    return render_template('base.html', title="About", header="About Flask")

@app.route('/cluster')
def cluster():
    # Nentuin kolom yang perlu dinormalisasi min max
    columns_to_transform = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                            'Battery Drain (mAh/day)', 'Data Usage (MB/day)',
                            'Number of Apps Installed', 'Age']

    # Min-max normalisasi dari skala 1-10
    scaler = MinMaxScaler(feature_range=(1, 10))
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df[columns_to_transform]),
        columns=columns_to_transform
    )

    # Mencari jumlah cluster optimal pakai elbow method
    sse = []  # Sum of Squared Errors
    max_clusters = 10

    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_normalized)
        sse.append(kmeans.inertia_)

    # Plotting Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1, 1), sse, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)

    # Simpen plot ke objek BytesIO dan encode ke base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # K-means clustering dengan 5 cluster
    optimal_clusters = 5
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df_normalized['Cluster'] = kmeans.fit_predict(df_normalized)

    # Evaluasi pakai silhouette score
    silhouette_avg = silhouette_score(df_normalized, df_normalized['Cluster'])

    # Reduksi dimensi dengan pca untuk visualisasi
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_normalized)
    df_normalized['PCA1'], df_normalized['PCA2'] = X_pca[:, 0], X_pca[:, 1]

    # Visualisasi clustering pakai scatter plot
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df_normalized['PCA1'], df_normalized['PCA2'],
                          c=df_normalized['Cluster'], cmap='viridis', s=50)
    plt.title('K-Means Clustering Results')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid(True)

    # Simpen scatter plot ke objek BytesIO dan encode ke base64
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    cluster_plot_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    # Menyusun hasil evaluasi dan plot untuk dikirim ke template
    evaluation = {
        'Silhouette Score': silhouette_avg,
        'SSE': sse
    }

    return render_template('cluster.html',
                           title="Clustering",
                           header="Clustering",
                           plot_url=plot_url,
                           cluster_plot_url=cluster_plot_url,
                           evaluation=evaluation,
                           optimal_clusters=optimal_clusters)


# @app.route('/classification')
#bentar ya ges menyusul, msh revisi

# @app.route('/association')
# def association():
#     # Generate Pie Chart
#     pie_img = io.BytesIO()
#     behavior_counts = data['user_behavior_class'].value_counts()
#     plt.figure(figsize=(6, 6))
#     plt.pie(behavior_counts, labels=behavior_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
#     plt.title('Distribution of User Behavior Classes')
#     plt.tight_layout()
#     plt.savefig(pie_img, format='png')
#     pie_img.seek(0)
#     pie_chart_url = base64.b64encode(pie_img.getvalue()).decode('utf8')
#
#     # Generate Scatter Plot
#     scatter_img = io.BytesIO()
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='app_usage_time', y='screen_on_time', hue='gender', data=data, palette='coolwarm')
#     plt.title('App Usage Time vs Screen On Time')
#     plt.tight_layout()
#     plt.savefig(scatter_img, format='png')
#     scatter_img.seek(0)
#     scatter_plot_url = base64.b64encode(scatter_img.getvalue()).decode('utf8')
#
#     # Generate Correlation Heatmap
#     heatmap_img = io.BytesIO()
#     plt.figure(figsize=(10, 8))
#     corr = data.corr()
#     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
#     plt.title('Correlation Heatmap')
#     plt.tight_layout()
#     plt.savefig(heatmap_img, format='png')
#     heatmap_img.seek(0)
#     heatmap_url = base64.b64encode(heatmap_img.getvalue()).decode('utf8')
#
#     return render_template('association.html',
#                            title="Association Analysis",
#                            header="Analisis Hubungan Perangkat dan Pengguna",
#                            pie_chart_url=pie_chart_url,
#                            scatter_plot_url=scatter_plot_url,
#                            heatmap_url=heatmap_url)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    if request.method == 'POST':
        try:
            # Mengambil input dari form, tanpa input untuk battery drain
            app_usage_time = float(request.form['app_usage_time'])
            screen_on_time = float(request.form['screen_on_time'])
            data_usage = float(request.form['data_usage'])
            num_apps_installed = int(request.form['num_apps_installed'])
            age = int(request.form['age'])

            # Preprocess input menjadi data yang sesuai
            input_data = pd.DataFrame({
                'App Usage Time (min/day)': [app_usage_time],
                'Screen On Time (hours/day)': [screen_on_time],
                'Battery Drain (mAh/day)': [0],  # Menambahkan nilai default untuk fitur yang hilang
                'Data Usage (MB/day)': [data_usage],
                'Number of Apps Installed': [num_apps_installed],
                'Age': [age]
            })

            # Load model yang sudah disimpan
            model = joblib.load('model.pkl')

            # Prediksi konsumsi baterai (battery drain)
            prediction = model.predict(input_data)

            # Tampilkan hasil prediksi
            first_prediction = round(prediction[0], 2)

            return render_template(
                'prediksi.html',
                title="Prediksi Konsumsi Baterai",
                header="Prediksi Konsumsi Baterai",
                prediction=first_prediction
            )
        except Exception as e:
            return f"Error occurred: {str(e)}", 400
    return render_template('prediksi.html', title="Prediksi Konsumsi Baterai", header="Prediksi Konsumsi Baterai")


if __name__ == '__main__':
    app.run(debug=True)
