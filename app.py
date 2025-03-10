from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime
from wordcloud import WordCloud
from mlflow.pyfunc import load_model
import mlflow
import mlflow.sklearn
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['IMAGE_FOLDER'] = 'static/images'

# Global Variables
MLFLOW_TRACKING_URI = "https://mlflow.cospl.my.id"

@app.route('/')
def index():
    return render_template('index.html', columns=[])

@app.route('/upload', methods=['POST'])
def upload():
    if 'dataset' not in request.files:
        return "File dataset tidak ditemukan!", 400
    
    file = request.files['dataset']
    if file.filename == '':
        return "Nama file tidak valid!", 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Baca dataset untuk mendapatkan kolom
    df = pd.read_csv(file_path)
    columns = df.columns.tolist()
    
    return render_template('index.html', columns=columns)

@app.route('/train', methods=['POST'])
def train():
    if 'dataset' not in request.files:
        return "File dataset tidak ditemukan!", 400
    
    file = request.files['dataset']
    if file.filename == '':
        return "Nama file tidak valid!", 400
    
    # Ambil data dari form
    x_col = request.form['x_col']
    y_col = request.form['y_col']
    sampling_method = request.form['sampling_method']
    experiment_name = request.form['experiment_name']
    test_size = float(request.form['test_size'])
    random_state = int(request.form['random_state'])
    max_features = int(request.form['max_features'])
    ngram_range = tuple(map(int, request.form['ngram_range'].split(',')))
    
    # Parameter algoritma
    knn_neighbors = int(request.form['knn_neighbors'])
    knn_metric = request.form['knn_metric']
    knn_weights = request.form['knn_weights']
    
    svm_c = float(request.form['svm_c'])
    svm_gamma = request.form['svm_gamma']
    svm_kernel = request.form['svm_kernel']  # Ambil kernel dari form
    
    nb_alpha = float(request.form['nb_alpha'])
    
    lr_max_iter = int(request.form['lr_max_iter'])
    lr_c = float(request.form['lr_c'])
    lr_solver = request.form['lr_solver']
    
    rf_n_estimators = int(request.form['rf_n_estimators'])
    rf_max_depth = int(request.form['rf_max_depth'])
    
    # Update model dengan parameter dari form
    MODELS = {
        "KNN": KNeighborsClassifier(n_neighbors=knn_neighbors, metric=knn_metric, weights=knn_weights),
        "SVM": SVC(kernel=svm_kernel, C=svm_c, gamma=svm_gamma, probability=True),  # Gunakan kernel yang dipilih
        "Naive Bayes": MultinomialNB(alpha=nb_alpha),
        "Logistic Regression": LogisticRegression(max_iter=lr_max_iter, C=lr_c, solver=lr_solver),
        "Random Forest": RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state)
    }
    
    # Simpan dataset
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Load dataset
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    
    # Pastikan kolom yang digunakan benar
    X = df[x_col]
    y = df[y_col]
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    mlflow.end_run()
    
    # Visualisasi distribusi label sebelum penanganan imbalance
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette='viridis')
    plt.title("Distribusi Label Sebelum Penanganan Imbalance")
    before_sampling_path = os.path.join(app.config['IMAGE_FOLDER'], 'before_sampling.png')
    plt.savefig(before_sampling_path)
    plt.close()
    
    # Preprocessing teks menggunakan TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Generate Word Cloud dari TF-IDF
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), np.asarray(X_tfidf.sum(axis=0)).ravel()))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_scores)
    
    # Simpan Word Cloud sebagai gambar
    wordcloud_path = os.path.join(app.config['IMAGE_FOLDER'], 'wordcloud.png')
    wordcloud.to_file(wordcloud_path)

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=test_size, random_state=random_state)
    
    # Pilihan untuk menangani data imbalance
    if sampling_method == "smote":
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif sampling_method == "oversample":
        oversampler = RandomOverSampler(random_state=random_state)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
    elif sampling_method == "undersample":
        undersampler = RandomUnderSampler(random_state=random_state)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
    
    # Visualisasi distribusi label setelah penanganan imbalance
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train, palette='viridis')
    plt.title("Distribusi Label Setelah Penanganan Imbalance")
    after_sampling_path = os.path.join(app.config['IMAGE_FOLDER'], 'after_sampling.png')
    plt.savefig(after_sampling_path)
    plt.close()
    
    # Store results
    results = {}
    
    # Train and evaluate models
    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Hitung precision, recall, dan f1-score (weighted)
            precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Generate classification report
            clf_report = classification_report(y_test, y_pred, zero_division=0)
            
            # Save confusion matrix
            plt.figure(figsize=(8, 6))  # Ukuran gambar lebih besar
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', linewidths=0.5)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            conf_matrix_path = os.path.join(app.config['IMAGE_FOLDER'], f'conf_matrix_{name}.png')
            plt.savefig(conf_matrix_path)
            plt.close()
            
            # Simpan hasil untuk ditampilkan di web
            results[name] = {
                'accuracy': acc,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'clf_report': clf_report,  # Tambahkan classification report
                'conf_matrix': f'conf_matrix_{name}.png'
            }
            
            # Log metrics and model ke MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", precision_macro)
            mlflow.log_metric("recall_macro", recall_macro)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("precision_weighted", precision_weighted)
            mlflow.log_metric("recall_weighted", recall_weighted)
            mlflow.log_metric("f1_weighted", f1_weighted)
            mlflow.sklearn.log_model(model, name)
            mlflow.log_artifact(conf_matrix_path)
    
    # Heatmap evaluasi model (macro avg)
    metrics_macro_df = pd.DataFrame.from_dict({
        name: [results[name]['accuracy'], results[name]['precision_macro'], results[name]['recall_macro'], results[name]['f1_macro']]
        for name in results
    }, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_macro_df, annot=True, fmt='.4f', cmap='coolwarm', linewidths=0.5)
    plt.title('Model Evaluation Metrics Heatmap (Macro Avg)')
    heatmap_macro_path = os.path.join(app.config['IMAGE_FOLDER'], 'heatmap_macro.png')
    plt.savefig(heatmap_macro_path)
    plt.close()
    mlflow.log_artifact(heatmap_macro_path)
    
    # Heatmap evaluasi model (weighted avg)
    metrics_weighted_df = pd.DataFrame.from_dict({
        name: [results[name]['accuracy'], results[name]['precision_weighted'], results[name]['recall_weighted'], results[name]['f1_weighted']]
        for name in results
    }, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_weighted_df, annot=True, fmt='.4f', cmap='coolwarm', linewidths=0.5)
    plt.title('Model Evaluation Metrics Heatmap (Weighted Avg)')
    heatmap_weighted_path = os.path.join(app.config['IMAGE_FOLDER'], 'heatmap_weighted.png')
    plt.savefig(heatmap_weighted_path)
    plt.close()
    mlflow.log_artifact(heatmap_weighted_path)
    
    # Log dataset dan visualisasi ke MLflow
    mlflow.log_artifact(file_path)
    mlflow.log_artifact(before_sampling_path)
    mlflow.log_artifact(after_sampling_path)
    mlflow.sklearn.log_model(vectorizer, "TF-IDF_Vectorizer")
    mlflow.log_artifact(wordcloud_path)
    
    # Redirect ke halaman hasil dengan parameter yang digunakan
    return redirect(url_for('results', results=results, parameters={
        'x_col': x_col,
        'y_col': y_col,
        'sampling_method': sampling_method,
        'experiment_name': experiment_name,
        'test_size': test_size,
        'random_state': random_state,
        'max_features': max_features,
        'ngram_range': ngram_range,
        'knn_neighbors': knn_neighbors,
        'knn_metric': knn_metric,
        'knn_weights': knn_weights,
        'svm_c': svm_c,
        'svm_gamma': svm_gamma,
        'svm_kernel': svm_kernel,  # Tambahkan kernel SVM
        'nb_alpha': nb_alpha,
        'lr_max_iter': lr_max_iter,
        'lr_c': lr_c,
        'lr_solver': lr_solver,
        'rf_n_estimators': rf_n_estimators,
        'rf_max_depth': rf_max_depth
    }))

@app.route('/results')
def results():
    results = request.args.get('results')
    parameters = request.args.get('parameters')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('results.html', results=eval(results), parameters=eval(parameters), timestamp=timestamp)

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)


@app.route('/deploy', methods=['GET', 'POST'])
def deploy():
    if request.method == 'POST':
        # Ambil URI model dari form
        tfidf_model_uri = convert_mlflow_artifact_uri_to_url(request.form['tfidf_model_url'], MLFLOW_TRACKING_URI)
        algorithm_model_uri = convert_mlflow_artifact_uri_to_url(request.form['algorithm_model_url'], MLFLOW_TRACKING_URI)
        
        try:
            # Langkah 1: Memuat Model Algoritma (SVM, dll)
            print("Mengunduh dan memuat model algoritma...")
            model_path = mlflow.artifacts.download_artifacts(algorithm_model_uri)

            # Periksa apakah path adalah direktori
            if os.path.isdir(model_path):
                print("Model algoritma disimpan sebagai direktori. Memuat menggunakan mlflow.sklearn.load_model...")
                algorithm_model = mlflow.sklearn.load_model(model_path)
                print("Model algoritma berhasil dimuat.")
            else:
                print("Model algoritma disimpan sebagai file tunggal. Memuat menggunakan pickle...")
                with open(model_path, "rb") as f:
                    algorithm_model = pickle.load(f)
                print("Model algoritma berhasil dimuat.")

            # Langkah 2: Memuat TF-IDF Vectorizer
            print("Mengunduh dan memuat TF-IDF Vectorizer...")
            tfidf_vectorizer_path = mlflow.artifacts.download_artifacts(tfidf_model_uri)

            # Periksa apakah path adalah direktori atau file
            if os.path.isdir(tfidf_vectorizer_path):
                print("TF-IDF Vectorizer disimpan sebagai direktori. Memeriksa isi direktori...")
                files = os.listdir(tfidf_vectorizer_path)
                print(f"Isi direktori: {files}")

                # Cari file yang berisi TF-IDF Vectorizer (misalnya, data.pkl)
                tfidf_file = None
                for file in files:
                    if file.endswith(".pkl"):
                        tfidf_file = os.path.join(tfidf_vectorizer_path, file)
                        break

                if tfidf_file:
                    print(f"Memuat TF-IDF Vectorizer dari file: {tfidf_file}")
                    with open(tfidf_file, "rb") as f:
                        tfidf_model = pickle.load(f)
                    print("TF-IDF Vectorizer berhasil dimuat.")
                else:
                    raise FileNotFoundError("File .pkl tidak ditemukan di direktori.")
            else:
                print("TF-IDF Vectorizer disimpan sebagai file tunggal.")
                with open(tfidf_vectorizer_path, "rb") as f:
                    tfidf_model = pickle.load(f)
                print("TF-IDF Vectorizer berhasil dimuat.")

        except Exception as e:
            return f"Error: {str(e)}", 400
        
        # Cek apakah pengguna memilih single prediction atau batch prediction
        if 'single_text' in request.form:
            # Single prediction
            text = request.form['single_text']
            try:
                # Transform teks menggunakan model TF-IDF
                print("Melakukan transformasi kalimat uji...")
                X_test = tfidf_model.transform([text])
                
                # Prediksi menggunakan model algoritma
                print("Melakukan prediksi...")
                prediction = algorithm_model.predict(X_test)
                
                # Output hasil prediksi
                print("Hasil Prediksi:", prediction)
                return render_template('deploy.html', prediction=prediction[0])
            except Exception as e:
                return f"Error during prediction: {str(e)}", 400
        
        elif 'dataset' in request.files:
            # Batch prediction
            file = request.files['dataset']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                # Baca dataset
                df = pd.read_csv(file_path)
                target_column = request.form['target_column']
                
                # Transform kolom target menggunakan model TF-IDF
                X_tfidf = tfidf_model.transform(df[target_column])
                
                # Prediksi menggunakan model algoritma
                df['predicted-label-xxx'] = algorithm_model.predict(X_tfidf)
                
                # Simpan dataset dengan hasil prediksi
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_' + file.filename)
                df.to_csv(output_path, index=False)
                
                return send_from_directory(app.config['UPLOAD_FOLDER'], 'predicted_' + file.filename, as_attachment=True)
            except Exception as e:
                return f"Error during batch prediction: {str(e)}", 400
    
    return render_template('deploy.html')

def convert_mlflow_artifact_uri_to_url(artifact_uri, mlflow_server_url):
    # Hapus prefix "mlflow-artifacts:/"
    path = artifact_uri[len("mlflow-artifacts:/"):]
    # Gabungkan dengan base URL MLflow server
    return f"{mlflow_server_url}/api/2.0/mlflow-artifacts/artifacts/{path}"

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
    app.run(debug=True)
