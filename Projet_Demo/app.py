import os
import joblib
import pandas as pd
import numpy as np
import re
from flask import Flask, render_template, request, jsonify

# --- Configuration et Chargement du Modèle ---

app = Flask(__name__)

# Assurez-vous d'AJUSTER ce chemin pour qu'il corresponde à l'endroit où Flask est exécuté 
# et où se trouve votre dossier 'model'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.joblib')

try:
    # Charger le modèle Random Forest entraîné
    MODEL = joblib.load(MODEL_PATH)
    print(f"Modèle chargé avec succès depuis: {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERREUR: Fichier modèle non trouvé à {MODEL_PATH}. Assurez-vous qu'il est dans le dossier 'model/'.")
    MODEL = None

# Définitions des constantes (Doivent correspondre à l'entraînement !)
ALPHABET_SIZES = {'lower': 26, 'upper': 26, 'digit': 10, 'symbol': 32}
LABELS = {0: "Weak (Faible) 🔴", 1: "Medium (Moyen) 🟡", 2: "Strong (Fort) 🟢"}
COMMON_PATTERNS = [
    '123456', '234567', '345678', '456789', '654321', '54321', '4321', '321',
    '111111', '222222', '000000', 
    'qwerty', 'azerty', 'qwert', 'asdfg', 'zxcvb', 'mnbvc',
    'password', 'admin', 'iloveyou', 'p@ssword', 'user', 'princess', 'rockyou' 
]
PATTERN_REGEX = '(?i)' + '|'.join(COMMON_PATTERNS)
REPETITION_REGEX = r'(.)\1{2,}'
FEATURE_COLS = ['length', 'count_lower', 'count_upper', 'count_digit', 'count_symbol', 
                'char_classes', 'shannon_entropy', 'is_common_pattern']


# --- Fonctions de Feature Engineering (Réplicat de l'entraînement) ---

def calculate_features(password):
    """Calcule toutes les 8 caractéristiques pour un seul mot de passe."""
    df_new = pd.DataFrame({'password': [password]})
    
    # 1. Compteurs
    df_new['count_lower'] = df_new['password'].str.count(r'[a-z]')
    df_new['count_upper'] = df_new['password'].str.count(r'[A-Z]')
    df_new['count_digit'] = df_new['password'].str.count(r'[0-9]')
    df_new['count_symbol'] = df_new['password'].str.count(r'[^a-zA-Z0-9]') 

    # 2. Booléens et Classes
    df_new['has_lower'] = np.where(df_new['count_lower'] > 0, 1, 0)
    df_new['has_upper'] = np.where(df_new['count_upper'] > 0, 1, 0)
    df_new['has_digit'] = np.where(df_new['count_digit'] > 0, 1, 0)
    df_new['has_symbol'] = np.where(df_new['count_symbol'] > 0, 1, 0)
    df_new['length'] = df_new['password'].str.len()
    df_new['char_classes'] = df_new['has_upper'] + df_new['has_lower'] + df_new['has_digit'] + df_new['has_symbol']

    # 3. Entropie
    df_new['alphabet_size'] = (df_new['has_lower'] * ALPHABET_SIZES['lower']) + (df_new['has_upper'] * ALPHABET_SIZES['upper']) + (df_new['has_digit'] * ALPHABET_SIZES['digit']) + (df_new['has_symbol'] * ALPHABET_SIZES['symbol'])
    df_new['shannon_entropy'] = np.where(df_new['alphabet_size'] > 1, df_new['length'] * np.log2(df_new['alphabet_size']), 0.0)

    # 4. Pattern
    is_listed_pattern = df_new['password'].str.contains(PATTERN_REGEX, regex=True, na=False)
    is_repetition = df_new['password'].str.contains(REPETITION_REGEX, regex=True, na=False)
    df_new['is_common_pattern'] = np.where(np.logical_or(is_listed_pattern, is_repetition), 1, 0)

    return df_new[FEATURE_COLS]


# --- Routes Flask ---

@app.route('/')
def index():
    """Route principale pour afficher l'interface."""
    return render_template('index.html')

@app.route('/predict_strength', methods=['POST'])
def predict():
    """API pour recevoir le mot de passe et renvoyer la prédiction."""
    if not MODEL:
        return jsonify({'error': 'Model not loaded.'}), 500

    data = request.get_json(silent=True)
    password = data.get('password', '')

    if not password:
        return jsonify({'strength': 'Enter a password.'})

    # 1. Calcul des features
    X_predict = calculate_features(password)
    
    # 2. Prédiction
    prediction_class = MODEL.predict(X_predict)[0]
    predicted_label = LABELS[prediction_class]
    
    # 3. Retourner les résultats
    return jsonify({
        'strength': predicted_label,
        'entropy': f"{X_predict['shannon_entropy'].iloc[0]:.2f}",
        'classes': f"{X_predict['char_classes'].iloc[0]}"
    })


if __name__ == '__main__':
    # Lancez Flask en mode debug
    # Pour un environnement de production, utilisez un serveur WSGI comme Gunicorn
    app.run(debug=True)