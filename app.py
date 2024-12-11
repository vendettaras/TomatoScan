from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

app = Flask(__name__)
model = load_model('modele_entrainee.keras')

def preprocess_image(image_path):
    # Charger l'image, la redimensionner et la prétraiter
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Assurez-vous que les dimensions correspondent à celles utilisées pendant l'entraînement
    img = np.array(img) / 255.0  # Normaliser les valeurs des pixels
    img = img.reshape(1, 224, 224, 3)  # Ajouter une dimension pour le batch
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer l'image depuis la requête POST
    image_file = request.files['image']
    if image_file:
        image_path = 'temp_image.jpg'  # Nom de fichier temporaire pour stocker l'image
        image_file.save(image_path)
        # Prétraiter l'image
        processed_image = preprocess_image(image_path)
        # Faire une prédiction avec le modèle chargé
        predictions = model.predict(processed_image)
        # Convertir predictions en une liste Python standard
        predictions_list = predictions.tolist()
        # Renvoyer les résultats
        probabilities_percentage = [[prob * 100 for prob in probs] for probs in predictions_list]
        
        # Définir les étiquettes de classe (supposons que vous avez une liste appelée "classes" contenant les noms de classe)
        classes = ["Fraîcheur", "Pourriture", "Fraîcheur moyenne", "Autres"]

        # Formater les résultats dans une chaîne de caractères lisible
        results_str = ""
        for i, probs in enumerate(probabilities_percentage):
            for j, probs in enumerate(probs):
                results_str += f"   {classes[j]}: {probs:.2f}%\n"
            results_str += "\n"

        # Renvoyer les résultats formatés
        return render_template("index.html", results_str={"results_str":results_str})
    else:
        return 'No image found', 400

if __name__ == '__main__':
    app.run(debug=True)
