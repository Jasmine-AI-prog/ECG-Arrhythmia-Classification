# Classification des Arythmies Cardiaques avec CNN-LSTM-Attention

## Description du projet
Ce projet utilise un reseau de neurones profond combinant des couches CNN, LSTM bidirectionnel et un mecanisme d'attention pour classifier les battements cardiaques en 5 categories (N, S, V, F, Q) a partir de signaux ECG.

## Modeles et methodes
- Architecture: CNN-LSTM-Attention
- Focal Loss pour gerer le desequilibre des classes
- Normalisation Z-score
- Early Stopping, ReduceLROnPlateau

## Resultats
- Accuracy sur MIT-BIH: 98.56%
- Matrice de confusion et F1-score par classe disponibles

## Structure du projet
- notebooks/: Notebooks Jupyter
- models/: Modeles entraines
- src/: Code source
- results/: Resultats et graphiques

## Installation
pip install -r requirements.txt

## Utilisation
from predict import Predictor
predictor = Predictor()
classe, nom, probs = predictor.predict(signal)
