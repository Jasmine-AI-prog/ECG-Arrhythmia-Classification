
import numpy as np
import tensorflow as tf
import joblib

class Predictor:
    def __init__(self, model_path='ecg_model.keras', scaler_path='scaler.pkl', config_path='config.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.config = joblib.load(config_path)
        self.segment_length = self.config['segment_length']
        self.class_names = self.config['classes']

    def predict(self, signal):
        if len(signal) != self.segment_length:
            if len(signal) > self.segment_length:
                signal = signal[:self.segment_length]
            else:
                signal = np.pad(signal, (0, self.segment_length - len(signal)), 'constant')

        signal_norm = self.scaler.transform(signal.reshape(1, -1))
        signal_reshaped = signal_norm.reshape(1, self.segment_length, 1)
        probs = self.model.predict(signal_reshaped, verbose=0)
        classe = np.argmax(probs)
        return int(classe), self.class_names[classe], probs[0].tolist()

if __name__ == "__main__":
    predictor = Predictor()
    print("Predictor ready.")
    test_signal = np.random.randn(187)
    classe, nom, probs = predictor.predict(test_signal)
    print(f"Test prediction: {nom}")
