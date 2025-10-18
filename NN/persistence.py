import pickle

from .architecture import NeuralNetwork


def load_model(filepath: str) -> NeuralNetwork:
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def save_model(nn: NeuralNetwork, filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(nn, f)
