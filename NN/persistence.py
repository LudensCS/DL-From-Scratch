import pickle

from .architecture import MultiLayerNet, TwoLayerNet


def load_model(filepath: str) -> TwoLayerNet | MultiLayerNet:
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def save_model(nn: TwoLayerNet | MultiLayerNet, filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(nn, f)
