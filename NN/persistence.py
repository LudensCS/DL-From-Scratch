import pickle

from .architecture import MutiLayerNet, TwoLayerNet


def load_model(filepath: str) -> TwoLayerNet | MutiLayerNet:
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def save_model(nn: TwoLayerNet | MutiLayerNet, filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(nn, f)
