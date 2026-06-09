"""Ceres predict API — team package."""
from ceres_package.interface.inference import predict
from ceres_package.interface.predictor import Predictor, get_predictor

__all__ = ["Predictor", "get_predictor", "predict"]
