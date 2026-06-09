import os
import time
import pickle
import glob
from google.cloud import storage
from colorama import Fore, Style
from tensorflow import keras
import joblib


from ceres_package.params import LOCAL_REGISTRY_PATH, MODEL_TARGET, BUCKET_NAME


def save_results(params: dict, metrics: dict) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --- Local ---
    if params is not None:
        params_dir = os.path.join(LOCAL_REGISTRY_PATH, "params")
        os.makedirs(params_dir, exist_ok=True)
        params_path = os.path.join(params_dir, f"{timestamp}.pickle")
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

    if metrics is not None:
        metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"{timestamp}.pickle")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)

    print("✅ Results saved locally")

    # --- GCS ---
    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        if params is not None:
            blob = bucket.blob(f"results/params/{timestamp}.pickle")
            blob.upload_from_string(pickle.dumps(params))

        if metrics is not None:
            blob = bucket.blob(f"results/metrics/{timestamp}.pickle")
            blob.upload_from_string(pickle.dumps(metrics))

        print("✅ Results saved to GCS")


def save_model(model=None) -> None:
    """
    Sauvegarde le modèle localement dans :
    - Keras : "{LOCAL_REGISTRY_PATH}/models/{timestamp}.keras"
    - ML (XGBoost, sklearn...) : "{LOCAL_REGISTRY_PATH}/models/{timestamp}.pkl"
    Si MODEL_TARGET='gcs', aussi dans le bucket GCS.
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    models_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Détection du type de modèle
    if isinstance(model, keras.Model):
        model_path = os.path.join(models_dir, f"{timestamp}.keras")
        model.save(model_path)
    else:
        model_path = os.path.join(models_dir, f"{timestamp}.pkl")
        joblib.dump(model, model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        model_filename = os.path.basename(model_path)
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)
        print("✅ Model saved to GCS")

    return None



def load_model():
    """
    Charge le modèle le plus récent :
    - depuis le disque local si MODEL_TARGET='local'
    - depuis GCS si MODEL_TARGET='gcs'
    Retourne None si aucun modèle trouvé.
    Supporte les modèles Keras (.keras) et ML (.pkl).
    """

    def _load_file(path):
        if path.endswith(".keras"):
            return keras.models.load_model(path)
        else:
            return joblib.load(path)

    if MODEL_TARGET == "local":
        print(Fore.BLUE + "\nChargement du dernier modèle local..." + Style.RESET_ALL)

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*.keras") + \
                            glob.glob(f"{local_model_directory}/*.pkl")

        if not local_model_paths:
            print("❌ Aucun modèle trouvé en local")
            return None

        most_recent_model_path = sorted(local_model_paths)[-1]
        latest_model = _load_file(most_recent_model_path)
        print("✅ Modèle chargé depuis le disque local")
        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + "\nChargement du dernier modèle depuis GCS..." + Style.RESET_ALL)

        try:
            client = storage.Client()
            blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))
            valid_blobs = [b for b in blobs if b.name.endswith(".keras") or b.name.endswith(".pkl")]

            if not valid_blobs:
                print(f"❌ Aucun modèle trouvé dans GCS bucket {BUCKET_NAME}")
                return None

            latest_blob = max(valid_blobs, key=lambda x: x.updated)

            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
            os.makedirs(local_model_directory, exist_ok=True)
            local_path = os.path.join(local_model_directory, os.path.basename(latest_blob.name))
            latest_blob.download_to_filename(local_path)

            latest_model = _load_file(local_path)
            print("✅ Modèle chargé depuis GCS")
            return latest_model

        except Exception as e:
            print(f"❌ Erreur lors du chargement depuis GCS : {e}")
            return None
