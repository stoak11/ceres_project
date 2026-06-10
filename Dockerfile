FROM python:3.10.6-slim

WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY ceres_package/ ceres_package/

# Lancer le serveur Uvicorn
CMD uvicorn ceres_package.api.fast:app --host 0.0.0.0 --port $PORT --reload
