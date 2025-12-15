## CI/CD & Model Deployment

Workflow CI menggunakan GitHub Actions:
- Training model otomatis menggunakan MLflow Project
- Logging artefak ke MLflow (DagsHub)
- Registrasi model ke MLflow Model Registry
- Build Docker Image menggunakan `mlflow models build-docker`
- Push Docker Image ke Docker Hub

Docker Image:
- Repository: https://hub.docker.com/repository/docker/zakialfadilah/loan-prediction-mlflow/
