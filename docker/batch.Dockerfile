FROM python:3.12-slim

# Runtime dependencies for the Training/Deployment/Monitoring/Traffic pipelines when
# they run as AWS Batch tasks. Versions are pinned to match pyproject.toml/uv.lock so
# remote runs behave the same as local ones.
RUN pip install --no-cache-dir \
    pandas==2.3.3 \
    numpy==2.0.2 \
    scikit-learn==1.6.1 \
    keras==3.11.3 \
    tensorflow==2.18.1 \
    mlflow==3.14.0 \
    evidently==0.7.21 \
    boto3==1.43.1 \
    pyyaml==6.0.3 \
    psutil==7.2.2
