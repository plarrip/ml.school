"""
----------------------------------------------------------------------
## ASSIGNMENT #1.10
Assignment 1.10: Introduction to MLflow
Simple script to log an experiment to MLflow

### MLflow
MLflow is an open-source platform for managing the machine learning lifecycle, including:
- **Tracking**: Log parameters, metrics, and artifacts from experiments
- **Projects**: Package code in a reproducible format
- **Models**: Deploy models to various platforms
- **Registry**: Store and version models

We'll focus on **MLflow Tracking** to log our penguin classification experiments.

### Key Concepts:

- **Experiment**: A collection of related runs (e.g., "Penguin Classification")
- **Run**: A single execution of your code (e.g., one model training attempt)
- **Parameters**: Input values (e.g., learning_rate=0.01, max_depth=5)
- **Metrics**: Output measurements (e.g., accuracy=0.95, f1_score=0.92)
- **Artifacts**: Files produced (e.g., plots, models, data files)
- **Tags**: Metadata for organization (e.g., "algorithm=random_forest")

"""

import mlflow
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# SETUP: Point to your running MLflow server
# =============================================================================

# Connect to your running server at port 5050
mlflow.set_tracking_uri("http://127.0.0.1:5500")
print(f"✓ Connected to MLflow server: {mlflow.get_tracking_uri()}")

# =============================================================================
# CREATE EXPERIMENT AND LOG DATA
# =============================================================================

# Create or get experiment
experiment_name = "SantiagoML__assignment_1_10"
experiment = mlflow.set_experiment(experiment_name)
print(f"✓ Experiment: {experiment.name} (ID: {experiment.experiment_id})")

# Start a run
with mlflow.start_run(run_name="test-run-2") as run:
    print(f"✓ Started run: {run.info.run_id}")
    
    # Log parameters
    mlflow.log_param("model_type", "gxboost")
    mlflow.log_param("max_depth", 15)
    print("✓ Parameters logged")
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.75)
    mlflow.log_metric("f1_score", 0.835)
    print("✓ Metrics logged")
    
    # Create and log a simple plot artifact
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Sample Artifact: Sine Wave")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.20)
    
    # Save and log
    plt.savefig("sample_plot.png")
    plt.close()
    mlflow.log_artifact("sample_plot.png")
    print("✓ Artifact logged")
    
    # Clean up
    import os
    os.remove("sample_plot.png")
    
    print("\n" + "="*60)
    print("✓ Run completed successfully!")
    print(f"View results at: http://127.0.0.1:5500")
    print("="*60)