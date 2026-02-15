"""Assignment 1.11: MLflow with SQLite Backend.

Verify that metadata goes to SQLite and artifacts to filesystem.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

# Connect to the server
mlflow.set_tracking_uri("http://127.0.0.1:5500")

# Create experiment
experiment_name = "sqlite-backend-test"
mlflow.set_experiment(experiment_name)

print("="*60)
print("Starting MLflow run with SQLite backend")
print("="*60)

# Start run
with mlflow.start_run(run_name="backend-verification") as run:
    print(f"\n✓ Run ID: {run.info.run_id}")

    # Log parameters
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    print("✓ Parameters logged to SQLite database")

    # Log metrics
    mlflow.log_metric("accuracy", 0.96)
    mlflow.log_metric("precision", 0.94)
    mlflow.log_metric("recall", 0.95)
    mlflow.log_metric("f1_score", 0.945)
    print("✓ Metrics logged to SQLite database")

    # Create and log artifact (plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    ax.plot(x, y1, label="sin(x)", linewidth=2)
    ax.plot(x, y2, label="cos(x)", linewidth=2)
    ax.set_title("Sample Artifact: Trigonometric Functions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    # Save and log
    artifact_file = Path("trig_plot.png")
    plt.savefig(artifact_file)
    plt.close()
    mlflow.log_artifact(artifact_file)
    artifact_file.unlink()
    print("✓ Artifact logged to filesystem (mlartifacts/)")

    # Create text artifact
    info_file = Path("model_info.txt")
    with info_file.open("w") as f:
        f.write("Model Training Report\n")
        f.write("="*50 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Run ID: {run.info.run_id}\n")
        f.write("Model: Random Forest\n")
        f.write("Accuracy: 96%\n")
        f.write("\nThis demonstrates SQLite backend storage.\n")

    mlflow.log_artifact(info_file)
    info_file.unlink()
    print("✓ Text artifact logged to filesystem")

    print("\n" + "="*60)
    print("Run completed successfully!")
    print("="*60)


# Verification summary
print("\n" + "="*60)
print("VERIFICATION INSTRUCTIONS")
print("="*60)
print("\n1. Check SQLite database:")
print("   ls -lh mlflow.db")
print("\n2. Check artifacts directory:")
print("   ls -R mlartifacts/")
print("\n3. Query database:")
print("   sqlite3 mlflow.db 'SELECT * FROM experiments;'")
print("\n4. View in UI:")
print("   http://127.0.0.1:5500")
print("="*60)
