
from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "duration_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python src/train_model.py first."
    )

model = joblib.load(MODEL_PATH)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Estimate task duration using the trained model."
    )
    parser.add_argument("--task", help="Task name to estimate.")
    parser.add_argument("--role", help="Resource role for the task.")
    return parser.parse_args()


args = parse_arguments()

if args.task is None:
    task_name = input("Enter task name: ").strip()
else:
    task_name = args.task.strip()

if args.role is None:
    resource_role = input("Enter resource role: ").strip()
else:
    resource_role = args.role.strip()

if not task_name or not resource_role:
    raise ValueError(
        "Task name and resource role cannot be empty. "
        "Use --task and --role or provide both interactive values."
    )

sample = pd.DataFrame([{
    "task_name": task_name,
    "resource_role": resource_role,
    "task_role": f"role {resource_role} task {task_name}"
}])

duration_minutes = max(0, float(model.predict(sample)[0]))

print(f"\nTask: {task_name}")
print(f"Resource role: {resource_role}")
print(f"Estimated duration: {duration_minutes:.1f} minutes")
print(f"Estimated duration: {duration_minutes / 60:.2f} hours")
