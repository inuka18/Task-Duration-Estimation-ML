
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "duration_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python src/train_model.py first."
    )

model = joblib.load(MODEL_PATH)

task_name = input("Enter task name: ").strip()
resource_role = input("Enter resource role: ").strip()

if not task_name or not resource_role:
    raise ValueError("Task name and resource role cannot be empty.")

sample = pd.DataFrame([{
    "task_name": task_name,
    "resource_role": resource_role
}])

duration_minutes = max(0, float(model.predict(sample)[0]))

print(f"\nTask: {task_name}")
print(f"Resource role: {resource_role}")
print(f"Estimated duration: {duration_minutes:.1f} minutes")
print(f"Estimated duration: {duration_minutes / 60:.2f} hours")
