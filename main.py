import pandas as pd
import numpy as np
from models.isolation_forest_model import run_iforest
from utils.evaluation import evaluate_model_success

# Read data
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y_true = df['Class']


# CLI
print("""
--------------------------------
    Anomaly Detection System

1 - Isolation Forest
2 - ...

0 - Exit System
--------------------------------
""")
try:
    while True:

        print("Please select a model number:")
        try:
            selection = input()

        except ValueError:
            print("Please enter a valid number.")

        if selection == "0":
            print("Exiting system...")
            break

        elif selection == "1":
            print("→ Isolation Forest Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred, scores, clf = run_iforest(X.values)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                evaluate_model_success(y_true, y_pred)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

            print("""
            
            """)

        elif selection == "2":
            ...

        else:
            print("Invalid selection. Please try again.")


except KeyboardInterrupt:
    print("\n🔴 Process interrupted by user. Exiting safely...")