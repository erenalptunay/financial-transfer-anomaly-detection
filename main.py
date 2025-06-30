from models.isolation_forest_model import run_iforest
from models.knn_model import run_knn
from models.lof_model import run_lof
from models.ocsvm_model import run_ocsvm
from models.logistic_regression_model import run_logreg
from models.random_forest_model import run_rforest
from models.gradient_boosting_model import run_gboost
from models.linear_svc import run_linear_svc

from utils.evaluation import evaluate_model_success
from sklearn.model_selection import train_test_split
import pandas as pd


# Read data
df = pd.read_csv('data/creditcard.csv')

# Split features and target label
x = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and testing sets
x_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# CLI
print("""
--------------------------------
    Anomaly Detection System

1 - Isolation Forest
2 - KNN (K-Nearest Neighbors)
3 - LOF (Local Outlier Factor)
4 - OCSVM (One-Class SVM)!
5 - Logistic Regression
6 - Random Forest Classifier
7 - Gradient Boosting Classifier
8 - Linear SVC

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
                y_pred_train, scores_train, clf = run_iforest(x_train.values)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                y_pred_test = clf.predict(X_test.values)
                evaluate_model_success(y_test.values, y_pred_test)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

        elif selection == "2":
            print("→ KNN (K-Nearest Neighbors) Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred_train, scores_train, clf = run_knn(x_train.values)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                y_pred_test = clf.predict(X_test.values)
                evaluate_model_success(y_test.values, y_pred_test)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

        elif selection == "3":
            print("→ LOF (Local Outlier Factor) Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred, scores, clf = run_lof(X_test.values)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                evaluate_model_success(y_test.values, y_pred)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

        elif selection == "4":
            print("→ OCSVM (One-Class SVM) Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred_train, scores_train, clf = run_ocsvm(x_train.values)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                y_pred_test = clf.predict(X_test.values)
                evaluate_model_success(y_test.values, y_pred_test)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

        elif selection == "5":
            print("→ Logistic Regression Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred_test, model, scaler = run_logreg(x_train.values, y_train.values, X_test.values)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                evaluate_model_success(y_test.values, y_pred_test)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

        elif selection == "6":
            print("→ Random Forest Classifier Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred_train, clf = run_rforest(x_train, y_train)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                y_pred_test = clf.predict(X_test)
                evaluate_model_success(y_test.values, y_pred_test)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

        elif selection == "7":
            print("→ Gradient Boosting Classifier Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred_train, clf = run_gboost(x_train, y_train)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                y_pred_test = clf.predict(X_test)
                evaluate_model_success(y_test.values, y_pred_test)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue

        elif selection == "8":
            print("→ Linear SVC Classifier Model selected.")
            print("→ Please wait while the model is running...")

            try:
                y_pred_train, clf = run_linear_svc(x_train, y_train)
                print("✅ Model run successfully!\n")
            except Exception as e:
                print(f"❌ Model training failed:\n{e}\n")
                continue

            try:
                y_pred_test = clf.predict(X_test)
                evaluate_model_success(y_test.values, y_pred_test)
            except Exception as e:
                print(f"❌ Evaluation failed:\n{e}\n")
                continue


        else:
            print("Invalid selection. Please try again.")


except KeyboardInterrupt:
    print("\n🔴 Process interrupted by user. Exiting safely...")
