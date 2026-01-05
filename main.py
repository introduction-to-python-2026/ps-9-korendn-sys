import os

if not os.path.exists("parkinsons.csv"):
    os.system("wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O parkinsons.csv")

if not os.path.exists("lab_setup_do_not_edit.py"):
    os.system("wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O lab_setup_do_not_edit.py")

import pandas as pd
import lab_setup_do_not_edit
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import yaml

df = pd.read_csv("parkinsons.csv")

X = df[["MDVP:Jitter(%)", "MDVP:Shimmer", "MDVP:Fo(Hz)"]]
y = df["status"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC(kernel="rbf", C=10, gamma=0.1)
model.fit(X_train, y_train)

accuracy = model.score(X_val, y_val)
print("Validation Accuracy:", accuracy)

if accuracy >= 0.8:
    print("✅ Accuracy is acceptable.")
else:
    print("❌ Accuracy is below 0.8. Consider tuning further.")

model_filename = "parkinsons_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

config_data = {
    "selected_features": ["MDVP:Jitter(%)", "MDVP:Shimmer", "MDVP:Fo(Hz)"],
    "path": model_filename
}

with open("config.yaml", "w") as f:
    yaml.dump(config_data, f)

print("config.yaml updated with selected features and model path.")

