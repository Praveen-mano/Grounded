import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv("Crop_recommendation2.csv")

# 2. Prepare Data (Map 'humidity' to 'Moisture')
df = df.rename(columns={'humidity': 'Moisture', 'label': 'Crop'})
X = df[['N', 'P', 'K', 'Moisture']]  # Inputs
y = df['Crop']                        # Output

# 3. Train Model
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Check Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Trained. Accuracy: {accuracy * 100:.2f}%")

# 5. SAVE the Model
joblib.dump(model, 'agri_model.pkl')
print("Success! Model saved as 'agri_model.pkl'")