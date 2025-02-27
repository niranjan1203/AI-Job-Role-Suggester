import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("tech_skills_recommendation_500.csv")

# Drop unnecessary columns
df.drop(columns=["User_ID"], inplace=True)

# Convert categorical columns to numerical using Label Encoding
label_encoders = {}
categorical_columns = ["Current_Skills", "Recommended_Skills", "Experience_Level", "Preferred_Tech_Role"]

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Split dataset into features and labels
X = df.drop(columns=["Preferred_Tech_Role"])  # Features (input)
y = df["Preferred_Tech_Role"]  # Target (output)

# Split into Training & Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model as a Pickle file
joblib.dump(model, "Tech_Skill_Model.pkl")

print("✅ Model trained and saved as Tech_Skill_Model.pkl")

