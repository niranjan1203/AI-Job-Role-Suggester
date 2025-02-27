from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder="template")

# ✅ Load trained model
model = joblib.load("Tech_Skill_Model.pkl")

# ✅ Define job role mappings
job_roles = {
    0: "Software Engineer",
    1: "Data Scientist",
    2: "AI Engineer",
    3: "Cybersecurity Analyst",
    4: "Cloud Engineer",
    5: "Web Developer",
    6: "Business Analyst",
    7: "Mechanical Engineer",
    8: "Electrical Engineer"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle job role prediction."""
    try:
        # ✅ Get user input
        skill = request.form.get("feature1")
        experience = request.form.get("feature2")
        trend_score = request.form.get("feature3")
        job_openings = request.form.get("feature4")
        education_level = request.form.get("feature5")

        # ✅ Check for missing values
        if not all([skill, experience, trend_score, job_openings, education_level]):
            return jsonify({"error": "Missing input values. Please fill in all fields."})

        # ✅ Convert input values to integers
        try:
            experience = int(experience)
            trend_score = int(trend_score)
            job_openings = int(job_openings)
            education_level = int(education_level)
        except ValueError:
            return jsonify({"error": "Invalid input. Please enter valid numbers for experience, trend score, job openings, and education level."})

        # ✅ Encode categorical skills
        skills_mapping = {
            "Python": 0, "Java": 1, "C++": 2, "JavaScript": 3, "Data Science": 4,
            "AI": 5, "Cybersecurity": 6, "Cloud Computing": 7, "Web Development": 8
        }
        skill_encoded = skills_mapping.get(skill, -1)

        if skill_encoded == -1:
            return jsonify({"error": "Invalid skill input"})

        # ✅ Prepare input for model
        features = np.array([[skill_encoded, experience, trend_score, job_openings, education_level]])

        # ✅ Ensure correct number of features
        if features.shape[1] != 5:
            return jsonify({"error": f"Incorrect number of features. Expected 5 but got {features.shape[1]}"})

        # ✅ Predict job role
        job_role_index = model.predict(features)[0]
        recommended_job_role = job_roles.get(job_role_index, "Not Found")

        return jsonify({"job_role": recommended_job_role})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
