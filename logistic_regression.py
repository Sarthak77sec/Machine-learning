"""Predict whether a student will pass from their study hours."""

import pandas as pd
from sklearn.linear_model import LogisticRegression


# 1 means pass; 0 means fail.
hours = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
passed = [0,   0, 0,   0, 0,   0, 1,   1, 1,   1, 1,   1]

# Put the training data into a table.
data = pd.DataFrame({"Hours": hours, "Passed": passed})

# Teach the model: use Hours to predict Passed.
model = LogisticRegression()
model.fit(data[["Hours"]], data["Passed"])

# Ask the user about a new student.
study_hours = float(input("How many hours did the student study? "))
student = pd.DataFrame({"Hours": [study_hours]})

# Find the predicted result and the probability of passing.
result = model.predict(student)[0]
pass_probability = model.predict_proba(student)[0, 1]

print(f"\nChance of passing: {pass_probability:.0%}")
print("Prediction:", "Pass" if result == 1 else "Fail")
