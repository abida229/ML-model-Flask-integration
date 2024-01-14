from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessing steps
loaded_pipeline = joblib.load ('model/trained_model_with_preprocessing.joblib')

# Define categories based on predicted numerical labels
def map_to_category(label):
    if label == 0:
        return "Severe Depression"
    elif label == 1:
        return "Moderately Severe Depression"
    elif label == 2:
        return "Moderate Depression"
    elif label == 3:
        return "Mild Depression"
    elif label == 4:
        return "Minimal Depression / Happy / Normal"
    else:
        return "Unknown Label"

@app.route('/')
def home():
    return render_template('fyp_form.html', predictions=None)  # Set predictions to None initially

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        phq1 = int(request.form['phq1'])
        phq2 = int(request.form['phq2'])
        phq3 = int(request.form['phq3'])
        phq4 = int(request.form['phq4'])
        phq5 = int(request.form['phq5'])
        phq6 = int(request.form['phq6'])
        phq7 = int(request.form['phq7'])
        phq8 = int(request.form['phq8'])
        phq9 = int(request.form['phq9'])
        age = int(request.form['age'])
        sex = request.form['sex']
        period_name = request.form['period_name']

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'phq1': [phq1],
            'phq2': [phq2],
            'phq3': [phq3],
            'phq4': [phq4],
            'phq5': [phq5],
            'phq6': [phq6],
            'phq7': [phq7],
            'phq8': [phq8],
            'phq9': [phq9],
            'age': [age],
            'sex': [sex],
            'period.name': [period_name]
            # Add more features as needed
        })

        # Make predictions using the loaded model
        predictions = loaded_pipeline.predict(input_data)

        # Map predicted numerical labels to descriptive labels
        prediction_labels = list(map(map_to_category, predictions))

        # Display the predicted labels
        return render_template('fyp_form.html', predictions=prediction_labels)

    except Exception as e:
        # Handle any errors and display an error message
        return render_template('fyp_form.html', error=str(e), predictions=None)

if __name__ == '__main__':
    app.run(debug=True)
