from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained recommendation model
model_path = "medicine_model.pkl"
df_path = "medicine_data.csv"

try:
    with open(model_path, 'rb') as file:
        vectorizer, similarity_matrix = pickle.load(file)
    df = pd.read_csv(df_path)
    print("Model and dataset loaded successfully")
except Exception as e:
    print("Error loading model or dataset:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        medicine_name = data.get("medicine")
        
        # Find the index of the medicine in the dataframe
        indices = df[df['Drug_Name'].str.contains(medicine_name, case=False)].index
        
        if len(indices) == 0:
            return jsonify({"recommendations": []})
        
        # Get the index of the most relevant match
        idx = indices[0]
        
        # Calculate similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5 similar medicines (excluding the input medicine)
        medicine_indices = [i[0] for i in sim_scores]
        
        # Get recommended medicines with their details
        recommendations = []
        for i in medicine_indices:
            medicine_info = {
                "name": df.iloc[i]['Drug_Name'],
                "reason": df.iloc[i]['Reason'],
                "description": df.iloc[i]['Description']
            }
            recommendations.append(medicine_info)
        
        return jsonify({"recommendations": recommendations})
    
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

