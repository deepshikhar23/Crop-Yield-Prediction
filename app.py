import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the merged and cleaned dataset
new_df = pd.read_csv('dataset.csv')

# Get the unique values for 'Area' and 'Item'
area_options = new_df['Area'].unique().tolist()
crop_options = new_df['Item'].unique().tolist()

# Perform data preprocessing and encoding
df_encoded = pd.get_dummies(new_df, columns=['Area', 'Item'], prefix=['Country', 'Item'])

# Split the dataset into input features 'X' and target variable 'y'
X = df_encoded.drop('hg/ha Yield', axis=1)
y = df_encoded['hg/ha Yield']

# Apply MinMaxScaler to X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Decision Tree Regressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Save the trained model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

# Define the main function for the app
@app.route('/')
def main():
    return render_template('index.html', area_options=area_options, crop_options=crop_options)

# Define a route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_features = {
        'avg_mm_rainfall': float(request.form['avg_mm_rainfall']),
        'tonnes_of_pesticides': float(request.form['tonnes_of_pesticides']),
        'avg_temp': float(request.form['avg_temp']),
        'area': request.form['area'],
        'item': request.form['item']
    }

    processed_features = preprocess_input_features(input_features)
    yield_prediction = make_prediction(processed_features)

    return render_template('result.html', prediction=yield_prediction)

# Function to preprocess the input features
def preprocess_input_features(input_features):
    input_df = pd.DataFrame([input_features])
    input_encoded = pd.get_dummies(input_df, columns=['area', 'item'])
    input_aligned = input_encoded.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_aligned)
    return input_scaled

# Function to make predictions
def make_prediction(processed_features):
    prediction = regressor.predict(processed_features)
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
