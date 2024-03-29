# Crop-Yield-Predictor                                                                                                               
(http://65.0.134.124:5000/)

This is a simple web application that predicts crop yield based on various input features such as rainfall, pesticide usage, and temperature. The app utilizes a Decision Tree Regression model to make the predictions.

# Screenshots
![ss2](https://github.com/deepshikhar23/Crop-Yield-Prediction/assets/116090674/95cf37e1-24d6-4e71-a659-1da3d8478abe)
![ss3](https://github.com/deepshikhar23/Crop-Yield-Prediction/assets/116090674/413ae955-a29f-46d7-b2ca-26cc510d6e8b)


## Installation

To run the Crop Yield Prediction App locally, you need to follow these steps:

1. Clone the repository:
git clone https://github.com/deepshikhar23/Crop-Yield-Prediction

2. Install the required dependencies. Run the following command in your terminal or command prompt:
pip install pandas scikit-learn flask

3. Download the dataset file 'processed and merged dataframe.csv' and place it in the same directory as the app code.

## Usage

1. Run the Flask web application by executing the following command in your terminal or command prompt:
python app.py

2. Access the app by opening your web browser and visiting http://localhost:8080
3. Fill in the input fields with the relevant information:

Average mm of Rainfall: Enter the average amount of rainfall in millimeters.
Tonnes of Pesticides: Enter the amount of pesticides used in tonnes.
Average Temperature: Enter the average temperature.
Area: Select the area where the crop is grown from the dropdown list.
Crop Type: Select the type of crop from the dropdown list.
4. Click the Predict button to see the predicted crop yield based on the provided inputs.

## AWS Deployment

The Crop Yield Prediction App is also hosted on AWS for convenient access. You can visit the app at the following URL: http://65.0.134.124:5000/

Please note that the AWS deployment may have specific configurations and maintenance requirements that are not covered in this README. For further information or inquiries about the AWS deployment, please contact Deepshikhar Saxena at deepshikhar2305@gmail.com

## Development
If you wish to modify or enhance the Crop Yield Prediction App, you can follow these steps:

Open the Jupyter Notebook file 'Untitled.ipynb' to explore the dataset and perform any necessary data preprocessing or analysis.

Make changes to the Flask app code in 'app.py' using your preferred code editor.

Test your modifications by running the Flask app locally as described in the Usage section.

Iterate and refine your changes as needed.

## Dataset
The dataset used for training the prediction model is stored in the file 'processed and merged dataframe.csv'. It contains information about various crop yield factors, such as rainfall, pesticide usage, and temperature, along with the corresponding crop yield values.

## License
The Crop Yield Prediction App is released under the MIT License. You are free to modify and use the code for personal and commercial purposes.

Note: This app is provided for demonstration purposes only and may not yield accurate results for real-world scenarios. It is advisable to consult domain experts and conduct further research before making any agricultural decisions based on the predictions generated by this app.

For further information or inquiries, please contact Deepshikhar Saxena at deepshikhar2305@gmail.com
