from flask import Flask, request, jsonify
# from flask_ngrok import run_with_ngrok
import requests
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import keras
import json
import os
MODEL_PICKLE = "ann_clf.pkl"
EXAMPLE_DATA_FILE = "example_data_cleaned.csv"
EXAMPLE_MEAN_STD_FILE = "example_data_preprocessed_mean_std.csv"
TARGET_NAME = "Osteoporosis"


app = Flask(__name__)
# run_with_ngrok(app)


@app.route("/")
def home():
    return jsonify({ 'status' : 'running'})

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    parameters = req['sessionInfo']['parameters']
    print(parameters)

    height_meters = parameters['height'] / 100
    weight_kg = parameters['weight']
    bmi = weight_kg / (height_meters ** 2)

    # Convert to the desired format
    input_data = {
        "Gender": parameters['gender'],
        "Race": f"Non-Hispanic {parameters['race']}",
        "Age": int(parameters['age']),
        "Sleep Duration (Hours)": int(parameters['sleep']),
        "BMI": round(bmi, 1),
        "Smoking": "Yes" if parameters['smoke'] == 'yes' else 'No',
        "Heavy Drinking": "Yes" if parameters['alcoholic'] == 'yes' else 'No',
        "Arthritis": "Yes" if parameters['arthritis'] == 'yes' else 'No',
        "Liver Condition": "Yes" if parameters['liver'] == 'yes' else 'No',
        "Parental Osteoporosis": "Yes" if parameters['genetic'] == 'yes' else 'No'
    }
    print("inputdata=")
    print(input_data)
    loaded_model = load_model()
    # 调用预测函数
    # df_input_preprocessed = preprocess_input_data(df_input, example_features, mean, std)
    prediction_str, prediction_prob = predict_osteoporosis(input_data)

    # 打印预测结果
    print("Prediction:", prediction_str)
    print("Confidence Score:", round(prediction_prob * 100, 1))
    score=round(prediction_prob * 100, 1)

    response_text = f"The Confidence Score is {score} and the Prediction result is {prediction_str}."

    # 构建回复
    fulfillment_response = {
        "fulfillment_response": {
            "messages": [{"text": {"text": [response_text]}}]
        }
    }


    return fulfillment_response


    # loaded_model = load_model()
    # # 调用预测函数
    # prediction_str, prediction_prob = predict_osteoporosis(input_data)

    # # 打印预测结果
    # print("Prediction:", prediction_str)
    # print("Confidence Score:", round(prediction_prob * 100, 1))
    # score=round(prediction_prob * 100, 1)
    # # score=0

    # response_text = f"The Confidence Score is {score} and the Prediction result is {prediction_str}."

    # # 构建回复
    # fulfillment_response = {
    #     "fulfillment_response": {
    #         "messages": [{"text": {"text": [response_text]}}]
    #     }
    # }
    # print('..............................................................')
    # print(fulfillment_response)
    # return jsonify(fulfillment_response)


def load_model():
    try:
        model = pickle.load(open(MODEL_PICKLE, 'rb'))
        return model
    except FileNotFoundError:
        return None


def encode_input_data(input_data, example_data):
    """Return the encoded user input data"""

    # Combine user input features with entire survey dataset for encoding the input features
    df_all = pd.concat([input_data, example_data], axis=0)
    # 将新数据和训练数据连接起来
    # One hot encode
    df_all_encoded = one_hot_encode(df_all)
    # The first row is the encoded input data
    df_input_encoded = df_all_encoded[:1]

    return df_input_encoded
 # 预处理用户输入数据
def preprocess_input_data(input_data, example_data, mean, std):
    df_input_encoded = encode_input_data(input_data, example_data)
    df_input_encoded = binning_data(df_input_encoded)
    df_input_scaled = (df_input_encoded - mean) / std
    return df_input_scaled



# 获取预测结果
def get_prediction(model, input_data_preprocessed):
    prediction_logic = model.predict(input_data_preprocessed)
    tf_prediction = tf.nn.sigmoid(prediction_logic)
    prediction_prob = tf.get_static_value(tf_prediction)[0].item()
    prediction_index = round(prediction_prob)
    existence = np.array(['No', 'Yes'])
    prediciton_str = existence[prediction_index]
    return prediciton_str, prediction_prob



# 处理用户输入数据并返回预测结果
def predict_osteoporosis(input_data):
    try:
        # Load the cleaned example data
        example_data = pd.read_csv(EXAMPLE_DATA_FILE)
        example_data = example_data.drop(columns=[TARGET_NAME])
        # Load the mean and std of preprocessed example data
        data = pd.read_csv(EXAMPLE_MEAN_STD_FILE)
        mean = data.iloc[0]
        std = data.iloc[1]
        # Load the saved model
        model = load_model()
        if model is None:
            return "Error: Model not found", None
        # Convert input_data to DataFrame
        df_input = pd.DataFrame(input_data, index=[0])
        # Preprocess input data
        df_input_preprocessed = preprocess_input_data(df_input, example_data, mean, std)
        # Get prediction
        prediction_str, prediction_prob = get_prediction(model, df_input_preprocessed)
        return prediction_str, prediction_prob
    except FileNotFoundError:
        return "Error: Data not found", None




# ml_helper
def binning_bmi(data):
    """
    Binning Body Mass Index (BMI)
    """
    # underweight
    group = 1
    # healthy weight
    if 18.5 <= data < 25:
        group = 2
    # overweight
    if 25 <= data < 30:
        group = 3
    # obesity
    if data >= 30:
        group = 4

    return group

def binning_sleep_duration(data):
    """
    Binning sleep duration (hours)
    """
    # less than 7 hours
    group = 1
    # 7-9 hours (recommended)
    if 7 <= data < 9:
        group = 2
    # more than 9 hours
    if data > 9:
        group = 3

    return group

def binning_data(data):
    """
    Binning BMI and sleep duration
    """
    data['BMI Group'] = data['BMI'].apply(binning_bmi)
    data['Sleep Duration Group'] = data['Sleep Duration (Hours)'].apply(binning_sleep_duration)
    data = data.drop(columns=['BMI', 'Sleep Duration (Hours)'])

    return data

def one_hot_encode(data):
    """
    Method for One-Hot Encoding
    """
    cate_list = list(data.select_dtypes(include=['category', 'object']).columns)
    df_encoded = pd.get_dummies(data, columns=cate_list, prefix_sep='_')
    # drop columns end with '_No'
    df_encoded = df_encoded[df_encoded.columns.drop(list(df_encoded.filter(regex='_No$')))]
    # remove '_Yes', 'Gender_', and 'Race_' from column names
    df_encoded.columns = df_encoded.columns.str.replace("_Yes|Gender_|Race_", "", regex=True)
    # drop redundant columns to reduce the impact of multicollinearity
    df_encoded = df_encoded.drop(columns=['Male', 'Other Race - Including Multi-Racial'], errors='ignore')

    return df_encoded

def oversampling(X, y, oversampler, sampling_strategy="auto"):
    """Oversampling the minority class

    Args:
        X: independent variables
        y: corresponding target variable
        oversampler: oversampling method that will be applied to X, y

    Returns:
        X and y after oversampling
    """
    model = oversampler(random_state=42, sampling_strategy=sampling_strategy)
    X_oversample, y_oversample = model.fit_resample(X, y)

    return X_oversample, y_oversample

def standardize(data, save_csv=False, file_path=""):
    data_mean = data.mean()
    data_std = data.std()
    data_scaled = (data - data_mean)/data_std
    # save mean and std as .csv
    if save_csv and file_path:
        df_mean = data_mean.to_frame().T
        df_std = data_std.to_frame().T
        combined = pd.concat([df_mean, df_std])
        combined.to_csv(file_path, index=False)

    return data_scaled

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
