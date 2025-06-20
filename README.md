📄 Dataset Overview

The dataset used in this project is a CSV file containing textual data related to mental health and suicide detection. It consists of two columns:

    text: The input sentence or paragraph.

    target: The corresponding label, indicating whether the text implies "Suicide" or "Non-suicide".

This binary classification dataset is designed to help identify potentially harmful or suicidal language in text, which can be valuable in mental health monitoring and support systems.
🧠 Model Development

To build a reliable suicide detection system, several machine learning models were trained and evaluated using the labeled dataset. The models include:

    Logistic Regression

    K-Nearest Neighbors (KNN)

    XGBoost

The training pipeline involves the following steps:

    Text preprocessing: Cleaning, tokenization, stop-word removal, and vectorization using techniques like TF-IDF.

    Model training: Each model is trained on a portion of the data and validated using performance metrics such as accuracy, precision, recall, and F1-score.

    Model selection: The best-performing model is selected based on its ability to accurately distinguish between suicidal and non-suicidal texts.

🌐 Web Application

A simple Flask-based web application has been developed to demonstrate the suicide detection model in action.
🔧 Features:

    Users can input text into a web form.

    The application sends the input to the trained model on the server.

    The model analyzes the text and returns a prediction: "Suicide" or "Non-suicide".

    The result is displayed to the user in a clear and intuitive interface.

This web app serves as a proof of concept for using machine learning in real-time mental health support applications.
