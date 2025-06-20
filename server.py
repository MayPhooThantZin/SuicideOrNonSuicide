from flask import Flask, render_template, request, redirect, url_for
import pickle
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions,os
from nltk.stem import WordNetLemmatizer
import wordninja

app = Flask(__name__)
path = os.path.join(os.getcwd() , 'data')
#Loading models
with open(os.path.join(path,'GridSearchLR.pkl'), 'rb') as file:  
    LRmodel = pickle.load(file)
with open('./data/GridSearchRF.pkl','rb') as file:
    RFmodel = pickle.load(file)
with open('./data/GridSearchXgb.pkl','rb') as file:
    XGBoostmoodel = pickle.load(file)
with open('./data/Vectorizer.pkl', 'rb') as file:  
    vectorizer = pickle.load(file)
with open('./data/Scaler.pkl', 'rb') as file:  
    scaler = pickle.load(file)
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
label = ['suicidal','not suicidal']

#Preprocessing
def preprocess(sent):
    demojized_sent = emoji.demojize(sent)
    if isinstance(demojized_sent, str):  # Ensure text is a string
        try:
            contract_sent = contractions.fix(demojized_sent)
        except Exception as e:
            print(f"Error processing text: {demojized_sent}")
            print(e)
    else:
        contract_sent = demojized_sent

    text = re.sub(r'https?://\S+|www\.\S+', '', contract_sent)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\n', ' ', text)  # Replace newline with space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatized_tokens = [wnl.lemmatize(t, pos="v") for t in tokens]
    processed_words = []
    for word in lemmatized_tokens:
        split_words = wordninja.split(word)
        processed_words.extend(split_words)
    processed_words = list(filter(None, processed_words))
    sent = ' '.join(processed_words)
    tokens = word_tokenize(sent)
    removeStop_token = [word for word in tokens if word not in stop_words]
    feed = [str(removeStop_token)]
    vector_feed = vectorizer.transform(feed).toarray()
    vector_feed = scaler.transform(vector_feed)
    return vector_feed


# Routes
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        input_text = request.form.get('input_text')
        preprocess_text = preprocess(input_text)
        LR_result = LRmodel.predict(preprocess_text)
        print(LR_result[0])
        if LR_result[0] == 1:
            LR_result = 'suicidal'
        else:
            LR_result = 'not suicidal'
        RF_result = RFmodel.predict(preprocess_text)
        if RF_result[0] == 1:
            RF_result = 'suicidal'
        else:
            RF_result = 'not suicidal'
        XGBoost_result = XGBoostmoodel.predict(preprocess_text)
        if XGBoost_result[0] == 1:
            XGBoost_result = 'suicidal'
        else:
            XGBoost_result = 'not suicidal'
        print(input_text,'LR: ', LR_result, ' RF: ',RF_result, ' XGBoost: ',XGBoost_result)
        return render_template('index.html',text = input_text, LR_result=LR_result,RF_result = RF_result,XGBoost_result = XGBoost_result)


# Main driver
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
