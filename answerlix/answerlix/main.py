import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import csv
import re
from itertools import zip_longest
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

import pandas as pd

# sample = pd.read_csv("outputs/subsettinginput.csv", header=None)
df = pd.read_csv('inputs_final/Datasource_.csv')
df = df.replace(np.nan, 'No Value Found', regex=True)
df = df.apply(lambda x: x.astype(str).str.lower())
headers = df.columns
for header in headers:
    if header == "Date" or header == "date":
        df[header] = pd.to_datetime(df[header])
        df['Year'], df['Month'], df['Day'] = df[header].dt.year, df[header].dt.strftime('%B'), df[header].dt.day
        del df[header]
df = df.apply(lambda x: x.astype(str).str.lower())
df.columns = map(str.lower, df.columns)

# sample = sample.apply(lambda x: x.astype(str).str.lower())


def write_tags(message):
    # generate tags
    with open('outputs/subsettinginput.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        test_text = str(message).lower()
        if "fy" in test_text:
            test_text = re.sub(r'fy (\d{4})-\d{4}', r'fy \1', test_text)
        doc = ner_loaded_model(test_text)
        for ent in doc.ents:
            tag = ''
            word_tagged = [ent.text]
            label_tagged = [ent.label_]
            tag = [word_tagged, label_tagged]
            # print(tag)
            export_data = zip_longest(*tag, fillvalue='')
            wr.writerows(export_data)
        print("yaay")
        # tablelookup


def lookup(sub_list_e, sub_list_t, df, target_column):
    # print("inside lookup funtion")
    sample_sub_dict = dict(zip(sub_list_t, sub_list_e))
    #   print(sample_sub_dict)
    val = sample_sub_dict.values()
    key = sample_sub_dict.keys()
    for k, v in sample_sub_dict.items():
        sdf = subset_return(df, k, v)
        # print(sdf.shape)
        df = sdf
    print("answer here :", df[target_column])
    final_answer = str(df[target_column])
    return final_answer


def subset_return(df, k, v):
    return df[df[str(k)] == str(v)]


app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('chat.html')


@app.route("/ask", methods=['POST'])
def ask():
    message = request.form['messageText'].encode('utf-8').strip()

    # kernel now ready for use
    while True:
        if message == "":
            message = "I Didn't get what are you saying"
        else:
            message = message.lower()
            message = str(message)
            target_column = classify_model.classify(message)
            target_column = target_column.lower()
            print(message)
            print(target_column)
            write_tags(message)
            sample = pd.read_csv("outputs/subsettinginput.csv", header=None)
            sample = sample.apply(lambda x: x.astype(str).str.lower())

            ###
            sample[0] = sample[0].map(lambda x: re.sub(r'\s?\?', '', x))
            # read csv sample
            list_e = sample[0].tolist()
            list_t = sample[1].tolist()
            counter = 0

            print(list_e)

            print(list_t)

            value = lookup(list_e, list_t, df, target_column)
            print(value)
            return jsonify({'status': 'OK', 'answer': value})


if __name__ == "__main__":
    ner_filename = 'input_model_pkl_files/finalized_model_answer.pkl'
    classify_filename = 'input_model_pkl_files/classification_model_answer.pkl'
    vect_filename = 'input_model_pkl_files/vectorizer.pkl'
    classify_model = pickle.load(open(classify_filename, 'rb'))
    ner_loaded_model = pickle.load(open(ner_filename, 'rb'))
    app.run(host='0.0.0.0', debug=True)

