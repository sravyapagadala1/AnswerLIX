
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import sklearn
import logging
from textblob.classifiers import NaiveBayesClassifier as NBC
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import sklearn
LOG_FILENAME = "logfile.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

# data_source = 'Inputs_final/Datasource.csv'
print("Enter the CSV name with complete path : For instance : Inputs_final/Datasource.csv")
data_source=input()
df = pd.read_csv(data_source)

logging.info("New session ")
logging.info("Reading Data source ")


# # Text Classification

# In[ ]:


logging.info("Reading input files ")


# In[ ]:


# final_train = "Inputs_final/train_quesitons_70_textclassification.csv"
print("Enter the train questions CSV for text classifictaion(Question and target column) : For instance : Inputs_final/train_quesitons_70_textclassification.csv")
final_train=input()
# df_train = pd.read_csv(final_train)

# print("/n")
# test_question="Inputs_final/test_questions_30.txt"


# In[ ]:


print("Enter the .txt file containing train questions : For instance : Inputs_final/test_questions_30.txt")
test_question=input()


# In[ ]:


final_train_questions = pd.read_csv(final_train)
final_train_questions = final_train_questions.values.tolist()


# In[ ]:


x_train ,x_test = train_test_split(final_train_questions,test_size=0.3) 


# In[ ]:


clf = NBC(x_train) 


# In[ ]:


print("Naive Bayes Classifier Accuracy is",clf.accuracy(x_test))


# In[ ]:


target_column = []
f = open(test_question, "r")
lines = f.readlines()
for i in lines:
    target_column.append(clf.classify(i))


# In[ ]:


print("target_column : \n", target_column)


# In[ ]:


# Save our trained Model in pkl model
filename = 'pkl_files/classification_model_answer.pkl'
pickle.dump(clf, open(filename, 'wb'))


# # Generate custom tags

# In[ ]:


logging.info("Start generating customer tags")
df = df.replace(np.nan, 'No Value Found', regex=True)
df=df.apply(lambda x: x.astype(str).str.lower())


# In[ ]:


# a ="tag_values_"
tag_values_csv =[]
headers=df.columns
for header in headers:
    if header == "Date" or header == "date":
        #print(header)
        df[header] = pd.to_datetime(df[header])
        df['Year'], df['Month'], df['Day'] = df[header].dt.year, df[header].dt.strftime('%B'), df[header].dt.day
        del df[header]
df = df.apply(lambda x: x.astype(str).str.lower())
df.columns = map(str.lower, df.columns)
# print(df)
headers=df.columns
for x in range(len(headers)):
    f_tag=headers[x]
    #print(f_tag)
    tag_values_csv.append(f_tag)
    tag_values_csv
dictionary = {}
for i in range(len(tag_values_csv)):
    dictionary[tag_values_csv[i]] = df.iloc[:,i].unique().tolist()
dictionary
[dict([a, str(x)] for a,x in dictionary.items())]


# In[ ]:


dictionary['unit_id']


# In[ ]:


key_label=list(dictionary.keys())
values=list(dictionary.values())
logging.info("customer tags generated")


# In[ ]:


import spacy
from spacy.matcher import PhraseMatcher, Matcher
from pathlib import Path
import plac
from spacy.util import minibatch, compounding
import random
logging.info("Load spacy model")
nlp = spacy.load('en')
matcher = PhraseMatcher(nlp.vocab)
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')


# In[ ]:


label_list=[]
matchitem_list=[]
def offseter(doc, matchitem, dictionary):
    o_one = len(str(doc[0:matchitem[1]]))+1
    subdoc= doc[matchitem[1]:matchitem[2]]
    #print(subdoc)
    o_two = o_one + len(str(subdoc))
    for k in dictionary.keys():
        if str(subdoc) in dictionary[k]:
            label_list.append(k)
    matchitem_list.append(subdoc)
    return (o_one, o_two)


# In[ ]:


for l in key_label:
    for v in values:
        for i in v:
            matcher.add(l, None, nlp(i))


# In[ ]:


one = nlp('india united states dollar ?')
matches = matcher(one)
[match for match in matches]


# In[ ]:


two = nlp('united states dollar for big data')
matches = matcher(two)
[match for match in matches]


# # Loop For tagging

# In[ ]:


# file_name='Inputs_final/train_quesitons_70.csv'

print("Enter the train questions for custom NER model (just the Questions ) : For instance : Inputs_final/train_quesitons_70.csv")
file_name=input()
# file_name = pd.read_csv(file_name)



import re
train_data =[]
def linebyline (line):
    res = []
    to_train_ents = []
    entities = {'entities':[]}
    line=line.lower()
    if "fy" in line:
        line = re.sub(r'fy (\d{4})-\d{4}', r'fy \1', line)
    mnlp_line = nlp(line)
    matches = matcher(mnlp_line)
    res = [offseter(mnlp_line, x, dictionary)
       for x
       in matches]
    entities = {'entities':[]}
    for i in range(len(res)):
        a= list(res[i])
        a.append(label_list[i])
        entities['entities'].append(tuple(a))
    to_train_ents.append((line,entities))
    return to_train_ents

with open(file_name) as data1:
    line= True
    while line:
        label_list = []
        line = data1.readline()
        final=linebyline(line)

        train_data.append(final)

flat_list = [item for sublist in train_data for item in sublist]

print(flat_list)

logging.info("Train data created!")


# In[ ]:


final_train_data=[]
count=0
for z in range(len(flat_list)):
    if len(flat_list[z][1]['entities'])!=0:
        count=count+1
        final_train_data.append(flat_list[z])
print(final_train_data)


# # Model Training

# In[ ]:


import csv
from itertools import zip_longest

TRAIN_DATA =final_train_data
logging.info("Runnning spacy model")

def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)
logging.info("Fininshed 20 iterations!")

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)


# In[ ]:


# Save our trained Model in pkl model
filename = 'pkl_files/finalized_model_answer.pkl'
pickle.dump(prdnlp, open(filename, 'wb'))


# # Separate file

# In[ ]:


f = open(test_question, "r")
lines = f.readlines()
logging.info("writing into subsettinginput file")
with open('Outputs/subsettinginput.csv', 'w', encoding='utf-8',errors='ignore', newline='') as myfile:
    wr = csv.writer(myfile)
    for line in lines:
        year_count=0
        test_text=line.lower()
        if "fy" in test_text:
            test_text = re.sub(r'fy (\d{4})-\d{4}', r'fy \1', test_text)
        doc = prdnlp(test_text)
        for ent in doc.ents:
            tag=''
            word_tagged=[ent.text]
            label_tagged=[ent.label_]
            tag= [word_tagged,label_tagged]
            export_data = zip_longest(*tag, fillvalue = '')
            wr.writerows(export_data)
        wr.writerows('#')


# # Subsetting

# In[ ]:


sample = pd.read_csv("Outputs/subsettinginput.csv",header=None)
df = pd.DataFrame(df)
df = df.apply(lambda x: x.astype(str).str.lower())
sample = pd.DataFrame(sample)
sample = sample.apply(lambda x: x.astype(str).str.lower())
df.columns = map(str.lower, df.columns)
logging.info("Reading from subsettinginput file")


# In[ ]:


print(df.head(5))


# In[ ]:


sample[0] = sample[0].map(lambda x: re.sub(r'\s?\?', '', x))
sample.head(50)


# In[ ]:


print(df.shape)


# In[ ]:


#detect question boundary
a=sample.index[sample[0] == '#']
index=list(a)
index
no_of_question= len(index)
print(no_of_question)


# In[ ]:


target_column = [x.lower() for x in target_column] 
print(type(target_column))


# In[ ]:


# logging.info("subsetting logic begins")
# def lookup (sub_list_e,sub_list_t,df,q_num):
#     sample_sub_dict = dict(zip(sub_list_t, sub_list_e))
#     val = sample_sub_dict.values()
#     key = sample_sub_dict.keys()
#     valu=['no ans']
#     res = pd.DataFrame(valu)
#     for k,v in sample_sub_dict.items():
#         sdf = subset_return(df,k,v)
#         df = sdf  
#     print("answer,",df[target_column[q_num]])
#     return df[target_column[q_num]]

# def subset_return(df,k,v):
#     return df[df[str(k)] == str(v)]
# #read csv sample
# list_e = sample[0].tolist()
# list_t = sample[1].tolist()
# counter=0
# q_num=0
# #question wise lists
# for i in index:
#     sub_list_e=list_e[counter:i]
#     sub_list_t=list_t[counter:i]
#     counter=i+1
#     value=lookup(sub_list_e,sub_list_t,df,q_num)
# #     print(value)
#     q_num=q_num+1
#     print("*")
# logging.info("subsetting logic ends")
# logging.info("")


# In[ ]:


#read csv sample
#**** changed subsetting function ****#
list_e = sample[0].tolist()
list_t = sample[1].tolist()
counter=0
q_num=0
list_result = []
for i in index:
    sample
    sub_list_e=list_e[counter:i]
    #print(sub_list_e)
    sub_list_t=list_t[counter:i]
    #print(sub_list_t)
    sample_sub_dict = dict(zip(sub_list_t, sub_list_e))
    #print(sample_sub_dict)
    qry = ' and '.join(["{} == '{}'".format(k,v) for k,v in sample_sub_dict.items()])
    #result_df = merge(pd.DataFrame(sample_sub_dict,index = [0]), df)
    result_df = df.query(qry)
    print(result_df[target_column[q_num]])
    list_result.append(result_df[target_column[q_num]])
    counter=i+1
    q_num=q_num+1


# In[ ]:


# saving answers into csv file
final_df = pd.DataFrame({'col':list_result})
final_df.to_csv("Outputs/file_csv.csv", sep=' ')
file_csv = pd.read_csv("Outputs/file_csv.csv",error_bad_lines=False)
file_csv = file_csv[file_csv[' col'].str.contains("\"")]
file_csv[['question_number','answer']] = file_csv[' col'].str.split('\"',expand=True)
file_csv['answer'] = file_csv['answer'].str.replace("    ",",")
file_csv['answer'] = file_csv['answer'].str.replace(", ",",")
file_csv['answer'] = file_csv['answer'].str.replace(",,",",")
file_csv['answer'] = file_csv['answer'].str.replace(",,,",",")
file_csv[['answer_column','answer_final']] = file_csv['answer'].str.split(',',expand=True)
file_csv = file_csv.reset_index(drop=True)
file_csv = file_csv.drop(columns = ['answer_column','answer',' col'])
file_csv.to_csv("Outputs/file_csv.csv",index=False)
print(file_csv)

