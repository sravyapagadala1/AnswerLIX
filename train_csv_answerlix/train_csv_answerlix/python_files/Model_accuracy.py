
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# In[ ]:


generated_answers = pd.read_csv("Outputs/file_csv.csv") 


# In[ ]:


generated_answers['category'] = 'yes'
generated_answers.head()


# In[ ]:


total_questions = int(generated_answers.iloc[-1]["question_number"])
index_list = [x for x in range(0,total_questions+1)]


# In[ ]:


generated_answers.set_index(['question_number','category'], inplace=True)
(ques_index, cat_index) = generated_answers.index.levels
new_index = pd.MultiIndex.from_product([index_list, cat_index])


# In[ ]:


new_answers = generated_answers.reindex(new_index)
new_answers.answer_final[new_answers.answer_final == 'no value found'] = '0' 
new_answers = new_answers.fillna(0).astype(float)
new_answers['answer_final'] = new_answers['answer_final'].astype(int)


# In[ ]:


k= pd.read_csv("Outputs/final_answers.csv",header = None)


# In[ ]:


actual = k[0].values.tolist()
predicted = new_answers['answer_final'].values.tolist()
count = 0 
total = len(actual)
for i in range(0,total):
    if (actual[i] == predicted[i]):
        count = count+1
print('The final accuracy score for test questions :',count/total*100)


# In[1]:


print('shashi')

