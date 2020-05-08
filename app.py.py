#!/usr/bin/env python
# coding: utf-8

# In[1]:


# backend


# In[1]:


import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle
import math


# In[2]:


app=Flask(__name__)
model=pickle.load(open('taxi.pkl','rb'))


# In[3]:


@app.route('/')
def home():   
    return render_template("index.html")
@app.route('/predict',methods=["Post"])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)# to round off the value
    return render_template("index.html", prediction_text="number of weekly ride needed{}".format(math.floor(output)))
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
 
   


# 

# In[9]:


get_ipython().run_line_magic('tb', '')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




