# ASAP-TextAbstractorSummary
Predicts an abstract summary for the given text
## Project Setup

Create python virtual environment
  `pip install virtualenv`
  
  clone this repository
  
  In repository folder create virtual environment for project: `virualenv <environment_name>`
  
  activate environment : `source <environment_name>/bin/activate`

Install Libraries
  numpy: `pip install numpy`
  
  pandas: `pip install pandas`
  
  tensorflow 1.5: `pip install tensorflow==1.5`
  
  keras 2: `pip install keras==2`
  
  nltk: `pip install nltk`
 
Download dataset from https://www.kaggle.com/snap/amazon-fine-food-reviews

To train the model execute text_abstraction.py

text_abstraction.py will train the RNN and save the model and other necessary data structure

To generate summary execute predict.py

predict.py file will load the model and necessary data structre to predict.

Note : Please add or update libraries as necessary

##Android app

TextSummarizer is an android studio project built in kotlin

To run the android studio project create a new android studio project

Select empty activity

Name project as TextSummarizer and set language as kotlin

Replace file to respective folder from repositories TextSummarizer folder  

Run or Build app

An test apk app-debug.apk has been provided inside TextSummarizer folder
