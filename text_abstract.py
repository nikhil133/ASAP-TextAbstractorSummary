#keras 2.0
#tensorflow 1.5
from attention import AttentionLayer
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import pickle
from text_cleaner import text_cleaner,rareword_coverage



pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#568457
cols=["Id","ProductId","UserId","ProfileName","HelpfulnessNumerator","HelpfulnessDenominator","Score","Time","Summary","Text"]
data=pd.read_csv("Reviews.csv",nrows=666856,usecols=cols)
data=data.sample(frac=1)
#print(data)
data.drop_duplicates(subset=['Text'],inplace=True)#dropping duplicates
data.dropna(axis=0,inplace=True)#dropping na
#data.info()
#print(data.head)



#call the function
cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t,0)) 
#print(cleaned_text[:5])

cleaned_summary=[]

for t in data['Summary']:
    cleaned_summary.append(text_cleaner(t,1))

#print(cleaned_summary)
data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary

data.replace('',np.nan,inplace=True)
data.dropna(axis=0,inplace=True)

import matplotlib.pyplot as plt

text_word_count=[]
summary_word_count=[]

for i in data['cleaned_text']:
    text_word_count.append(len(i.split()))

for i in data['cleaned_summary']:
    summary_word_count.append(len(i.split()))
 
'''
length_df=pd.DataFrame({'text':text_word_count,'summary':summary_word_count})
length_df.hist(bins=30)
plt.show()
'''

cnt=0
for i in data['cleaned_summary']:
    if len(i.split())<=10:
        cnt=cnt+1

#print(cnt/len(data['cleaned_summary']))


max_text_len=80
max_summary_len=10

cleaned_text=np.array(data['cleaned_text'])
cleaned_summary=np.array(data['cleaned_summary'])

short_text=[]
short_summary=[]

for i in range(len(cleaned_text)):
    if len(cleaned_text[i].split())<=max_text_len and len(cleaned_summary[i].split())<=max_summary_len:
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])


df=pd.DataFrame({'text':short_text,'summary':short_summary})

df['summary']=df['summary'].apply(lambda x:'sostok '+x+' eostok')

from sklearn.model_selection import train_test_split
#x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.1,random_state=0,shuffle=True)
x_tr,x_val,y_tr,y_val=train_test_split(data['cleaned_text'],data['cleaned_summary'],test_size=0.1,random_state=0,shuffle=True) 

#print(x_tr)

pickle.dump(x_tr,open('X_training_value.pkl','wb'))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

x_tokenizer=Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

cnt,tot_cnt,freq,tot_freq=rareword_coverage(4,x_tokenizer)

x_tokenizer=Tokenizer(num_words=tot_cnt-cnt)
x_tokenizer.fit_on_texts(list(x_tr))

x_tr_seq=x_tokenizer.texts_to_sequences(x_tr)



x_val_seq=x_tokenizer.texts_to_sequences(x_val)

x_tr=pad_sequences(x_tr_seq,maxlen=max_text_len,padding='post')
x_val=pad_sequences(x_val_seq,maxlen=max_text_len,padding='post')

x_voc=x_tokenizer.num_words+1

y_tokenizer=Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

cnt,tot_cnt,freq,tot_freq=rareword_coverage(6,y_tokenizer)
#print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
#print("Total Coverage of rare words:",(freq/tot_freq)*100)

y_tokenizer=Tokenizer(num_words=tot_cnt-cnt)
y_tokenizer.fit_on_texts(list(y_tr))

y_tr_seq=y_tokenizer.texts_to_sequences(y_tr)
y_val_seq=y_tokenizer.texts_to_sequences(y_val)

y_tr=pad_sequences(y_tr_seq,maxlen=max_summary_len,padding='post')
y_val=pad_sequences(y_val_seq,maxlen=max_summary_len,padding='post')

y_voc=y_tokenizer.num_words+1

ind=[]
for i in range(len(y_tr)):
    cnt=0
    for j in y_tr[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_tr=np.delete(y_tr,ind, axis=0)
#
x_tr=np.delete(x_tr,ind, axis=0)

ind=[]
for i in range(len(y_val)):
    cnt=0
    for j in y_val[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_val=np.delete(y_val,ind, axis=0)
x_val=np.delete(x_val,ind, axis=0)

from keras import backend as K
K.clear_session()

latent_dim=300
embedding_dim=100

encoder_inputs=Input(shape=(max_text_len,))
enc_emb=Embedding(x_voc,embedding_dim,trainable=True)(encoder_inputs)

encoder_lstm1=LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1,state_h1,state_c1=encoder_lstm1(enc_emb)

encoder_lstm2=LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2,state_h2,state_c2=encoder_lstm2(encoder_output1)

encoder_lstm3=LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs,state_h,state_c=encoder_lstm3(encoder_output2)

decoder_inputs=Input(shape=(None,))
dec_emb_layer=Embedding(y_voc,embedding_dim,trainable=True)
dec_emb=dec_emb_layer(decoder_inputs)

decoder_lstm=LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state,decoder_back_state=decoder_lstm(dec_emb,initial_state=[state_h,state_c])

attn_layer=AttentionLayer(name="attention_layer")
attn_out,attn_states=attn_layer([encoder_outputs,decoder_outputs])

decoder_concat_input=Concatenate(axis=-1,name='concat_layer')([decoder_outputs,attn_out])

decoder_dense=TimeDistributed(Dense(y_voc,activation='softmax'))
decoder_outputs=decoder_dense(decoder_concat_input)

model=Model([encoder_inputs,decoder_inputs],decoder_outputs)
model.summary()


model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy')
es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=2)

history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

from matplotlib import pyplot
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()
pyplot.show()

reverse_target_word_index=dict(map(reversed, y_tokenizer.word_index.items()))
reverse_source_word_index=dict(map(reversed, y_tokenizer.word_index.items()))
target_word_index=y_tokenizer.word_index
pickle.dump(target_word_index,open('target_word_index.pkl','wb'))
pickle.dump(reverse_target_word_index,open('reverse_target_word_index.pkl','wb'))
pickle.dump(reverse_source_word_index,open('reverse_source_word_index.pkl','wb'))


encoder_model=Model(inputs=encoder_inputs,outputs=[encoder_outputs,state_h,state_c])

decoder_state_input_h=Input(shape=(latent_dim,))
decoder_state_input_c=Input(shape=(latent_dim,))
decoder_hidden_state_input=Input(shape=(max_text_len,latent_dim))

dec_emb2=dec_emb_layer(decoder_inputs)
decoder_outputs2,state_h2,state_c2=decoder_lstm(dec_emb2,initial_state=[decoder_state_input_h,decoder_state_input_c])

attn_out_inf,attn_state_inf=attn_layer([decoder_hidden_state_input,decoder_outputs2])
decoder_inf_concat=Concatenate(axis=-1,name='concat')([decoder_outputs2,attn_out_inf])

decoder_outputs2=decoder_dense(decoder_inf_concat)
decoder_model=Model([decoder_inputs]+[decoder_hidden_state_input,decoder_state_input_h,decoder_state_input_c],[decoder_outputs2]+[state_h2,state_c2])


encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')

