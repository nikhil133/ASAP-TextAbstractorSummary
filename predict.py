import tensorflow as tf
from attention import AttentionLayer
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from text_cleaner import text_cleaner,rareword_coverage
import pickle

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


max_text_len=30
max_summary_len=8
#test_value="Gave me such a caffeine overdose I had the shakes, a racing heart and an anxiety attack. Plus it tastes unbelievably bad. I'll stick with coffee, tea and soda, thanks."
test_value=input("Enter comment:")

cleaned_text=[]
cleaned_text.append(text_cleaner(test_value,0))
cleaned_text=np.array(cleaned_text)
short_text=[]
for i in range(len(cleaned_text)):
    if len(cleaned_text[i].split())<=max_text_len: 
        short_text.append(cleaned_text[i])

x_tr_test=short_text



file=open('X_training_value.pkl','rb')
x_trained_text=pickle.load(file)
file.close()

x_trained_text=np.append(x_trained_text,x_tr_test)

x_tokenizer=Tokenizer()
x_tokenizer.fit_on_texts(x_trained_text)

cnt,tot_cnt,freq,tot_freq=rareword_coverage(4,x_tokenizer)


x_tokenizer=Tokenizer(num_words=tot_cnt-cnt)
x_tokenizer.fit_on_texts(list(x_trained_text))

x_tr_seq=x_tokenizer.texts_to_sequences(x_tr_test)
x_tr=pad_sequences(x_tr_seq,maxlen=max_text_len,padding='post')

y_tokenizer=Tokenizer()
reverse_target_word_index=dict(map(reversed, y_tokenizer.word_index.items()))
file=open('reverse_target_word_index.pkl','rb')
reverse_target_word_index=pickle.load(file)
file.close()
file=open('reverse_source_word_index.pkl','rb')
reverse_source_word_index=pickle.load(file)
file.close()
file=open('target_word_index.pkl','rb')
target_word_index=pickle.load(file)
file.close()

max_summary_len=8
#target_word_index=y_tokenizer.word_index
encoder_model=load_model('encoder_model.h5',custom_objects={'AttentionLayer' : AttentionLayer})
decoder_model=load_model('decoder_model.h5',custom_objects={'AttentionLayer' : AttentionLayer})

def decode_sequence(input_seq):
    e_out,e_h,e_c=encoder_model.predict(input_seq)
    target_seq=np.zeros((1,1))
    target_seq[0,0]=target_word_index['sostok']

    stop_condition=False
    decoded_sentence=''
    while not stop_condition:
        output_tokens,h,c=decoder_model.predict([target_seq]+[e_out,e_h,e_c])
        sampled_token_index=np.argmax(output_tokens[0,-1,:])
        sampled_token=reverse_target_word_index[sampled_token_index]
        if(sampled_token!='eostok'):
            decoded_sentence+=' '+sampled_token
        if (sampled_token=='eostok') or len(decoded_sentence.split())>=(max_summary_len-1):
            stop_condition=True 
        
        target_seq=np.zeros((1,1))
        target_seq[0,0]=sampled_token_index
        e_h,e_c=h,c 
    return decoded_sentence

print("Predicted summary:",decode_sequence(x_tr.reshape(1,max_text_len)))