import numpy as np 
import pandas as pd 

dataset=pd.read_csv("/media/karthik/EC742F3F742F0C40/Friends_tech/gesture-recognition/speechprocessing/emotion_dataset/emotion.csv")
print(dataset.head())

dataset.emotions.value_counts().plot.bar()
x_train=[text.split(" ") for text in dataset["text"].values.tolist()]
#print(x_train[0:9])
y_train=dataset["emotions"].values.tolist()
#print(y_train[0:9])


word2id=dict()
label2id=dict()
max_words=0

for senetance in x_train:
    for word in senetance:
        if word not in word2id:
            word2id[word]=len(word2id)
    if len(senetance)>max_words:
        max_words=len(senetance)
#print(worslib)

label2id={l:i for i ,l in enumerate(set(y_train))}
id2label={v:k for k , v in label2id.items()}
print(label2id)

import tensorflow as tf
import keras


x=[[word2id[word] for word in senetance] for word in x_train]
y=[label2id[label] for label in y_train]


from keras.preprocessing.sequence import pad_sequences
x=pad_sequences(x,max_words)

y=keras.utils.to_categorical(y,num_classes=len(label2id),dtype='float32')
print("shape of x :{}".format(x.shape))
print("shape of x :{}".format(y.shape))





