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


embeding_dim=100 #The dimention of the word embedings

sequence_input=keras.Input(shape=(max_words,),dtype='int32')

embeded_input=keras.layers.Embedding(len(word2id)+1,embeding_dim,input_length=max_words)(sequence_input)
embeded_input=keras.layers.Dropout(0.2)(embeded_input)

lstm_out=keras.layers.wrappers.Bidirectional(
    keras.layers.LSTM(embeding_dim,return_sequences=True)
)(embeded_input)

lstm_out=keras.layers.Dropout(0.2)(lstm_out)


lstm_out=keras.layers.wrappers.Bidirectional(
    keras.layers.LSTM(embeding_dim,return_sequences=True)
)(embeded_input)


fc=keras.layers.Dense(embeding_dim,activation='relu')(attention_output)
output=keras.layers.Dense(len(label2id),activation='softmax')(fc)


model=keras.Model(inputs=[sequence_input],outputs=output)
model.compile(loss="categorical_crossentropy",metrics=["accuracy"],optimizer="adam")


model.summary()

model.fit(x,y,batch_size=64,validation_split=0.1,shuffle=True)


