import os
if os.name == 'nt': 
    _ = os.system('cls') 
else: 
    _ = os.system('clear')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def nanProcessing(lst):
    return [i for i, val in enumerate(lst.isnull()) if val]
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Survived')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

training_set = pd.read_csv("train.csv",na_values='')
##Assign NaN with data type
training_set.loc[nanProcessing(training_set.Cabin), 'Cabin']='Nan'
##Fill Age with Median
training_set.loc[nanProcessing(training_set.Age), 'Age'] = np.nanmedian(training_set.Age)
##Data Processing
xValue = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
training = training_set[xValue]

#Process data for TensorFlow
pd.options.mode.chained_assignment = None

for col in ['Sex', 'Cabin', 'Embarked']:
    training[col] = pd.Categorical(training[col])
    training[col] = training[col].cat.codes
pd.options.mode.chained_assignment = 'warn'

print(training.head())

#Split data
split_size = 0.1
train, test = train_test_split(training, test_size=split_size)
train, val = train_test_split(train, test_size=split_size)


#Convert dataframe to tensorflow
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#Feature Columns:
feature_columns = []
# numeric cols
for header in ['Age', 'Fare']:
  feature_columns.append(feature_column.numeric_column(header))
# categorical cols
cateColList = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']
for cols in cateColList:
    colVal = feature_column.categorical_column_with_vocabulary_list(cols, training[col].unique())
    feature_columns.append(feature_column.indicator_column(colVal))
    
#Creating Feature Layer:
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

results = model.evaluate(test_ds)

testing_set = pd.read_csv("test.csv",na_values='')
passengerId = testing_set.PassengerId
testing_result = pd.read_csv("gender_submission.csv", na_values='')
##Assign NaN with data type
testing_set.loc[nanProcessing(testing_set.Cabin), 'Cabin']='Nan'
##Fill Age with Median
testing_set.loc[nanProcessing(testing_set.Age), 'Age'] = np.nanmedian(training_set.Age)
pd.options.mode.chained_assignment = None
xValue = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
testing_set = testing_set[xValue]
for col in ['Sex', 'Cabin', 'Embarked']:
    testing_set[col] = pd.Categorical(testing_set[col])
    testing_set[col] = testing_set[col].cat.codes
pd.options.mode.chained_assignment = 'warn'
testing_ds = tf.data.Dataset.from_tensor_slices((dict(testing_set), testing_result['Survived'])).batch(batch_size)
predictions = model.predict(testing_ds)
final_result = []
accuracy = 0
counter = 0
for val in predictions:
    if val[0] < 0:
        final_result.append(0)
        if testing_result.Survived[counter] == 0:
            accuracy = accuracy + 1
    else:
        final_result.append(1)
        if testing_result.Survived[counter] == 1:
            accuracy = accuracy + 1
    counter = counter + 1

print(accuracy)
print(accuracy / len(final_result))

data_tuples = list(zip(passengerId,final_result))
finalResult = pd.DataFrame(data_tuples, columns=['PassengerId','Survived'])
finalResult.to_csv('MySubmission.csv', index = False)

