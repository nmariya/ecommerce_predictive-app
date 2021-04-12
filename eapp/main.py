# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Conv2D, Dense, Dropout, Flatten, Activation, Input, BatchNormalization, Model
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from nltk.corpus import stopwords
import string
import csv
import re
import math
from nltk.stem import PorterStemmer
import os, sys, math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm
import warnings
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens
with open('/kaggle/input/uw-cs480-fall20/train.csv', 'r') as f:
    reader =  csv.reader(f)
    next(reader)
    with open('train_new.txt', 'w') as g:
        writer = csv.writer(g)
        for row in reader:
            new_row = [' '.join([row[2].lower(), row[3].lower(), row[4].lower(), row[5].lower(), row[6].lower()])]
            writer.writerow(new_row)
with open('/kaggle/input/uw-cs480-fall20/train.csv', 'r') as f:
    train = pd.read_csv(f)
all_categories = sorted(list(train['category'].unique()))
print(all_categories)
total = len(train['category'])
train_category = train['category'][0:int(math.floor(0.85*total))]
val_category = train['category'][int(math.floor(0.85*total)):]
ps = PorterStemmer()
with open('train_new.txt', 'r') as f:
    train_new = f.readlines()
for i in range(len(train_new)):
    train_new[i] = clean_doc(train_new[i])
    train_new[i] = [ps.stem(token) for token in train_new[i]]
re = [' '.join(ele) for ele in train_new]
res1 = (" ".join(re))
res2=sorted(set(res1.split()))
n_words = len(res2)
validation = re[int(math.floor(0.85*total)):]
res = re[0:int(math.floor(0.85*total))]
def wordToIndex(word):
    if word in res2:
        return res2.index(word)
    else:
        return -1
def wordToTensor(word):
    tensor = torch.zeros(1, n_words)
    if wordToIndex(word) != -1:
        tensor[0][wordToIndex(word)] = 1
        return tensor
    else:
        return tensor    
def lineToTensor(line):
    l= line.split()
    tensor = torch.zeros(len(l), 1, n_words)
    for li, word in enumerate(l):
        tensor[li][0][wordToIndex(word)] = 1
    return tensor
In [8]:
n_categories = len(all_categories)
n_hidden = 128
class RNN_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_lstm, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.LSTMCell(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input1, hidden):
        combined = torch.cat((input1, hidden), 1)
        hidden = self.i2h(combined)[0]
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
rnn_lstm = RNN_lstm(n_words, n_hidden, n_categories)
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
criterion = nn.NLLLoss()
optimizer_lstm = torch.optim.Adam(rnn_lstm.parameters())
def TrainingExample(i):
    category = train_category[i]
    line = res[i]
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
def training(category_tensor, line_tensor):
    hidden_lstm = rnn_lstm.initHidden()

    # reset gradient
    rnn_lstm.zero_grad()

    for i in range(line_tensor.size()[0]):
        output_lstm, hidden_lstm = rnn_lstm(line_tensor[i], hidden_lstm)
    loss_lstm = criterion(output_lstm, category_tensor)

    # compute gradient by backpropagation
    loss_lstm.backward()

    # update parameters
    optimizer_lstm.step()

    return output_lstm, loss_lstm.item()
import time

n_iters = len(res)
print(n_iters)
print_every = 1000
plot_every = 1000

# Keep track of losses for plotting
train_loss_lstm = 0
all_train_losses_lstm = []
all_validation_losses_lstm = []
all_validation_losses2 = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Just return an output given a line
def evaluate(line_tensor):
    hidden_lstm = rnn_lstm.initHidden()
    for i in range(line_tensor.size()[0]):
        output_lstm, hidden_lstm = rnn_lstm(line_tensor[i], hidden_lstm)
    return output_lstm

def eval_dataset(dataset):
    loss_lstm = 0
    n_instances = 0
    confusion_lstm = torch.zeros(n_categories, n_categories)
    for i in range(len(dataset)):
        category_tensor = Variable(torch.LongTensor([all_categories.index(val_category[i+int(math.floor(0.85*total))])]))
        n_instances += len(dataset)
        line = dataset[i]
        line_tensor = Variable(lineToTensor(line))
        output_lstm = evaluate(line_tensor)
        loss_lstm += criterion(output_lstm, category_tensor)
        guess_lstm, guess_i_lstm = categoryFromOutput(output_lstm)
        category_i = all_categories.index(category)
        confusion_lstm[category_i][guess_i_lstm] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion_lstm[i] = confusion_lstm[i] / confusion_lstm[i].sum()

    return loss_lstm.item() / n_instances, confusion_lstm
  
print('\nIter \tTrain% \tTime \t\tTrain_loss_lstm \tExample')
start = time.time()
for iter in range(1, n_iters):
#     print(type(iter))
#     for i in range(len(res)):
    category, line, category_tensor, line_tensor = TrainingExample(iter)
    output_lstm, loss_lstm = training(category_tensor, line_tensor)
    train_loss_lstm += loss_lstm

    # Print iter number, train loss average, name and guess
    if iter % print_every == 0:
        guess_lstm, guess_i_lstm = categoryFromOutput(output_lstm)
        correct_lstm = '✓' if guess_lstm == category else '✗ (%s)' % category
        print(iter, iter / n_iters * 100, timeSince(start), train_loss_lstm/plot_every, line, guess_lstm, correct_lstm)

    # Add current train loss average to list of losses
    if iter % plot_every == 0:
        all_train_losses_lstm.append(train_loss_lstm / plot_every)
        train_loss_lstm = 0

    # Compute loss based on validation data
    if iter % plot_every == 0:
        average_validation_loss_lstm,_ = eval_dataset(validation)

        # save model with best validation loss
        if len(all_validation_losses_lstm) == 0 or average_validation_loss_lstm < min(all_validation_losses_lstm):
            torch.save(rnn_lstm, 'char_rnn_lstm_classification_model.pt')
        all_validation_losses_lstm.append(average_validation_loss_lstm)
        import csv
with open('/kaggle/input/uw-cs480-fall20/test.csv', 'r') as f:
    reader =  csv.reader(f)
    next(reader)
    with open('test_new.txt', 'w') as g:
        writer = csv.writer(g)
        for row in reader:
            new_row = [' '.join([row[1].lower(), row[2].lower(), row[3].lower(), row[4].lower(), row[5].lower()])]
            writer.writerow(new_row)
with open('test_new.txt', 'r') as f:
    test_new = f.readlines()
for i in range(len(test_new)):
#     print(type(test_new[i])
    test_new[i] = clean_doc(test_new[i])
    test_new[i] = [ps.stem(token) for token in test_new[i]]
te = [' '.join(ele) for ele in test_new]
def predict(input_line, n_predictions=1):
    with torch.no_grad():
        output_lstm = evaluate(lineToTensor(input_line))
    return output_lstm
prob_p = []
for i in range(len(re)):
        prediction = predict(re[i]).detach().cpu().numpy()
        prob_p.append(prediction)
prob_p=[e for sl in prob_p for e in sl]
for idx,i in enumerate(prob_p):
    b=np.amin(prob_p[idx])
    prob_p[idx]=prob_p[idx]/b
    prob_p[idx]=1-prob_p[idx]
prob_t = []
for i in range(len(te)):
        prediction_t = predict(te[i]).detach().cpu().numpy()
        prob_t.append(prediction_t)
prob_t=[e for sl in prob_t for e in sl]
for idx,i in enumerate(prob_t):
    b=np.amin(prob_t[idx])
    prob_t[idx]=prob_t[idx]/b
    prob_t[idx]=1-prob_t[idx]


warnings.filterwarnings("ignore")
INPUT_SHAPE = (299,299,3)
BATCH_SIZE = 10
import torch
def wordIndex(word):
    return cat.index(word)
path_to_train = '/kaggle/input/uw-cs480-fall20/suffled-images/shuffled-images'
data = pd.read_csv('/kaggle/input/uw-cs480-fall20/train.csv')

categories=[]
for name, labels in zip(data['id'], data['category'].str.split('  ')): 
    categories=categories+labels
cat=sorted(set(categories))

train_dataset_info = []
for name, label in zip(data['id'], data['category']): 
    labels = wordIndex(label)
    train_dataset_info.append({
        'path':os.path.join(path_to_train, str(name)),
        'labels':np.array(labels)}) 
train_dataset_info = np.array(train_dataset_info)
from sklearn.model_selection import train_test_split
train_ids, test_ids, train_targets, test_target = train_test_split(
    data['id'], data['category'], test_size=0.2, random_state=42)

class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True): 
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 27))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        image = np.array(Image.open(path+'.jpg'))
        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image  
                
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug

train_datagen = data_generator.create_train(
    train_dataset_info, 5, (299,299,3), augument=True)

images, labels = next(train_datagen)

fig, ax = plt.subplots(1,5,figsize=(25,5))
for i in range(5):
    ax[i].imshow(images[i])
print('min: {0}, max: {1}'.format(images.min(), images.max()))
def create_model(input_shape, n_out):
    
    pretrain_model = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape)    
    
    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model
def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    return K.mean(f1)
def show_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('acc')
    ax[2].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[2].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
keras.backend.clear_session()

model = create_model(
    input_shape=(299,299,3), 
    n_out=27)

model.summary()
checkpointer = ModelCheckpoint(
    '/kaggle/working/InceptionResNetV2.model',
    verbose=2, save_best_only=True)

train_generator = data_generator.create_train(
    train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=False)
validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

model.layers[2].trainable = False

model.compile(
    loss='binary_crossentropy',  
    optimizer=Adam(1e-3),
    metrics=['acc', f1])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=15, 
    verbose=1,
    callbacks=[checkpointer])
show_history(history)

train_generator = data_generator.create_train(
    train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

model.layers[2].trainable = True

model.compile(
    loss='binary_crossentropy',  
    optimizer=Adam(1e-4),
    metrics=['acc', f1])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=200, 
    verbose=1,
    callbacks=[checkpointer])

model = load_model(
    '/kaggle/working/InceptionResNetV2.model', 
    custom_objects={'f1': f1})
train_data = pd.read_csv('/kaggle/input/uw-cs480-fall20/train.csv')
predicted = []
prob_pred = []
labels = []
for name,label in zip(train_data['id'],train_data['category']):
    labels.append(label)
    path = os.path.join('/kaggle/input/uw-cs480-fall20/suffled-images/shuffled-images/', str(name))
    image = data_generator.load_image(path, INPUT_SHAPE)
    score_predict = model.predict(image[np.newaxis])[0]
    prob_pred.append(score_predict)
    label_predict = np.argmax(score_predict)
    predicted.append(label_predict)
test_data = pd.read_csv('/kaggle/input/uw-cs480-fall20/test.csv')
predicted_t = []
prob_pred_t = []
for name in tqdm(test_data['id']):
    path = os.path.join('/kaggle/input/uw-cs480-fall20/suffled-images/shuffled-images/', str(name))
    image = data_generator.load_image(path, INPUT_SHAPE)
    score_predict_t = model.predict(image[np.newaxis])[0]
    prob_pred_t.append(score_predict_t)


label_idx=[]
np.concatenate((prob_pred,prob_p),axis=1)
for idx,i in enumerate(labels):
    label_idx.append(wordIndex(i))
In [38]:
np.concatenate((prob_pred_t,prob_t),axis=1)

 
X_train, X_test, y_train, y_test = train_test_split(prob_pred, label_idx, random_state = 0) 

# training a linear SVM classifier 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions)
svm_predictions = svm_model_linear.predict(prob_pred_t)
with open('/kaggle/input/uw-cs480-fall20/test.csv', 'r') as f:
    test_data = pd.read_csv(f)
with open("output.csv", 'w+') as f:
    fieldnames = ['id', 'category']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i,ea in enumerate(svm_predictions):
        pred_cat=cat[ea]
        writer.writerow({'id': test_data['id'][i], 'category': pred_cat})