import pickle

import numpy as np

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam

import keras

from keras.utils import plot_model
import matplotlib.pyplot as plt

from keras.models import Model 
from keras.callbacks import Callback

from scipy import interp

#import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve,auc

#from sklearn import metrics

from itertools import cycle

import random

from PIL import Image as PILImage

import tensorflow as tf
from conv_model import conv_model

#from keras.preprocessing import image 
#
#import pylab


batch_size =80

epochs =200

Sclass = 5


X=np.load('data.npy')      
y=np.load('label.npy')
y=np.squeeze(y.reshape(1, -1))
result = [a - 1 for a in y]
y=result

#a=X[:,2,:,:]
#a=a.reshape(5266,1,32,32)
#print(a.shape)
#
#p=np.concatenate((a,a,a),axis=1)
#p=p.reshape(5266,3,32,32)
#X=p
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

x_train = x_train.reshape(-1,3,32,32)/255.
x_train = x_train.transpose(0,3,1,2)
x_train = x_train.transpose(0,3,1,2)
#print(x_train.shape)
#x_val = x_val.reshape(-1,3,32,32)/255.
#x_val = x_val.transpose(0,3,1,2)
#x_val = x_val.transpose(0,3,1,2)

x_test = x_test.reshape(-1,3,32,32)/255.
x_test = x_test.transpose(0,3,1,2)              
x_test = x_test.transpose(0,3,1,2)

y_train = keras.utils.to_categorical(y_train,num_classes=Sclass)
y_test = keras.utils.to_categorical(y_test, num_classes=Sclass)

model = conv_model()

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]} 
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))

        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
#        val_pred = self.model.predict(self.validation_data[0])
#        val_targ = np.argmax(self.validation_data[1], axis=1)
#        roc = roc_auc_score(val_targ, val_pred,average='macro')   
        print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
#        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
#            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="upper right")
        plt.savefig('./acc3.eps', format='eps', dpi=1000)
        
    def loss_plot1(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
#        plt1.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
#            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig('./loss2.eps', format='eps', dpi=1000)
        
#    def loss_plot2(self, loss_type):
#        iters = range(len(self.losses[loss_type]))
#        plt.figure()
#        # acc
##        plt1.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
#        # loss
#        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
#        if loss_type == 'epoch':
#            # val_acc
##            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
#            # val_loss
#            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
#        plt.grid(True)
#        plt.xlim([25, 200])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel(loss_type)
#        plt.ylabel('loss')
#        plt.legend(loc="upper right")
#        plt.savefig('./loss1.eps', format='eps', dpi=1000)

history = LossHistory()

model.compile(loss='categorical_crossentropy',

              optimizer=adam,

              metrics=['accuracy'])

#json_string = model.to_json()
#with open('mlp_model.json','w') as of:
#    of.write(json_string)

checkpointer = ModelCheckpoint(filepath="./test2/weights.hdf5", verbose=1,monitor='val_loss',save_best_only=True, save_weights_only=True)

try:

    model.fit(x_train, y_train,

                batch_size=batch_size,

                epochs=epochs,

                validation_split=0.2,shuffle=True,

               callbacks=[history,checkpointer]

               )

except KeyboardInterrupt:

    print("training interrupted")

history.loss_plot('epoch')
history.loss_plot1('epoch')


#model.load_weights("./test2/weights.hdf5")
score = model.evaluate(x_test, y_test,verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


y_score = model.predict_proba(x_test)


for i in range(len(y_score)):

    max_value=max(y_score[i])

    for j in range(len(y_score[i])):

        if max_value==y_score[i][j]:

            y_score[i][j]=1

        else:

            y_score[i][j]=0

print(classification_report(y_test, y_score,digits=4))


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(Sclass):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
 
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Sclass)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(Sclass):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= Sclass

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(3)
lw = 0.5
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve'
               ''.format(roc_auc["macro"]),
         color='navy',lw=lw)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','brown'])
for i, color in zip(range(Sclass), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})' 
             ''.format(i,roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--',linestyle=':', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('./roc.eps', format='eps', dpi=1000)

plt.figure(4)
lw = 0.5
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve)'
               ''.format(roc_auc["macro"]),
         color='navy',lw=lw)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','brown'])
for i, color in zip(range(Sclass), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0}' 
             ''.format(i))
  
plt.plot([0, 1], [0, 1], lw=lw,linestyle=':')
plt.xlim(0,0.1)
plt.ylim(0.9, 1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('./roc3.eps', format='eps', dpi=1000)

layer_model=Model(inputs=model.input,outputs=model.layers[5].output)
layer_model1=Model(inputs=model.input,outputs=model.layers[8].output)
#
x= x_train[45:54]
features=layer_model.predict(x)
print(features.shape)
features1 = layer_model1.predict(x)

x= x*0.5+0.5
features= features*0.5+0.5
features1 = features1*0.5+0.5
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(x[i])
    plt.axis('off')
plt.savefig('./raw4.eps', format='eps', dpi=1000)
#   
for i in range(9):
#    plt.figure()
    plt.subplot(3, 3, i+1)
#    features[i] =PILImage.fromarray(np.asarray(features[i])) 
    plt.imshow(features[i])
    plt.axis('off')
plt.savefig('./lcex4.eps', format='eps', dpi=1000)

for i in range(9):
#    plt.figure()
    plt.subplot(3, 3, i+1)
#    features[i] =PILImage.fromarray(np.asarray(features[i])) 
    plt.imshow(features1[i])
    plt.axis('off')
plt.savefig('./stnex3.eps', format='eps', dpi=1000)






