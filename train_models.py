## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os
import tensorflow as tf
import numpy as np
import time
from PIL import Image
from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

def train(data, file_name, params, num_epochs=50, batch_size=32, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    # model.add(Dense(10))
    model.add(Dense(3))
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

  # here we have to 1.perturb train_data 2.change train_labels.
   
    with tf.Session() as sess:
      data, trained_model =  MNIST(), MNISTModel("models/mnist", sess)
      #data, trained_model =  CIFAR(), CIFARModel("models/cifar", sess)
      attack_L2 = CarliniL2(sess, trained_model, batch_size=6, max_iterations=1000, confidence=0)
      attack_L0 = CarliniL0(sess, trained_model, max_iterations=1000, initial_const=10,
                      largest_const=15)
      attack_Linf = CarliniLi(sess, trained_model, max_iterations=100)

      inputs, targets = data.train_data, data.train_labels
      timestart = time.time()
      # adv = attack.attack(inputs, targets)
      # print("L0 runnig...")
      # adv_L0= attack_L0.attack(inputs[0:3000],targets[0:3000])
      
      # np.save('/content/nn_robust_attacks/adv_L0', adv_L0)
      # adv_L0 = np.load('/content/nn_robust_attacks/adv_L0.npy')
      # adv_L2 = np.load('/content/nn_robust_attacks/adv_L2.npy')
      # adv_Linf = np.load('/content/nn_robust_attacks/adv_Linf.npy')
      # adv_data_train=np.concatenate((adv_L0,adv_L2,adv_Linf))
      # np.save('/content/nn_robust_attacks/adv_data_train', adv_data_train)
      # print(adv_data_train.shape)
      # return
      # print("L2 runnig...")

      # adv_L2= attack_L2.attack(inputs[3000:6000],targets[3000:6000])
      # np.save('/content/nn_robust_attacks/adv_L2', adv_L2)
      # timeend = time.time()
      # print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
      # return
      # print("Linf runnig...")
      # adv_Linf= attack_Linf.attack(inputs[6000:9000],targets[6000:9000])
      # np.save('/content/nn_robust_attacks/adv_Linf', adv_Linf)
      # adv_data= np.concatenate((adv_L0, adv_L2, adv_Linf))
      # print(adv_data.shape)
      # return
      # adv_label_train=np.zeros((9000,3))


      # print(data.validation_data)
      # print(data.validation_labels.shape)
      # print(data.train_labels.shape)
      
      #label
      # for i in range(len(targets[0:3000])):
      #   adv_label_train[i]=np.array([1,0,0])
      # for i in range(len(targets[3000:6000])):
      #   adv_label_train[i+3000]=np.array([0,1,0])
      # for i in range(len(targets[6000:9000])):
      #   adv_label_train[i+6000]=np.array([0,0,1])
      
      # np.save('/content/nn_robust_attacks/adv_label_train', adv_label_train)
     
      # return
          
   
      # for i in range(0,len(inputs[0:10])) :
      #   data= inputs[i]
      #   data = data.reshape(28,28)
      #   rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

      #   im = Image.fromarray(rescaled)
      #   im.save("/content/nn_robust_attacks/perturbed/"+"imges"+"Test" + str(i)+ ".png")
      # for i in range(0,len(adv_data)) :
      #   data= adv_data[i]
      #   data = data.reshape(28,28)
      #   rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

      #   im = Image.fromarray(rescaled)
      #   im.save("/content/nn_robust_attacks/perturbed/"+"Pimges"+"Test" + str(i)+ ".png")
      # timeend = time.time()
      # print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
      # return 

      #validation 
      # print("L0 validation runnig...")
      # adv_L0_validation= attack_L0.attack(inputs[9000:9300],targets[9000:9300])
      
      # np.save('/content/nn_robust_attacks/adv_L0_validation', adv_L0_validation)
      
      
      # print("L2 validation runnig...")

      # adv_L2_validation= attack_L2.attack(inputs[9300:9600],targets[9300:9600])
      # np.save('/content/nn_robust_attacks/adv_L2_validation', adv_L2_validation)

      
      
      # print("Linf validation runnig...")
      # adv_Linf_validation= attack_Linf.attack(inputs[9600:9900],targets[9600:9900])
      # np.save('/content/nn_robust_attacks/adv_Linf_validation', adv_Linf_validation)
      # adv_data_validation= np.concatenate((adv_L0_validation, adv_L2_validation, adv_Linf_validation))
      # print(adv_data_validation.shape)
      # np.save('/content/nn_robust_attacks/adv_data_validation', adv_data_validation)
      # return

      #concatenate validation 
      # adv_L0_validation = np.load('/content/nn_robust_attacks/adv_L0_validation.npy')
      # adv_L2_validation = np.load('/content/nn_robust_attacks/adv_L2_validation.npy')
      # adv_Linf_validation = np.load('/content/nn_robust_attacks/adv_Linf_validation.npy')
      # adv_data_validation=np.concatenate((adv_L0_validation,adv_L2_validation,adv_Linf_validation))
      # np.save('/content/nn_robust_attacks/adv_data_validation', adv_data_validation)
    
      
      #label_validation
      # adv_label_validation=np.zeros((900,3))
      # for i in range(300):
      #   adv_label_validation[i]=np.array([1,0,0])
      # for i in range(300):
      #   adv_label_validation[i+300]=np.array([0,1,0])
      # for i in range(300):
      #   adv_label_validation[i+600]=np.array([0,0,1])
      
      # np.save('/content/nn_robust_attacks/adv_label_validation', adv_label_validation)
      
      # return
      
      # for i in range(len(adv)):
          # print("Valid:")
          #show(inputs[i])
          #print("Adversarial:")
          #show(adv[i])
          
          # print("Classification:", model.model.predict(adv[i:i+1]))

          # print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

    ###### 
      adv_data_train = np.load('/content/nn_robust_attacks/adv_data_train.npy')
      adv_label_train= np.load('/content/nn_robust_attacks/adv_label_train.npy')
      adv_data_validation= np.load('/content/nn_robust_attacks/adv_data_validation.npy')
      adv_label_validation= np.load('/content/nn_robust_attacks/adv_label_validation.npy')
      # print(adv_data_train.shape)
      # print(adv_label_train.shape)
      # print(adv_data_validation.shape)
      # print(adv_label_validation.shape)            
      # return
      model.fit(adv_data_train, adv_label_train,
                batch_size=batch_size,
                validation_data=(adv_data_validation, adv_label_validation),
                nb_epoch=num_epochs,
                shuffle=True)
      

    
      model.save(file_name)

    

# def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
#     """
#     Train a network using defensive distillation.

#     Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
#     Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
#     IEEE S&P, 2016.
#     """
#     if not os.path.exists(file_name+"_init"):
#         # Train for one epoch to get a good starting point.
#         train(data, file_name+"_init", params, 1, batch_size)
    
#     # now train the teacher at the given temperature
#     teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,
#                     init=file_name+"_init")

#     # evaluate the labels at temperature t
#     predicted = teacher.predict(data.train_data)
#     with tf.Session() as sess:
#         y = sess.run(tf.nn.softmax(predicted/train_temp))
#         print(y)
#         data.train_labels = y

#     # train the student model at temperature t
#     student = train(data, file_name, params, num_epochs, batch_size, train_temp,
#                     init=file_name+"_init")

#     # and finally we predict at temperature 1
#     predicted = student.predict(data.train_data)

#     print(predicted)
    
if not os.path.isdir('models'):
    os.makedirs('models')

# train(CIFAR(), "models/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)
train(MNIST(), "models/NormClassifier", [32, 32, 64, 64, 200, 200], num_epochs=200)

# train_distillation(MNIST(), "models/mnist-distilled-100", [32, 32, 64, 64, 200, 200],
                  #  num_epochs=50, train_temp=100)
# train_distillation(CIFAR(), "models/cifar-distilled-100", [64, 64, 128, 128, 256, 256],
                  #  num_epochs=50, train_temp=100)
