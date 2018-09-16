import tflearn
from tflearn.layers.conv import conv_2d
from tflearn.layers.core import fully_connected, input_data
from tflearn.layers.estimator import regression

from tqdm import tqdm
import numpy as np
from utils import INPUT_SHAPE, batch_generator, load_data
import argparse
import os



#command line arguments
parser = argparse.ArgumentParser(description='Autonomous driving car')
parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
parser.add_argument('-b', help='batch size for image processing',dest='batch_size',type=int,   default=50)
args = parser.parse_args()

print('---------------------------------------------------------------------------------------------------------------------------')
for arg in vars(args):
    print (arg, getattr(args, arg))
print('---------------------------------------------------------------------------------------------------------------------------')
print(' ')

#loading data
X,Y=load_data(args)


print(" ")
j=0
for i in tqdm(range(1000)):
    x_temp,y_temp=batch_generator(args.data_dir,X,Y,20,True)
    if j==0:
        X_train=x_temp
        Y_train=y_temp
        j+=1
    else:
        X_train=np.concatenate((X_train,x_temp))
        Y_train=np.concatenate((Y_train,y_temp))


print(" ")
print("**********************************************data loading completed*****************************************************")
print("--------------------------------------------train:test split is 80:20----------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------------------------")

#model
X_train=X_train.reshape([-1,66,200,3])
Y_train=Y_train.reshape([-1,1])

#input layer
network=input_data(shape=[None,66,200,3], name='input')

#convolutional layers
network=conv_2d(network, 24, activation='elu', strides=2, filter_size=5)
network=conv_2d(network, 36, activation='elu', strides=2, filter_size=5)
network=conv_2d(network, 48, activation='elu', strides=2, filter_size=5)
network=conv_2d(network, 64, activation='elu', filter_size=3)
network=conv_2d(network, 64, activation='elu', filter_size=3)
#fully connected layers
network=fully_connected(network, 100, activation='elu')
network=fully_connected(network, 50, activation='elu')
network=fully_connected(network, 10, activation='elu')
network=fully_connected(network, 1)
network=regression(network,optimizer='adam', learning_rate=0.0001, loss='mean_square', name='targets')

model=tflearn.DNN(network)


model.fit({'input':X_train},{'targets':Y_train},shuffle=True,n_epoch=10,validation_set=0.2,snapshot_step=500,show_metric=True,run_id='autonomous_driving_car')

model.save('conv.model')
print("model saved as :conv.model")
