import gym
import time
import random
import numpy as np

from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system('export CUDA_VISIBLE_DEVICES=""')

import gc

#from keras import backend


# Load Breakout Atari Game

# In[2]:


import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.7):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def run_on_cpu():

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    return tf.Session(config=config)

#KTF.set_session(get_session())
KTF.set_session( run_on_cpu() )

env = gym.make( 'BreakoutDeterministic-v4')

# Preprocess observation
# Crop image - do not keep score
# Grayscale image - reduce the rgb space to grayscale, save space

import cv2
import matplotlib.pyplot as plt

#get_ipython().magic('matplotlib inline')

def to_grayscale( observation ):
    r, g, b = observation[:,:,0], observation[:,:,1], observation[:,:,2]
    ret = 0.299 * r + 0.587 * g + 0.114 * b
    return ( np.array( ret, dtype=np.uint8 ) )

def preprocess_observation( observation ):
    res = cv2.resize( observation, (84,110) )
    crop = res[18:110-8:,:,:]
    grayscale = to_grayscale( crop )#cv2.cvtColor( crop, cv2.COLOR_BGR2GRAY )
    return ( grayscale )

"""
def preprocess_observation( observation ):
    res = cv2.resize( observation, (84,110) )#resize to 110x84
    crop = res[18:110-8,:,:]#crop image
    grayscale = to_grayscale( crop )
    thresh, bn = cv2.threshold( grayscale, 80, 255, cv2.THRESH_BINARY )
    #grayscale=cv2.cvtColor( crop, cv2.COLOR_BGR2GRAY )#apply grayscale
    #grayscale = grayscale.astype( float ) / 255.0#normalize image
    return ( np.array( bn / np.max(bn), dtype=np.uint8 ) )
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import initializers

# Build model

model = Sequential()

#32 filters of kernel(3,3), stride=4, input shape must be in format row, col, channels
#init='uniform',
model.add( Conv2D(32, (8,8), strides=(4,4), padding='same', input_shape=(84,84,4) ) )#deep mind
#model.add( Conv2D(16, (8,8), strides=(2,2), kernel_initializer=initializers.random_normal(stddev=0.01), padding='same', input_shape=(84,84,4) ) )

model.add( Lambda(lambda x: x / 255.0, dtype='float32') )

model.add( Activation( 'relu' ) )

model.add(Conv2D(64, (4,4), strides=(2,2), padding='same' ) )#deep min
#model.add(Conv2D(32, (4,4), strides=(2,2), kernel_initializer=initializers.random_normal(stddev=0.01), padding='same' ) )
model.add( Activation( 'relu' ) )

model.add(Conv2D(64, (3,3), strides=(1,1), kernel_initializer=initializers.random_normal(stddev=0.01), padding='same' ) )
model.add( Activation( 'relu' ) )

model.add(Flatten())
model.add(Dense(512, kernel_initializer=initializers.random_normal(stddev=0.01), activation='relu'))
model.add(Dense(256, kernel_initializer=initializers.random_normal(stddev=0.01), activation='relu'))
model.add(Dense(128, kernel_initializer=initializers.random_normal(stddev=0.01), activation='relu'))
model.add( Dense( env.action_space.n, kernel_initializer=initializers.random_normal(stddev=0.01), activation='relu' ) )
#model.compile(RMSprop(), 'MSE')
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
learning_rate = 0.00025
model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=['accuracy'] )

model.summary()

init_state = preprocess_observation( env.reset() )

recent_frames = deque(maxlen=4)

for i in range( 4 ):
    recent_frames.append( init_state )

import time

gamma = 0.99
alpha = 1#0.999999#00025

max_reward = 0.0

epoch = 0

start_episode = 1

epsilon = 1
epsilon_min = 0.1

exploration_steps = 1000000

epsilon_discount = ( epsilon - epsilon_min ) / exploration_steps

MAX_SIZE = 40000#capacity of deque
MIN_MIN_SIZE = 20000#min size for replay
D = deque( maxlen=MAX_SIZE )#[]

frames = 0

def load_deque():
    global D
    pkl_file = open( 'mydeque.pkl', 'rb')
    D = pickle.load( pkl_file )
    pkl_file.close()

def save_deque():
    output = open( 'mydeque.pkl', 'wb' )
    pickle.dump( D, output )
    output.close()

def load_dqn_model():
    global model
    from keras.models import model_from_json
    # load json and create model
    json_file = open('model_background.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_background.h5")
    print("Loaded model from disk")
    #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=['accuracy'] )

import pandas as pd
import pickle

episodes = []
rewards = []
epsilons = []
total_frames = []

def save_train():

    global episodes, rewards, epsilons, total_frames

    #save [episodes, rewards, epsilons ] to csv file
    d = {'episode': episodes, 'reward': rewards, 'epsilon': epsilons, 'total_frames': total_frames}
    df = pd.DataFrame(data=d, index=None)

    if not os.path.isfile('filename.csv'):
        df.to_csv('filename.csv',header ='column_names', index=None)
    else: # else it exists so append without writing the header
        df.to_csv('filename.csv',mode = 'a',header=False, index=None)

    episodes = []
    rewards = []
    epsilons = []
    total_frames = []

    #save model to disk
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_background.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_background.h5")
    print("Saved model to disk")

    #save deque to disk
    #save_deque()

def load_train():

    global start_episode, epsilon, frames

    #get last episode and epsilon
    if not os.path.isfile('filename.csv'):
        start_episode, epsilon = 1, 1
    else: # else it exists so append without writing the header
        df = pd.read_csv( 'filename.csv')

        if len(df) == 0:
            start_episode, epsilon, frames = 1, 1, 0
        else:
            epsilon = list( df['epsilon'].tail(1) )[0]
            start_episode = list( df['episode'].tail(1) )[0] + 1
            frames = list( df['total_frames'].tail(1) )[0]

    if os.path.isfile('model_background.json'):
        load_dqn_model()

    #if os.path.isfile('mydeque.pkl'):
        #load_deque()

load_train()

#print( start_episode, epsilon )
#print( type( start_episode ) )
#print( type( epsilon ) )
#print( model.summary() )
#print( D )

total_observe = 12000#total_episodes
MIN_SIZE = 32

observe_frame = 0

def must_observe():
    return ( observe_frame < MIN_MIN_SIZE )

def replay( ):

    if len( D ) < MIN_MIN_SIZE:
        return

    #print( "sample" )

    samples = random.sample( D, MIN_SIZE )

    all_x = []
    all_y = []

    for sample in samples:

        observation, reward, done, new_observation, action = sample

        y = model.predict( observation.reshape(  ( 1, 84, 84, 4) ) )

        Q_next = model.predict( new_observation.reshape(  ( 1, 84, 84, 4) ) )

        if done:
            y[0,action] += alpha * reward
        else:
            y[0,action] += alpha * ( reward + gamma * ( np.max( Q_next[0]  ) ) - y[0,action] )

        #print( y )

        neural_network_observation = observation.reshape(  ( 1, 84, 84, 4) )

        all_x.append( neural_network_observation )
        all_y.append( y )
        #model.fit( neural_network_observation, y, epochs=1, verbose=0 )
        #model.train_on_batch( neural_network_observation, y )

    all_x = np.array( all_x ).reshape( (MIN_SIZE,84,84,4) )
    all_y = np.array( all_y ).reshape( (MIN_SIZE,4) )

    model.train_on_batch( all_x, all_y )

    del all_x, all_y

start = time.time()

episode = start_episode

while episode <= total_observe:#3600*5):

    observation = env.reset()

    observation = preprocess_observation( observation )

    recent_frames = deque(maxlen=4)

    for i in range( 4 ):
        recent_frames.append( observation )

    total_reward = 0

    #print( episode )

    cur_lives = 5

    step = 0

    action = 0

    steps = 0

    while True:

        #env.render()

        stack_observation = np.stack(recent_frames,axis=0)

        if must_observe():
            observe_frame += 1

        if must_observe() == False:
            steps += 1

        #if step % 4 == 0:
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            Q = model.predict( stack_observation.reshape(  ( 1, 84, 84, 4) ) )[0]
            action = np.argmax( Q )
        #step = 0

        new_observation, reward, done, info = env.step( action )

        new_observation = preprocess_observation( new_observation )#apply preprocess

        #plt.imshow( new_observation )

        #plt.show(block=False)
        #plt.pause(0.5)#.sleep(3)
        #plt.close()

        next_recent_frames = recent_frames.copy()
        next_recent_frames.append( new_observation )
        next_new_observation = np.stack(next_recent_frames,axis=0)

        memory_reward = reward

        if info['ale.lives'] < cur_lives:
            cur_lives = info['ale.lives']
            memory_reward = -1

        D.append( ( stack_observation, memory_reward, done, next_new_observation, action ) )

        total_reward += reward

        replay()

        if done:
            #print(  str(episode) + "Game over!", end= ' ' ),
            #replay()]]]
            if must_observe() == False:
                episodes.append( episode )
                rewards.append( total_reward )
                epsilons.append( epsilon )
            break

        observation = new_observation

        recent_frames.append( observation )

        if must_observe() == False:
            epsilon = max( epsilon_min, epsilon - epsilon_discount )

    if must_observe() == False:
        frames += steps
        total_frames.append( frames )
        print( "Episode " + str(episode) + " | total reward := " + str(total_reward) + " | steps := " + str(steps) + " total frames := " + str(frames) )
    else:
        print( "Observe total frames := " + str(observe_frame) )

    if episode % 10 == 0 and episode > 1:
        if must_observe() == False:
            save_train()
            gc.collect()

    if must_observe() == False:
        episode += 1

end = time.time()

print("total time is " + str( end - start ) )
