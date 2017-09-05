import gym
import time
import random
import numpy as np

from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system('export CUDA_VISIBLE_DEVICES=""')

#from keras import backend


# Load Breakout Atari Game

# In[2]:


import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())


# In[3]:


env = gym.make( 'Breakout-v0')
#env = gym.make( 'CartPole-v0' )


# Get initial state

# In[4]:


observation = env.reset()


# In[5]:


observation.shape
#210*160*3*4


# In[6]:


print( np.prod( observation.shape ) )


# Preprocess observation
# Crop image - do not keep score
# Grayscale image - reduce the rgb space to grayscale, save space

# In[7]:


import cv2
import matplotlib.pyplot as plt

#get_ipython().magic('matplotlib inline')

def preprocess_observation( observation ):
    res = cv2.resize( observation, (84,110) )
    crop = res[18:110-8:,:,:]
    grayscale = cv2.cvtColor( crop, cv2.COLOR_BGR2GRAY )
    return ( grayscale )

#plt.imshow( preprocess_observation( observation ), cmap='gray')

#plt.show()

preprocess_observation( observation ).shape


# In[9]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop

# Build model

model = Sequential()

#32 filters of kernel(3,3), stride=4, input shape must be in format row, col, channels
#init='uniform',
model.add( Conv2D(32, (8,8), strides=(4,4), padding='same', input_shape=(84,84,4) ) )
model.add( Activation( 'relu' ) )

model.add(Conv2D(64, (4,4), strides=(2,2), padding='same' ) )
model.add( Activation( 'relu' ) )

model.add(Conv2D(64, (3,3), strides=(1,1), padding='same' ) )
model.add( Activation( 'relu' ) )

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add( Dense( env.action_space.n, kernel_initializer='uniform', activation='linear' ) )
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
alpha = 0.1

max_reward = 0.0

start_episode = 1

epsilon = 1
epsilon_min = 0.1

exploration_steps = 1000000

epsilon_discount = ( epsilon - epsilon_min ) / exploration_steps

MAX_SIZE = 10000#capacity of deque
MIN_MIN_SIZE = 1000#min size for replay
D = deque( maxlen=MAX_SIZE )#[]

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

def save_train():

    global episodes, rewards, epsilons

    #save [episodes, rewards, epsilons ] to csv file
    d = {'episode': episodes, 'reward': rewards, 'epsilon': epsilons}
    df = pd.DataFrame(data=d, index=None)

    if not os.path.isfile('filename.csv'):
        df.to_csv('filename.csv',header ='column_names', index=None)
    else: # else it exists so append without writing the header
        df.to_csv('filename.csv',mode = 'a',header=False, index=None)

    episodes = []
    rewards = []
    epsilons = []

    #save model to disk
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_background.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_background.h5")
    print("Saved model to disk")

    #save deque to disk
    save_deque()

def load_train():

    global start_episode, epsilon

    #get last episode and epsilon
    if not os.path.isfile('filename.csv'):
        start_episode, epsilon = 1, 1
    else: # else it exists so append without writing the header
        df = pd.read_csv( 'filename.csv')

        if len(df) == 0:
            start_episode, epsilon = 1, 1
        else:
            epsilon = list( df['epsilon'].tail(1) )[0]
            start_episode = list( df['episode'].tail(1) )[0] + 1

    if os.path.isfile('model_background.json'):
        load_dqn_model()

    if os.path.isfile('mydeque.pkl'):
        load_deque()

load_train()

#print( start_episode, epsilon )
#print( type( start_episode ) )
#print( type( epsilon ) )
#print( model.summary() )
#print( D )

total_observe = 12000#total_episodes
MIN_SIZE = 32

def replay( ):

    if len( D ) < MIN_MIN_SIZE:
        return

    samples = random.sample( D, MIN_SIZE )

    for sample in samples:

        observation, reward, done, new_observation, action = sample

        y = model.predict( observation.reshape(  ( 1, 84, 84, 4) ) )

        Q_next = model.predict( new_observation.reshape(  ( 1, 84, 84, 4) ) )

        if done:
            y[0,action] += alpha * reward
        else:
            y[0,action] += alpha * ( reward + gamma * ( np.max( Q_next[0]  ) ) - y[0,action] )

        neural_network_observation = observation.reshape(  ( 1, 84, 84, 4) )
        model.fit( neural_network_observation, y, epochs=1, verbose=0 )
        #model.train_on_batch( neural_network_observation, y )

start = time.time()

for episode in range( start_episode, total_observe+1 ):#3600*5):

    observation = env.reset()

    observation = preprocess_observation( observation )

    recent_frames = deque(maxlen=4)

    for i in range( 4 ):
        recent_frames.append( observation )

    total_reward = 0

    #print( episode )

    while True:

        env.render()

        stack_observation = np.stack(recent_frames,axis=0)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            Q = model.predict( stack_observation.reshape(  ( 1, 84, 84, 4) ) )[0]
            action = np.argmax( Q )

        new_observation, reward, done, info = env.step( action )

        new_observation = preprocess_observation( new_observation )#apply preprocess

        next_recent_frames = recent_frames.copy()
        next_recent_frames.append( new_observation )
        next_new_observation = np.stack(next_recent_frames,axis=0)

        D.append( ( stack_observation, reward, done, next_new_observation, action ) )

        total_reward += reward

        if done:
            print(  str(episode) + "Game over!", end= ' ' ),
            replay()
            episodes.append( episode )
            rewards.append( total_reward )
            epsilons.append( epsilon )
            break

        observation = new_observation

        recent_frames.append( observation )

        epsilon = max( epsilon_min, epsilon - epsilon_discount )

    print( "episode " + str(episode) + " done with total reward := " + str(total_reward) )

    if episode % 100 == 0 and episode > 1:
        save_train()

end = time.time()

print("total time is " + str( end - start ) )