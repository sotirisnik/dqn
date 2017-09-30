import gym
import time
import random
import numpy as np

from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.system('export CUDA_VISIBLE_DEVICES=""')

import gc

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

env = gym.make( 'CartPole-v0')

import matplotlib.pyplot as plt

def preprocess_observation( observation ):
    return ( observation )

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Lambda
from keras.layers.convolutional import Conv2D, Convolution1D
from keras.optimizers import RMSprop, Adam
from keras import initializers

# Build model

DEQUE_LEN = 1#4

model = Sequential()

init_distr = "normal"
#lecun_uniform
model.add( Convolution1D( 24, (1), input_shape=(4,DEQUE_LEN), init=init_distr ) )
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, kernel_initializer=init_distr, activation='relu'))
model.add(Dense(64, kernel_initializer=init_distr, activation='relu'))
model.add(Dense(32, kernel_initializer=init_distr, activation='relu'))
model.add( Dense( env.action_space.n, kernel_initializer=init_distr, activation='linear' ) )

learning_rate = 0.001
model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=['accuracy'] )
#model.compile(Adam(lr=0.001), 'mse')
model.summary()

init_state = preprocess_observation( env.reset() )

recent_frames = deque(maxlen=DEQUE_LEN)

for i in range( DEQUE_LEN ):
    recent_frames.append( init_state )

import time

gamma = 0.95
alpha = 1#0.999999#00025

max_reward = 0.0

epoch = 0

start_episode = 1

epsilon = 1
epsilon_min = 0.1

exploration_steps = 5000#1000000

epsilon_discount = ( epsilon - epsilon_min ) / exploration_steps

MAX_SIZE = 2000#capacity of deque
MIN_MIN_SIZE = 1000#min size for replay
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

total_observe = 5000#total_episodes
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

        y = model.predict( observation.reshape( (1,4,DEQUE_LEN) ) )

        #print(y.shape)

        Q_next = model.predict( new_observation.reshape( (1,4,DEQUE_LEN) ) )

        #print( Q_next.shape )

        reward = np.clip( reward, -1, 1 )

        if done:
            y[0,action] = reward
        else:
            y[0,action] = reward + gamma * ( np.max( Q_next[0]  ) )

        #print( y )

        neural_network_observation = observation

        all_x.append( neural_network_observation )
        all_y.append( y )

    #print( np.array( all_x).shape )
    #print( np.array( all_y ).shape )

    all_x = np.array( all_x ).reshape( (MIN_SIZE,4,DEQUE_LEN) )
    all_y = np.array( all_y ).reshape( (MIN_SIZE,2) )

    #print( all_x )

    #model.train_on_batch( all_x, all_y )

    model.fit(all_x, all_y, epochs=1, batch_size=MIN_SIZE, verbose=0)

    del all_x, all_y

start = time.time()

episode = start_episode

while episode <= total_observe:

    observation = env.reset()

    observation = preprocess_observation( observation )

    recent_frames = deque(maxlen=DEQUE_LEN)

    for i in range( DEQUE_LEN ):
        recent_frames.append( observation )

    total_reward = 0

    #print( episode )

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
            Q = model.predict( stack_observation.reshape(  ( 1, 4, DEQUE_LEN) ) )[0]
            action = np.argmax( Q )
        #step = 0

        new_observation, reward, done, info = env.step( action )

        new_observation = preprocess_observation( new_observation )#apply preprocess

        next_recent_frames = recent_frames.copy()
        next_recent_frames.append( new_observation )
        next_new_observation = np.stack(next_recent_frames,axis=0)

        memory_reward = reward

        D.append( ( stack_observation, memory_reward, done, next_new_observation, action ) )

        total_reward += reward

        replay();

        if done:
            #print(  str(episode) + "Game over!", end= ' ' ),
            #replay()]]]
            if must_observe() == False:
                episodes.append( episode )
                rewards.append( total_reward )
                epsilons.append( epsilon )
            #D.append( ( stack_observation, -1, done, next_new_observation, action ) )
            break

        observation = new_observation

        recent_frames.append( observation )

        if must_observe() == False:
            epsilon = max( epsilon_min, epsilon - epsilon_discount )

    if must_observe() == False:
        frames += steps
        total_frames.append( frames )
        print( "Episode " + str(episode) + " | total reward := " + str(total_reward) + " | steps := " + str(steps) + " total frames := " + str(frames) + " epsilon := " + str(epsilon) )
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
