#Deep-Q-Learning with Super Mario Gym (OpenAI)

import numpy as np
import gym


env = gym.make('meta-SuperMarioBros-Tiles-v0')
from random import random, randint
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import thread
from keras import models as Models
from keras.models import Sequential, load_model
from keras.layers import Dense, LocallyConnected2D, Flatten, TimeDistributed,Dropout, Activation
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU as leakyrelu
from keras import backend as K


def _huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
num_models = 5
models = [0] * num_models
for i in range(num_models):
    models[i] = Sequential()
    models[i].add(Dense(512,activation='linear', input_dim=209))
    models[i].add(Dense(512,activation='softmax'))
    #models[i].add(leakyrelu())
    #models[i].add(Dropout(.05))
    models[i].add((Dense(256,activation='softmax')))
    #models[i].add(leakyrelu())
    #models[i].add(Dropout(.05))
    models[i].add((Dense(256,activation='softmax')))
    #models[i].add(leakyrelu())
    #models[i].add(Dropout(.05))
    models[i].add((Dense(64,activation='softmax')))
    #models[i].add(leakyrelu())
    models[i].add((Dense(20,activation='linear')))
    models[i].compile(loss=_huber_loss, optimizer=Adam(lr=0.005))

'''
model = Sequential()
model.add(TimeDistributed(Dense(128), input_shape=(13,16)))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dropout(.001))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(20, activation='relu'))
model.compile(loss=_huber_loss, optimizer=Adam(lr=0.001))  
'''  
debug = []
def ui(i):
    return np.unravel_index(i,(2,2,2,2,2,2))
def _save_model():
    print('Saving Models')
    for i in range(num_models):
        models[i].save('./DQMario_model'+str(i)+'.h5')

def _load_model(models, f):
    print('Loading Models')
    m=[0]*num_models
    for i in range(num_models):
        try:
            m[i] = load_model('./DQMario_model'+str(i)+'.h5',custom_objects={'_huber_loss':_huber_loss})
            #del models[i]
        except:
            print('Model not loaded, using default')
            m[i] = models[i]
    return m
    
class GymSuperMario(object):
    global debug
    def __init__(self, max_steps = 8000, max_score = 2000):
        self.data = deque(maxlen=20000)
        self.mod_save = True
        self.mod_load = False
        self.max_steps = max_steps
        self.max_score = max_score
        self.actions = []
        self.states = []
        self.rewards = []
        self.max_dist = 40.0
        self.dist = 40.0
        self.Q = None
        self.gamma = 0.9
        self.mb_size = 2000
        self.epsi = 0.8
        self.epsi_decay = 0.8
        self.epsi_min = 0.02
        self.history = []
        self.episode = 0
        self.Valid_Inputs = [i for i in range(64) if (ui(i)[0] + ui(i)[1] + ui(i)[2] + ui(i)[3]) <= 1]




        
    def _loop(self,state_then):
        self.episode += 1
        steps = 1
        self.dist = 40
        done = False
        reward = 0
        score = 0
        state_now = np.array([],dtype='int')
        state_then = state_then.flatten()
        state_new = state_then.flatten()
        self.db = state_now
        stuck = False
        self.rewards = []
        action_index = 0
        while not done:
            reward = 0
            steps += 5
            state_now = np.hstack((state_new,[action_index])).flatten().reshape((1,209))
            self.db = state_now
            actions = [0] * num_models
            vote_weights = [0] * num_models
            for i in range(num_models):
                Q = np.round(models[i].predict(state_now),10)[0]
                #print Q
                Q = np.nan_to_num(np.clip(Q,0,np.inf))
                plist = Q/np.sum(Q)
                try:                    
                    #plist = np.array([0 if x <= 0 else x for x in plist])
                    action_index = np.where(Q==np.random.choice(Q,p=plist))[0]
                    #print(action_index)
                    action_index = np.where(action_index==np.random.choice(action_index))[0] if len(action_index) > 1 else action_index
                    action_index = np.asscalar(action_index) if not np.isscalar(action_index) else action_index
                except:
                    print('error - action_index failed')
                    action_index = np.argmax(Q)
                
                if random() < self.epsi:
                    action_index = randint(0,19)
                actions[i] = action_index
                vote_weights[i] = float(Q[action_index])
                #print vote_weights
            #print actions
            #print vote_weights
            plist = vote_weights/np.sum(vote_weights)
            plist = np.array([0 if x <= 0 else x for x in plist])
            
                
            #print(np.sum(plist))
            if np.sum(plist) == 1:
                vote_index = np.where(vote_weights==np.random.choice(vote_weights, p=plist))[0]
                vote_index = np.where(vote_index==np.random.choice(vote_index))[0] if len(vote_index) > 1 else vote_index
                vote_index = np.asscalar(vote_index) if not np.isscalar(vote_index) else vote_index
                action_index = actions[vote_index]
            action = np.array(np.unravel_index(self.Valid_Inputs[action_index],(2,2,2,2,2,2)))

                       
            
            for i in range(5):
                state_new, temp_reward, done, info = env.step(action)
                score = info['total_reward']
                reward += temp_reward
                if done:
                    #self._train()
                    reward = 0
                    break
            state_new = state_new.flatten()
            self.data.append((state_then,state_new,action_index,reward,done))
            self.rewards.append(reward)
            state_then = state_new
            
            self.dist = float(info['distance']) if info['distance'] > 0 else self.dist
            self.max_dist = self.dist if self.dist > self.max_dist else self.max_dist
            self.epsi = self.epsi + 0.01 if stuck else self.epsi
            #self.epsi = (self.dist/self.max_dist)
            #print(self.epsi)
        self.history.append((self.episode,score))
        return(score,steps)
        
        
    def _train(self):
        #print("Training...")
        '''
        for i in range(len(self.rewards)):
            for ii in range(len(self.rewards)-i):
                self.rewards[i] += self.gamma ** ii * self.rewards[i+ii]
        '''
        for im in range(num_models):
            data_len = len(self.data)
            data_indexes = np.random.choice(data_len,self.mb_size)
            mb_data = [self.data[i] for i in data_indexes]
            inputs = np.zeros((data_len,) + (209,))
            targets = np.zeros((data_len,20))
            
            for i in range(min([self.mb_size,data_len])):
                state_then,state_new,action,reward,done = mb_data[i]
                state_now = np.hstack((state_new,[action])).flatten().reshape((1,209))
                inputs[i,:] = state_now
                targets[i,:] = np.round(models[im].predict(state_now),10)[0]
                if not done:
                    
                    for ii in range(min([self.mb_size-i,i+10])):
                        reward += self.gamma ** ii * mb_data[i+ii][3]
                        #targets[i] += self.gamma ** ii * self.rewards[i+ii]
                    
                    #reward += self.gamma * self.data[i+1][3]
                        
                #print targets[i][action] - reward
                targets[i,action] = reward
            
            models[im].train_on_batch(inputs,targets)
        #print("Training Done")
        
        
    def evaluate(self,state_then):
            
        score, steps = self._loop(state_then)
        
        return (score,steps)
        
        
    def solve(self):
            
        score, steps = self._loop()
            
        if score < self.max_score:
            print("Failed... Score: ",score," in ",steps," Steps")
            return 0
            
        return int(score > self.max_score)
        
        
        
    def get_Q(self):
        return self.Q



o = GymSuperMario(max_steps=10000,max_score=3200)

style.use('fivethirtyeight')
plt.ion()
plt.figure()
plt.plot([],[])
plt.show(block=False)
def update_plot():
    episodes = []
    scores = []
    for episode,score in o.history:
        episodes.append(episode)
        scores.append(score)
    z = np.polyfit(episodes, scores, 1)
    p = np.poly1d(z)
    plt.cla()
    plt.plot(episodes, p(episodes), 'b--',episodes,scores,'g-')
    #plt.gca().lines[0].set_xdata(episodes)
    #plt.gca().lines[0].set_ydata(scores)
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.pause(0.05)


completed = False
models[:] = _load_model(models,_huber_loss) if o.mod_load else models[:]
o.epsi = o.epsi_min if o.mod_load else o.epsi
state_then = env.reset()
while not completed:
    score,step = o.evaluate(state_then)
    update_plot()
    #print o.history
    print('Practice - score: ',score,' Steps: ',step)
    if o.mod_save:
        _save_model()
    if score >= 500:
        print('Winning!')
        for i in range(num_models):
            models[i].save('./DQMario_model'+str(i)+'_winner.h5')
        env.change_level()
    else:
        o._train()
        o.epsi *= o.epsi_decay
score,step = o.evaluate(state_then)
print('Test - score: ',score,' Steps: ',step)
    
    