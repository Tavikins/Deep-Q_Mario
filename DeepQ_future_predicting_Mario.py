#Deep-Q-Learning with Super Mario Gym (OpenAI)

import numpy as np
import gym


env = gym.make('meta-SuperMarioBros-Tiles-v0')
from random import random, randint
from keras import models as Models
from keras.models import Sequential, load_model
from keras.layers import Dense, LocallyConnected2D, Flatten, TimeDistributed,Dropout
from keras.optimizers import Nadam
from keras import backend as K
from keras.constraints import min_max_norm


def _huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
num_models = 2
models = [0] * num_models
for i in range(num_models):
    models[i] = Sequential()
    models[i].add(Dense(1024, activation='relu',input_dim= 209))
    models[i].add(Dense(512, activation='relu'))
    
models[0].add(Dense(20, activation='linear'))
models[0].compile(loss='mse', optimizer=Nadam(lr=0.0001))
models[1].add(Dense(208, activation='linear'))
models[1].compile(loss='mse',optimizer=Nadam(lr= 0.0001))

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
            del models[i]
        except:
            print('Model not loaded, using default')
            m[i] = models[i]
    return m
    
class GymSuperMario(object):
    global debug
    def __init__(self, max_steps = 8000, max_score = 2000):
        self.mod_save = True
        self.mod_load = False
        self.max_steps = max_steps
        self.max_score = max_score
        self.actions = []
        self.states = []
        self.rewards = []
        self.data = []
        self.max_dist = 40.0
        self.dist = 40.0
        self.Q = None
        self.gamma = 0.8
        self.mb_size = 200
        self.epsi = 0.9
        self.epsi_decay = 0.95
        self.epsi_min = 0.05
        self.Valid_Inputs = [i for i in range(64) if (ui(i)[0] + ui(i)[1] + ui(i)[2] + ui(i)[3]) <= 1]

        
    def _loop(self,state_then):
        steps = 1
        self.dist = 40
        done = False
        reward = 0
        score = 0
        self.data = []
        state_then = state_then.flatten()
        state_now = np.array([],dtype='int')
        state_new = state_then
        self.db = state_new
        stuck = False
        action_index = 0
        future_predictions = 1
        while not done:
            reward = 0
            steps += 3
            future_reward = 0
            state_now = np.array([state_new],dtype='int')[0]
            state_next = np.round(models[1].predict(np.append(state_now.flatten(),action_index).reshape((1,209)))[0],0)
            for i in range(future_predictions):
                Q = models[0].predict(np.append(state_next.flatten(),action_index).reshape((1,209)))[0]
                #print 'future Q = ', Q
                plist = Q/np.sum(Q)
                action_index = None
                try:
                    action_index = np.where(Q==np.random.choice(Q,p=plist))[0]
                    print(action_index)
                    action_index = np.where(action_index==np.random.choice(action_index))[0] if len(action_index) > 1 else action_index
                    #action_index = np.asscalar(action_index) if not np.isscalar(action_index) else action_index
                except:
                    print('error - action_index failed')
                    action_index = np.argmax(Q)
                
                if random() < self.epsi:
                    action_index = randint(0,19)
                future_reward += Q[action_index]
                state_next = np.round(models[1].predict(np.append(state_next.flatten(),action_index).reshape((1,209)))[0],0)
                #actions[i] = action_index
                #vote_weights[i] = float(Q[action_index])
                #print vote_weights
            #print 'action_index = ', action_index
            #print 'future_reward = ', future_reward
            #print 'state_next = ', state_next
            #print actions
            #print vote_weights
            #plist = vote_weights/np.sum(vote_weights)
            #print(np.sum(plist))
            #if np.sum(plist) == 1:
                #vote_index = np.where(vote_weights==np.random.choice(vote_weights, p=plist))[0]
                #vote_index = np.where(vote_index==np.random.choice(vote_index))[0] if len(vote_index) > 1 else vote_index
                #vote_index = np.asscalar(vote_index) if not np.isscalar(vote_index) else vote_index
                #action_index = actions[vote_index]
            action = np.array(np.unravel_index(self.Valid_Inputs[action_index],(2,2,2,2,2,2)))
            #print 'action = ', action
                       
            
            for i in range(5):
                state_new, temp_reward, done, info = env.step(action)
                score = info['total_reward']
                reward += temp_reward
                if done:
                    #self._train()
                    break
            reward *= 10
            #print 'reward = ', reward
            print 'error = ', abs(Q[action_index] - reward)
            state_new = state_new.reshape((1,208))
            self.data.append((state_then,state_new,action_index,reward,future_reward,done))
            state_then = state_new
            
            self.dist = float(info['distance']) if info['distance'] > 0 else self.dist
            self.max_dist = self.dist if self.dist > self.max_dist else self.max_dist
            self.epsi = self.epsi + 0.01 if stuck else self.epsi
            #self.epsi = (self.dist/self.max_dist)
            #print(self.epsi)
            print 'epsi = ', self.epsi
        
        return(score,steps)
        
        
    def _train(self):
        #print("Training...")
        if True:
            data_len = len(self.data)
            #data_indexes = np.random.choice(data_len,self.mb_size)
            mb_data = [self.data[i] for i in range(data_len)]
            #inputs = np.zeros((data_len,) + (13,16))
            #targets = np.zeros((data_len,20))
            targets = np.zeros((data_len,1,20))
            state_targets = np.zeros((data_len,1,208))
            state_inputs = np.zeros((data_len,1,209))
            for i in range(data_len):
                state_then,state_new,action,reward,future_reward,done = mb_data[i]
                #state_now = np.array([state_new],dtype='int')
                state_inputs[i,:,:] = np.append(state_then,action).reshape((1,1,209))
                state_targets[i,:,:] = state_new.reshape((1,1,208))
                #Q = models[0].predict(np.expand_dims(np.append(state_then,action).reshape((1,209)),axis=0))[0]
                #print 'train Q = ', Q
                #targets[0,:] = Q
                #print 'targets = ', targets
                #if data_len - i > 6:
                    #print 'i = ', i
                    #print 'data_len =', data_len
                    #if not True in np.array(mb_data)[i:min(i+5,data_len),5]:
                for ii in range(data_len-i):
                    reward += self.gamma ** ii * mb_data[i+ii][3]
                #else:
                    #reward = -10
                if done:
                    reward -= 10
                targets[i,0,action] = reward * 10
                #print 'state_inputs = ', state_inputs
                #print 'targets = ', targets
                #print 'state_targets = ', state_targets
                models[1].fit(state_inputs[i],state_targets[i],epochs=1,verbose=0)
                models[0].fit(state_inputs[i],targets[i],epochs=1,verbose=0)
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
completed = False
models[:] = _load_model(models,_huber_loss) if o.mod_load else models[:]
o.epsi = o.epsi_min if o.mod_load else o.epsi
state_then = env.reset()
while not completed:
    score,step = o.evaluate(state_then)
    print('Practice - score: ',score,' Steps: ',step)
    if o.mod_save:
        _save_model()
    if score >= 500:
        print('Winning!')
        for i in range(num_models):
            models[i].save('./DQMario_model'+str(i)+'_winner.h5')
        #env.change_level()
    else:
        o._train()
        o.epsi *= o.epsi_decay
score,step = o.evaluate(state_then)
print('Test - score: ',score,' Steps: ',step)
    
    