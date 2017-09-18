#Deep-Q-Learning with Super Mario Gym (OpenAI)

import numpy as np
import gym


env = gym.make('meta-SuperMarioBros-Tiles-v0')
from random import random, randint

from keras.models import Sequential, load_model
from keras.layers import Dense, LocallyConnected2D, Flatten, TimeDistributed,Dropout
from keras.optimizers import Adam
from keras import backend as K


def _huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
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
         
debug = []
def ui(i):
    return np.unravel_index(i,(2,2,2,2,2,2))
def _save_model():
    print('Saving Model')
    model.save('./DQMario_model.h5')

def _load_model(model, f):
    print('Loading Model')
    m=None
    try:
        m = load_model('./DQMario_model.h5',custom_objects={'_huber_loss':_huber_loss})
        del model
    except:
        print('Model not loaded, using default')
        m = model
    return m
    
class GymSuperMario(object):
    global debug
    def __init__(self, max_steps = 8000, max_score = 2000):
        self.mod_save = True
        self.mod_load = True
        self.max_steps = max_steps
        self.max_score = max_score
        self.actions = []
        self.states = []
        self.rewards = []
        self.data = []
        self.max_dist = 40.0
        self.dist = 40.0
        self.Q = None
        self.gamma = 0.3
        self.mb_size = 200
        self.epsi = 1.0
        self.epsi_decay = 0.7
        self.epsi_min = 0.01
        self.Valid_Inputs = [i for i in range(64) if (ui(i)[0] + ui(i)[1] + ui(i)[2] + ui(i)[3]) <= 1]

        
    def _loop(self,state_then):
        steps = 1
        self.dist = 40
        done = False
        reward = 0
        score = 0
        self.data = []
        state_now = np.array([],dtype='int')
        state_new = state_then
        self.db = state_new
        stuck = False
        while not done:
            reward = 0
            steps += 3
            state_now = np.array([state_new],dtype='int')
            #state_now = state_now.reshape((1,1,416))
            self.Q = np.round(model.predict(state_now),10)[0]
            
            plist = self.Q/np.sum(self.Q)
            try:
                action_index = np.where(self.Q==np.random.choice(self.Q,p=plist))
                action_index = action_index
                print(action_index)
                action_index = np.where(self.Q[action_index]==np.random.choice(action_index)) if len(action_index) > 1 else action_index
                action_index = np.asscalar(action_index[0]) if not np.isscalar(action_index[0]) else action_index
            except:
                print('error - action_index failed')
                action_index = np.argmax(self.Q)
            
            if random() < self.epsi:
                action_index = randint(0,19)
            action = np.array(np.unravel_index(self.Valid_Inputs[action_index],(2,2,2,2,2,2)))
                       
            
            for i in range(3):
                state_new, temp_reward, done, info = env.step(action)
                score += temp_reward
                reward += temp_reward
                if done:
                    #self._train()
                    break
            
            self.data.append((state_then,state_new,action_index,reward,done))
            state_then = state_new
            
            self.dist = float(info['distance']) if info['distance'] > 0 else self.dist
            self.max_dist = self.dist if self.dist > self.max_dist else self.max_dist
            self.epsi = self.epsi + 0.01 if stuck else self.epsi
            #self.epsi = (self.dist/self.max_dist)
            #print(self.epsi)
            
        print('Score:',score)
        return(score,steps)
        
        
    def _train(self):
        #print("Training...")
        data_len = len(self.data)
        data_indexes = np.random.choice(data_len,self.mb_size)
        mb_data = [self.data[i] for i in range(data_len)]
        inputs = np.zeros((data_len,) + (13,16))
        targets = np.zeros((data_len,20))
        
        for i in range(data_len):
            state_then,state_new,action,reward,done = mb_data[i]
            state_now = np.array([state_new],dtype='int')
            inputs[i,:,:] = state_now
            targets[i,:] = np.round(model.predict(state_now),10)[0]
            if not done:
                for ii in range(len(self.rewards)-i):
                    reward += self.gamma ** ii * self.rewards[i+ii]
            targets[i,action] = reward
        

        model.fit(inputs,targets,batch_size=self.mb_size,epochs=1,shuffle=False)
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
model = _load_model(model,_huber_loss) if o.mod_load else model
o.epsi = o.epsi_min if o.mod_load else o.epsi
state_then = env.reset()
while not completed:
    score,step = o.evaluate(state_then)
    print('Practice - score: ',score,' Steps: ',step)
    if o.mod_save:
        _save_model()
    if score >= 500:
        print('Winning!')
        model.save('./DQMario_model_winner.h5')
        #env.change_level()
    else:
        o._train()
        o.epsi *= o.epsi_decay
score,step = o.evaluate(state_then)
print('Test - score: ',score,' Steps: ',step)
    
    