#Super Mario Gym (OpenAI)

import numpy as np
import gym


env = gym.make('meta-SuperMarioBros-Tiles-v0')
from random import random, randint, sample

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K


def _huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
model = Sequential()
model.add(Dense(416, activation='relu',input_shape=(1,416)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dropout(.002))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(.002))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(20,  activation='relu'))
model.compile(loss=_huber_loss, optimizer=Adam(lr=0.00001))
              
debug = []
def ui(i):
    return np.unravel_index(i,(2,2,2,2,2,2))

class GymSuperMario(object):
    global debug
    def __init__(self, max_steps = 8000, max_score = 2000):
        debug.append(self)
        self.max_steps = max_steps
        self.max_score = max_score
        self.actions = []
        self.states = []
        self.rewards = []
        self.data = []
        self.max_dist = 40
        self.dist = 40
        self.Q = None
        self.gamma = 0.8
        self.mb_size = 200
        self.epsi = 1
        self.epsi_decay = 0.99
        self.epsi_min = .005
        self.Valid_Inputs = [i for i in range(64) if (ui(i)[0] + ui(i)[1] + ui(i)[2] + ui(i)[3]) <= 1]

        
    def _loop(self,state_then):
        steps = 0
        self.max_dist = 40
        self.dist = 40
        done = False
        reward = 0
        score = 0
        state_new = state_then
        while steps < self.max_steps and not done:
            temp_reward = 0
            steps += 3
            state_now = np.vstack((state_then.reshape((1,1,208)),state_new.reshape((1,1,208))))
            state_now = state_now.reshape((1,1,416))
            self.Q = np.round(model.predict(state_now)[0][0],10)
            plist = self.Q/np.sum(self.Q)
            
            try:
                action_index = np.where(self.Q==np.random.choice(self.Q,p=plist))
                action_index = action_index[0][0]
            except:
                print('error - action_index failed')
                action_index = np.argmax(self.Q)
                
            if random() < self.epsi:
                action_index = randint(0,19)
            action = np.array(np.unravel_index(self.Valid_Inputs[action_index],(2,2,2,2,2,2)))
                       
            
            for i in range(3):
                state_new, reward, done, info = env.step(action)
                score += reward
                temp_reward += reward
                
            reward = temp_reward/(steps/10) if not done else -20
            
            self.data.append((state_then,state_new,action_index,reward,done))
            state_then = state_new
            
            self.dist = info['distance']
            if self.dist > self.max_dist:
                self.max_dist = self.dist
            self.epsi = (self.dist/self.max_dist)**(-1)
            
        print('Score:',score)
        return(score,steps)
        
        
    def _train(self):
        #print("Training...")
        data_indexes = np.random.choice(len(self.data),self.mb_size)
        mb_data = [self.data[i] for i in data_indexes]
        inputs = np.zeros((self.mb_size,) + (1,416))
        targets = np.zeros((self.mb_size,) + (1,20))
        
        for i in range(self.mb_size):
            state_then,state_new,action,reward,done = mb_data[i]
            state_now = np.vstack((state_then.reshape((1,1,208)),state_new.reshape((1,1,208)))).reshape((1,1,416))
            inputs[i,:,:] = state_now
            targets[i,:,:] = np.round(model.predict(state_now)[0][0],10)
            if not done:
                for ii in range(len(self.rewards)-i):
                    reward += self.gamma ** ii * self.rewards[i+ii]
            targets[i,0,action] = reward
            
        model.fit(inputs,targets,batch_size=self.mb_size,epochs=5,shuffle=False)
        #print("Training Done")
        
        
    def evaluate(self,state_then):
            
        score, steps = self._loop(state_then)
        
        return (score,steps)
        
        
    def solve(self):
            
        score, steps = self._loop()
            
        if score < self.max_score:
            print("Failed... Score: ",score/self.max_score," in ",steps," Steps")
            return 0
            
        return int(score > self.max_score)
        
        
    def get_Q(self):
        return self.Q



o = GymSuperMario(max_steps=10000,max_score=3200)
completed = False
state_then = env.reset()
while not completed:
    score,step = o.evaluate(state_then)
    print('Practice - score: ',score,' Steps: ',step)
    if score >= 3200:
        completed = True
        o.epsi = 0
    else:
        o._train()
score,step = o.evaluate(state_then)
print('Test - score: ',score,' Steps: ',step)
    
    