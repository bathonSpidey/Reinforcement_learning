#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:02:27 2019

@author: spidey
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use('ggplot')

size=10
hm_episodes=25000
move_penalty=1
enemy_penalty=300
food_reward=25

epsilon=.9
eps_decay=.9998
show_every=1000

start_q_table=None
learning_rate=.1
discount=.95

player_n=1
food_n=2
enemy_n=3
enemy2_n=4
d={1:(255,175,0), 2:(0,255,0),3:(0,0,255),4:(0,0,255)}

class Blob:
    def __init__(self):
        self.x=np.random.randint(0,size)
        self.y=np.random.randint(0,size)
        
    def __str__(self):
        return "{},{}".format(self.x,self.y)
    
    def __sub__(self,other):
        return (self.x-other.x,self.y-other.y)
    
    def action(self, choice):
        if choice==0:
            self.move(x=1,y=1)
        elif choice==1:
            self.move(x=-1,y=-1)
        elif choice==2:
            self.move(x=-1,y=1)
        elif choice==3:
            self.move(x=1,y=-1)
        
    
    def move(self,x=False,y=False):
        if not x:
            self.x +=np.random.randint(-1,2)
        else:
            self.x +=x
        if not y:
            self.y +=np.random.randint(-1,2)
        else:
            self.y +=y
        if self.x<0:
            self.x=0
        elif self.x>size-1:
            self.x=size-1
        if self.y<0:
            self.y=0
        elif self.y>size-1:
            self.y=size-1
            
if start_q_table is None:
    q_table= {}
    x3=0
    y3=0
    for x1 in range(-size+1,size):
        for y1 in range(-size+1,size):
            for x2 in range(-size+1,size):
                for y2 in range(-size+1,size):
                    q_table[((x1,y1),(x2,y2),(x3,y3))]=[np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q_table,'rb') as f:
        q_table=pickle.load(f)
episode_rewards=[]
for episode in range(hm_episodes):
    player= Blob()
    food=Blob()
    enemy=Blob()
    enemy2=Blob()
    
    if episode %show_every==0:
        print('on {}, epsilon: {}'.format(episode,epsilon))
        print('{} ep mean {}'.format(show_every,np.mean(episode_rewards[-show_every:])))
        show=True
    else:
        show=False
    
    episode_reward=0
    for i in range(200):
        obs=(player-food,((player-enemy)+(player-enemy2)))
        if np.random.random()>epsilon:
            action=np.argmax(q_table[obs])
        else:
            action=np.random.randint(0,4)
        
        player.action(action)
        
        #enemy.move()
        #food.move()
        if player.x==enemy.x and player.y==enemy.y or player.x==enemy2.x and player.y==enemy2.y:
            reward=-enemy_penalty
        elif player.x==food.x and player.y==food.y:
            reward=food_reward
        else:
            reward=-move_penalty
        
        new_obs=(player-food, (player-enemy)+(player-enemy2))
        max_future_q=np.max(q_table[new_obs])
        current_q=q_table[obs][action]
        
        if reward==food_reward:
            new_q=food_reward
        elif reward==-enemy_penalty:
            new_q=-enemy_penalty
        else:
            new_q=(1-learning_rate)*current_q+learning_rate*(reward+discount*max_future_q)
        q_table[obs][action]=new_q
        
        if show:
            env=np.zeros((size,size,3),dtype=np.uint8)
            env[food.y][food.x]=d[food_n]
            env[player.y][player.x]=d[player_n]
            env[enemy.y][enemy.x]=d[enemy_n]
            env[enemy2.y][enemy2.x]=d[enemy2_n]
            
            img=Image.fromarray(env,'RGB')
            img=img.resize((300,300))
            cv2.imshow('',np.array(img))
            if reward==food_reward or reward==-enemy_penalty:
               if cv2.waitKey(500) & 0xFF==ord('q'):
                   break
            else:
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    break
        episode_reward+=reward
        if reward==food_reward or reward==-enemy_penalty:
            break
    episode_rewards.append(episode_reward)
    epsilon *=eps_decay
    
moving_avg=np.convolve(episode_rewards,np.ones((show_every,))/show_every, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel('reward {}ma'.format(show_every))
plt.xlabel('episode')
plt.show()

with open('qtable-{}.pickle'.format(int(time.time())),'wb') as f:
    pickle.dump(q_table,f)
        
            