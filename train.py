#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np


# In[ ]:


#先生成2048个时间矩阵
class PFSPDataset(Dataset):
    def __init__(self,num_machines,num_jobs,num_samples,random_seed=111):
        super(PFSPDataset,self).__init__()
        torch.manual_seed(random_seed)
        self.data_set=[]
        for l in tqdm(range(num_samples)):
            x=np.random.randint(1,100,size=(num_machines,num_jobs))
            x=torch.Tensor(x)
            self.data_set.append(x)
        self.size=len(self.data_set)
    def __len__(self):
        return self.size
        
    def __getitem__(self,idx):
        return self.data_set[idx]

train_dataset = PFSPDataset(2,6,2048)
train_loader = DataLoader(train_dataset, batch_size=2048,            shuffle=True,pin_memory=True)



def TrainModel(Agent,env,batch_size,episode,train_loader):
    
    
    Reward_total = []
    C_total = []
    rewards_list = []
    C = []
    L=[]
    
    for i1 in range(episode):
        
        
        
        #函数1：从包含2048个矩阵中的data中选择128个的函数
        #输入：包含2048个矩阵的dataloader
        #输出：128个时间矩阵组成的列表
        time_matrix128 = Agent.random_choice(train_loader,,2048,128)
        
        
        
        
        
        reward128 = []
        action_chosen128 = []
        next_state128 = []
        for i2 in time_matrix128:
            
            
            
            
            
            #需要加一个环境重置的条件
            state,done = env.reset()

            
            
            


        
            #开始对128个工件选顺序

            
            
            #action_value = Agent.choose_action(state)
            #函数2：choose_action函数
            #输入：state
            #输出：action的动作值
            
            #也就是eval_net的forward函数（也就是RLnetwork文件里面的CNN—FNN类）
            #输入：state
            #输出：各个action的动作值
            
            #需要加10%的随机选择这个功能吗 需要添加10% 可以让epsilon随着episode去改变
            
            action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
            action_chosen.append(action)
            
            
            #开始与环境互动：连续选择六次，得到下一个s和reward
            state0 = []
            reward0 = []
            done_or_not0 = []

            for i3 in range(6):
                s0, r0, done0=env.step(action)
                #Gantt(env.Machines)
                state0.append(s0)
                reward0.append(r0)
                done_or_not0.append(done0)


            reward = sum(reward0)
            next_state = state0[5]
            done = done_or_not0[5]
                
            #本来要存储memory的，现在不用了
            #感觉还是要的
            Agent.store_transition(state, action, reward, next_state)
            #ep_reward += reward
            
            reward128.append(reward)
            action_chosen128.append(action)
            next_state128.append(next_state)
        
        
        
        
        
        #当128个action，reward，next_state出来之后，到函数3
        #函数3：learn函数（其中有PER_error函数）
        #注：原来的learn函数是从memory中sample出来state,action，reward，next_state的
        #改为
        #输入：128个state,action，reward，next_state
        #输出：各个action的动作值
        
        loss=Agent.learn()
        
        
        
        
        
        
        #if i1 % 10 == 0：
            #ret, f, C1, R1 = evaluate(i,Agent,env,loss)
            #函数4：evaluate函数
            #对模型进行评价
            #没有讲到怎么evaluate我来提供一个思路
            #再随机生成一批矩阵，用agent跑到最后，看看reward和loss


            
            
            
            
            
        state = copy.copy(next_state)


# In[ ]:




