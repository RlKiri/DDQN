#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import copy
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np
from Agent import Agent,random_choice
from PFSPCopy1 import PFSP_Env,Gantt
from IPython.display import clear_output
import matplotlib.pyplot as plt


# In[2]:


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

train_dataset = PFSPDataset(6,6,2000)
train_loader = DataLoader(train_dataset, batch_size=2000,            shuffle=True,pin_memory=True)

#生成10个时间矩阵用于评价
test_time_matrix = random_choice(train_loader,2000,10)


# In[3]:


def TrainModel(job_num, machine_num,Agent,batch_size,episode,train_loader):
    #train_loader中有2000个6*6的时间矩阵
    
    Loss_list=[]
    rewards_list = []#记录每10个episode，evaluate的时候的平均reward
    C_max_list = [] #记录每10个episode，evaluate的时候的平均cmax
    
    for i1 in range(episode): #episodes = 10000
        print('episode',i1)
        
        
        
        #1.从包含2048个矩阵中的data中随机选择1个，然后重置环境
        #反正都是来贡献memory的，随机选多少个影响不大
        #输入：包含2048个矩阵的dataloader
        #输出：1个时间矩阵组成的列表
        
        #时间矩阵
        PT = random_choice(train_loader,2000,1)[0]
        #顺序矩阵
        MT = []
        for i in range(job_num):
            MT_i = [_ for _ in range(machine_num)]
            MT.append(copy.copy(MT_i))
        env = PFSP_Env(job_num, machine_num, PT, MT)
        state,done = env.reset()
        ep_reward = 0

        
        
            
        #2.选工件
        action_chosen=[]   #记录已经选的工件
        while True:
            action_value = Agent.choose_action(state)
            if np.random.randn() > 0.1:   #10%的随机选择
                
                #先令选过的动作的价值为负无穷
                for i2 in action_chosen:   
                    action_value[0][i2] = -np.inf
                
                #再选最大值的动作
                action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
                
            else: 
                action = np.random.randint(0,6)
                while action in action_chosen:
                    action = np.random.randint(0,6)

            action_chosen.append(action)
            
            
            
            #3.选出工件之后开始与环境互动：
            #PFSP选择一次相当于JSP连续选择六次，得到下一个s和reward
            state0 = []
            reward0 = []
            done_or_not0 = []

            for i3 in range(6):
                s0, r0, done0=env.step(action)
                state0.append(s0)
                reward0.append(r0)
                done_or_not0.append(done0)


            reward = sum(reward0)
            next_state = state0[5]
            done = done_or_not0[5]
            
            #4.互动完之后，存储进memory
            Agent.store_transition(state, action, reward, next_state)
            ep_reward += reward
            

        
            
        
        
        
            #5.当一个时间矩阵排完之后，在跳出while，前往下一个时间矩阵之前，
            #要learn（每一步learn一次），并记录loss
            
            if Agent.memory_counter >= batch_size:
                loss=Agent.learn()
                Loss_list.append(loss)
                
            #5.done时跳出循环
            if done:
                action_chosen=[]
                
                
                
                #6.每10个epoch画一次图
                #用agent跑test_set到最后，看看reward和loss
                if done and i1%10==0:
                    test_C_max,test_rewards = evaluate(test_time_matrix,job_num, machine_num,Agent)
                    C_max_list.append(test_C_max)
                    rewards_list.append(test_rewards)
                    plot(Loss_list,C_max_list,rewards_list,i1)
                break

            state = copy.copy(next_state)
    return rewards,Cs,Ls


# In[4]:


#评价函数：与上面的trainmodel函数一样
def evaluate(test_time_matrix,job_num, machine_num,Agent):

    test_rewards_list = []#记录每10个episode，agent在10个固定矩阵上的reward
    test_C_max_list = [] #记录每10个episode（时间矩阵）的cmax
    
    
    for i4 in range(len(test_time_matrix)): 
        PT = test_time_matrix[i4]
        #顺序矩阵
        MT = []
        for i in range(job_num):
            MT_i = [_ for _ in range(machine_num)]
            MT.append(copy.copy(MT_i))
        env = PFSP_Env(job_num, machine_num, PT, MT)
        state,done = env.reset()
        ep_reward = 0


        action_chosen=[]   #记录已经选的工件
        while True:
            action_value = Agent.choose_action(state)
            if np.random.randn() > 0.1:   #10%的随机选择
                
                #先令选过的动作的价值为负无穷
                for i5 in action_chosen:   
                    action_value[0][i5] = -np.inf
                
                #再选最大值的动作
                action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
                
            else: 
                action = np.random.randint(0,6)
                while action in action_chosen:
                    action = np.random.randint(0,6)

            action_chosen.append(action)
            
            
            
            #3.选出工件之后开始与环境互动：
            #PFSP选择一次相当于JSP连续选择六次，得到下一个s和reward
            state0 = []
            reward0 = []
            done_or_not0 = []

            for i6 in range(6):
                s0, r0, done0=env.step(action)
                state0.append(s0)
                reward0.append(r0)
                done_or_not0.append(done0)


            reward = sum(reward0)
            next_state = state0[5]
            done = done_or_not0[5]
                
            ep_reward += reward
            

            #5.done时跳出循环
            if done:
                action_chosen=[]
                test_C_max_list.append(env.C_max())
                test_rewards_list.append(ep_reward)
                break
                
    test_C_max = sum(test_C_max_list)/len(test_time_matrix)
    test_rewards = sum(test_rewards_list)/len(test_time_matrix)
    return test_C_max,test_rewards


# In[5]:


#画图函数
def plot(Loss_list,C_max_list,rewards_list,episode):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('epoch %s C_max ' % (episode))
    plt.plot(C_max_list)
    plt.grid()
    plt.subplot(132)
    plt.title('epoch %s reward' % (episode))
    plt.plot(rewards_list)
    plt.grid()
    plt.subplot(133)
    plt.title('learn_count %s loss' % (episode))
    plt.plot(Loss_list)
    plt.grid()
    plt.show()


# In[6]:


dueling,DOUBLE,PER=0,0,0
Gamma = 0.9
net_update = 100
learning_rate = 0.00001
Memory_size = 10000
batch_size = 128
episode = 10000
job_num = 6
machine_num = 6
dueling,DOUBLE,PER=0,0,0


# In[ ]:


agent = Agent(job_num, machine_num, Gamma, net_update,learning_rate              ,Memory_size,batch_size,dueling,DOUBLE,PER)
rewards,Cs,Ls =TrainModel(job_num, machine_num,agent,                          batch_size,episode,train_loader)


# In[ ]:





# In[ ]:




