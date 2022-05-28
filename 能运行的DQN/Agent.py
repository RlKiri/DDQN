import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from RL_network import CNN_FNN,CNN_dueling


class Agent():
    """docstring for DQN"""
    def __init__(self,n,O_max_len,Gamma,net_update,learning_rate,M_size,B_size,dueling=False,double=False,PER=False,model_path=None):
        print('Agent')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.es=8000*n*O_max_len
        self.record=0
        self.double=double
        self.PER=PER
        self.GAMMA=Gamma
        self.n=n
        self.O_max_len=O_max_len
        
        
        
        
        super(Agent, self).__init__()
        if dueling:
            self.eval_net, self.target_net = CNN_dueling(self.n,self.O_max_len).to(self.device), CNN_dueling(self.n,self.O_max_len).to(self.device)
        else:
            self.eval_net, self.target_net = CNN_FNN(self.n, self.O_max_len).to(self.device), CNN_FNN(self.n, self.O_max_len).to(self.device)
        self.Q_NETWORK_ITERATION=net_update
        self.BATCH_SIZE=B_size
        self.learn_step_counter = 0
        self.memory_counter = 0
        
        
        
        
        
        
        #memory从一个类变成一个列表
        #if PER:
        #    self.memory = preMemory(M_size)
        #else:
        #    self.memory = Memory(M_size)
        self.memory = []
        
        
        
        
        
        
        
        
        self.Min_EPISILO=0.005
        self.EPISILO=1
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        if model_path!=None:
            if os.path.exists(model_path + '/eval_net.pkl'):
                self.eval_net.load_state_dict(torch.load(model_path + '/eval_net.pkl'))
                self.target_net.load_state_dict(torch.load(model_path + '/target_net.pkl'))

        self.memory_size = 100000  # 记忆上限        
                
                

                


    def choose_action(self, state):
        self.record+=1
        state=np.reshape(state,(-1,3,self.n,self.O_max_len))
        state=torch.FloatTensor(state).to(self.device)
        # print(state.size())
        # state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array

        action_value = self.eval_net.forward(state)

        
        #改动：直接返回action——value，这样就没有了随机性的变化了，是固定随机性的
        return action_value

    def PER_error(self,state, action, reward, next_state):

        state = torch.FloatTensor(np.reshape(state, (-1, 3, self.n, self.O_max_len))).to(self.device)
        next_state= torch.FloatTensor(np.reshape(next_state, (-1, 3, self.n, self.O_max_len))).to(self.device)
        p=self.eval_net.forward(state)
        p_=self.eval_net.forward(next_state)
        p_target=self.target_net(state)

        if self.double:
            q_a=p_.argmax(dim=1)
            q_a=torch.reshape(q_a,(-1,len(q_a)))
            qt=reward+self.GAMMA*p_target.gather(1,q_a)
        else:
            qt=reward+self.GAMMA*p_target.max()
        qt=qt.detach().cpu().numpy()
        p=p.detach().cpu().numpy()
        errors=np.abs(p[0][action]-qt)
        return errors

    def store_transition(self, state, action, reward, next_state):
        # print(reward)
        

            
        transition = [np.array(state),action,reward,np.array(next_state)]
        self.memory.append(transition)
        
        #总memory大小固定为100000，超过100000之后要实现先进先出的功能
        if self.memory_counter > self.memory_size:
            index = self.memory_counter % self.memory_size  #类似hashmap赋值思想
            self.memory[index, :] = transition  #进行替换
        
        self.memory_counter += 1
        
        
        
        
        #源代码：
        #if self.PER:
            #errors=self.PER_error(state, action, reward, next_state)
            #self.memory.remember((state, action, reward, next_state), errors)
            #self.memory_counter += 1
        #else:
            #self.memory.remember((state, action, reward, next_state))
            #self.memory_counter+=1

    def learn(self):
    
    
    
        #当learn了一定次数时，把eval_net赋值给target_net
        if self.learn_step_counter % self.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        
        

        #从memory中抽出batch_size（128个）的memory
        #并将所有的memory中的s，a，r，s_单独抽出来，成为一个列表

        if self.memory_counter > self.memory_size:   #说明记忆库已经存满，可以从记忆库任意位置收取
            sample_index = np.random.choice(self.memory_size, size=self.BATCH_SIZE)
        else:   #记忆库还没有存满，从现有的存储记忆提取
            sample_index = np.random.choice(self.memory_counter, size=self.BATCH_SIZE)
        
        batch = []
        for sampleindex in sample_index:
            batch.append(self.memory[sampleindex])
            
            
            
        #原来代码：batch = self.memory.sample(self.BATCH_SIZE)

        batch_state=np.array([o[0] for o in batch])
        #print('batch_state',batch_state)
        batch_next_state= np.array([o[3] for o in batch])
        #print('batch_next_state',batch_next_state)
        batch_action=np.array([o[1] for o in batch])
        #print('batch_action',batch_action)
        batch_reward=np.array([o[2] for o in batch])
        #print('batch_reward',batch_reward)

        
        #对原来的重要改动！！！把错的格式转换改成正确的格式转换
        #batch_action要行向量转化为列向量，其他三个其实没有发生变化
        batch_action = torch.LongTensor(np.reshape(batch_action, (-1, 1))).detach().to(self.device)
        #batch_action = torch.LongTensor(np.reshape(batch_action, (-1, len(batch_action)))).detach().to(self.device)
        batch_reward =  torch.FloatTensor(np.reshape(batch_reward, (-1,len(batch_reward)))).to(self.device)
        batch_state=torch.FloatTensor(np.reshape(batch_state, (-1, 3, self.n, self.O_max_len))).to(self.device)
        batch_next_state =torch.FloatTensor(np.reshape(batch_next_state, (-1, 3, self.n, self.O_max_len))).to(self.device)

        
        
        
        #先把s输入到evalnet中，得到6个工件的值
        #action作为索引，来取出evalnet中s对应之前选择a的值，即q（s，a）
        q_eval = self.eval_net(batch_state).gather(1, batch_action).to(self.device)
        
        #将next_state输入到target_net中，得到q（s_）
        #先把对q（s_）取最大值，乘上gamma，再加上batch_reward
        q_next = self.target_net(batch_next_state).detach().to(self.device)
        

        #print('batch_next_state',batch_next_state)

        #print('q_nexthouqian',q_next)
        #将qnext中的已经选过的动作令为0

        for memory_index in range(len(batch_next_state)):
            
            for job_index in range(len(q_next[memory_index])): #对于每一个memory中的next_state中的第二个矩阵，哪一行为0代表已经被选过
                
                if sum(batch_next_state[memory_index][0][job_index]) == 0:
                    #print('memory_index',memory_index)
                    #print('job_index',job_index)
                    q_next[memory_index][job_index] = 0
        
        #print('q_nexthou',q_next)
                
        
        
        
        
        
        q_target = batch_reward + self.GAMMA * q_next.max(1)[0]
        

            
        #MAX(q（s_）)*gamma + batch_reward   -   q（s，a）
        #128个memory的差的平方再加和，即为loss
        #这时候要把q_eval从列向量转化回行向量
        q_eval= q_eval.squeeze().to(self.device)
        q_target= q_target.squeeze().to(self.device)
        loss = self.loss_func(q_eval, q_target).to(self.device)

        
            
            
            
            
            
        l=loss.detach().cpu().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return l


    def save_model(self,file):
        if not os.path.exists(file):
            os.makedirs(file)
        torch.save(self.eval_net.state_dict(), file +'/' +'eval_net.pkl')
        torch.save(self.target_net.state_dict(), file +'/'+'target_net.pkl')
        
def random_choice(train_loader,batch_size,sample_number):
    episode_data = []
    for batch_id, sample_batch in enumerate(train_loader):
#data里面只有1个二维向量，0和2048个矩阵,但只有这样才能调用到里面的数据
#从其中随机选择128个矩阵

        choice = random.sample(range(0,batch_size),sample_number)
    for x in choice:

        episode_data.append(sample_batch[x])
     
    return episode_data