#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#前面是有很多个四维的memory
self.memory_size = 100000
self.memory_counter = 128
def store_transition(self, state, action, reward, next_state):
        
        #做一个列表：把四个元素放进一个列表里面
        transition = [np.array(state),action,reward,np.array(next_state)]
        self.memory.append(transition)
        
        #总memory大小固定为100000，超过100000之后要实现先进先出的功能
        if self.memory_counter > self.memory_size:
            index = self.memory_counter % self.memory_size  #类似hashmap赋值思想
            self.memory[index, :] = transition  #进行替换
        
        self.memory_counter += 1


# In[ ]:


def learn(self):
    
    
    
        #1.当learn了一定次数时，把eval_net赋值给target_net
        if self.learn_step_counter % self.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        
        

        #2.从memory中抽出batch_size（128个）的memory
        if self.memory_counter > self.memory_size:   #说明记忆库已经存满，可以从记忆库任意位置收取
            sample_index = np.random.choice(self.memory_size, size=self.BATCH_SIZE)
        else:   #记忆库未满，只能从已经储存了的记忆中进行选择
            sample_index = np.random.choice(self.memory_counter, size=self.BATCH_SIZE)
        
        batch = []
        for sampleindex in sample_index:
            batch.append(self.memory[sampleindex])
        #原来代码：batch = self.memory.sample(self.BATCH_SIZE)
        #我不用原来那个memory的类了，那个类有点问题
        
        
        
        
        #3.将所有的memory中的s，a，r，s_单独抽出来，成为四个列表
        batch_state=np.array([o[0] for o in batch])

        batch_next_state= np.array([o[3] for o in batch])

        batch_action=np.array([o[1] for o in batch])

        batch_reward=np.array([o[2] for o in batch])

        
        #4.对原来的重要改动！！！把错的格式转换改成正确的格式转换
        #batch_action要行向量转化为列向量，不然就索引不了
        #其他三个其实没有发生变化，就没有改其他三个
        batch_action = torch.LongTensor(np.reshape(batch_action, (-1, 1))).detach().to(self.device)
        #源代码：batch_action = torch.LongTensor(np.reshape(batch_action, (-1, len(batch_action)))).detach().to(self.device)
        batch_reward =  torch.FloatTensor(np.reshape(batch_reward, (-1,len(batch_reward)))).to(self.device)
        batch_state=torch.FloatTensor(np.reshape(batch_state, (-1, 3, self.n, self.O_max_len))).to(self.device)
        batch_next_state =torch.FloatTensor(np.reshape(batch_next_state, (-1, 3, self.n, self.O_max_len))).to(self.device)

        
        
        
        #5.先把s输入到evalnet中，得到6个工件的值
        
        #action作为索引，来取出evalnet中s对应之前选择a的值，即q（s，a）*128个
        
        q_eval = self.eval_net(batch_state).gather(1, batch_action).to(self.device)
        
        #将next_state输入到target_net中，得到q（s_），（128个*6个qvalue）

        q_next = self.target_net(batch_next_state).detach().to(self.device)
        

        
        
        
        #6.将qnext中的已经选过的动作令为0，原因比较复杂，在组会上解释
        for memory_index in range(len(batch_next_state)):
            
            for job_index in range(len(q_next[memory_index])): #对于每一个memory中的next_state中的第一个矩阵，哪一行为0代表已经被选过
                
                if sum(batch_next_state[memory_index][0][job_index]) == 0:

                    q_next[memory_index][job_index] = 0
        

        
        
        
        
        #7.把对q（s_）取最大值，乘上gamma，再加上batch_reward
        #即MAX(q（s_）)*gamma + batch_reward
        q_target = batch_reward + self.GAMMA * q_next.max(1)[0]
        

            
        #8.loss_i = MAX(q（s_）)*gamma + batch_reward   -   q（s，a）
        #128个memory的差的平方再加和，即为loss
        #这时候要把q_eval从列向量转化回行向量,不然会触发广播机制，即[1,128]- [128,1]
        q_eval= q_eval.squeeze().to(self.device)
        q_target= q_target.squeeze().to(self.device)
        loss = self.loss_func(q_eval, q_target).to(self.device)

        
            
            
            
            
        #9.清空前面算的梯度，再算梯度，然后用RMS更新参数
        l=loss.detach().cpu().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return l

