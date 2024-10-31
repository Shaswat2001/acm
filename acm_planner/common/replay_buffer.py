import numpy as np

class ReplayBuffer:

    def __init__(self,input_shape,mem_size,n_actions,batch_size=64,save_health=True):
        self.mem_size = mem_size
        self.current_mem = 0
        self.state = np.zeros(shape=(mem_size,input_shape))
        self.action = np.zeros(shape=(mem_size,n_actions))
        self.reward = np.zeros(shape=(mem_size,1))
        self.next_state = np.zeros(shape=(mem_size,input_shape))
        self.done = np.zeros(shape=(mem_size,1))
        self.health_reward = np.zeros(shape=(mem_size,1))
        self.batch_size = batch_size
        self.save_health = save_health

    def store(self,state,action,reward,next_state,done):
        index = self.current_mem%self.mem_size
        self.state[index] = state
        self.action[index] = action
        
        self.next_state[index] = next_state
        self.done[index] = done

        if self.save_health:
            self.reward[index] = reward[0]
            self.health_reward[index] = reward[1]
        else:
            self.reward[index] = reward
        self.current_mem+=1

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)

        if self.save_health:
            return (self.state[index],self.action[index],self.reward[index],self.next_state[index],self.done[index],self.health_reward[index])
        
        return (self.state[index],self.action[index],self.reward[index],self.next_state[index],self.done[index])