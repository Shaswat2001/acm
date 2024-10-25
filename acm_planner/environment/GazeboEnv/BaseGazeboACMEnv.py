import numpy as np
import gymnasium as gym
from gym import spaces

class BaseGazeboACMEnv:
   
   def __init__(self) -> None:

      self.high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
      self.time_steps = 3
      self.no_coordinates = 2
      action_max = np.array([0.04]*3*self.time_steps + [0.5]*3*self.time_steps)
      action_min = np.array([-0.04]*3*self.time_steps + [0.1]*3*self.time_steps)
      
      self.action_space = spaces.Box(low=action_min, high=action_max, dtype=np.float32)
      self.observation_space = spaces.Box(low=-self.high, high=self.high, dtype=np.float32)
         
   def step(self, action):

      self.current_time += 1
      trajectory = action[0]
      
      self.pose += trajectory[0:3]
      self.pose_error = self.get_error()
      reward,done = self.get_reward()

      if done:
         print(f"The position error at the end : {self.pose_error}")
         print(f"The end pose of UAV is : {self.pose}")

      pose_diff = self.q_des - self.pose
      pose_diff = np.clip(pose_diff,-self.high,self.high)
      prp_state = pose_diff
      prp_state = prp_state.reshape(1,-1)

      return prp_state, reward, done, False, {}
   
   def choose_best(self,trajectories):

      best_length = np.inf
      best_trajectory = None

      for trajectory in trajectories:
         
         length = np.linalg.norm(trajectory[0:3] - trajectory[3:6]) + np.linalg.norm(trajectory[3:6] - trajectory[6:9]) + 2*np.linalg.norm(trajectory[6:9] - self.q_des)

         if length < best_length:
            best_length = length
            best_trajectory = trajectory[0:3]

      return best_trajectory
   
   def get_reward(self):
        
      done = False
      pose_error = self.pose_error

      if pose_error < 0.1:
         done = True
         reward = 10
      else:
         reward = -(pose_error*10)

      if self.current_time > self.max_time:
         done = True
         reward = -1

      return reward, done
   
   def get_error(self):

      pose_error =  np.linalg.norm(self.pose - self.q_des) 

      return pose_error
      
   def reset(self, pose: np.ndarray = np.array([0.0,0.0,2.0]), pose_des = None, max_time: int = 10) -> np.ndarray:

      self.pose = pose      
      if pose_des is None:
         self.q_des = np.random.randint([-1.0,-1.0,1.0],[2.0,2.0,4.0])
      else:
         self.q_des = pose_des

      print(f"The target pose is : {self.q_des}")
      pose_diff = self.q_des - self.pose
      pose_diff = np.clip(pose_diff,-self.high,self.high)
      prp_state = pose_diff
      prp_state = prp_state.reshape(1,-1)

      self.current_time = 0
      self.max_time = max_time
      
      return prp_state
