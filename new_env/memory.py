import random
import numpy as np 

class Memory:
    def __init__(self, memory_size):
        self._memory_size = memory_size
        self._samples = []
    
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._memory_size:
            self._samples.pop(0)
    
    def get_samples(self, batch_size):
        '''If batch size > samples` size, return samples` size, otherwise return batch size.
        '''
        if batch_size > len(self._samples):
            batch_samples = random.sample(self._samples, len(self._samples))
        else:
            batch_samples = random.sample(self._samples, batch_size)
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        for batch_sample in batch_samples:
            batch_state.append(batch_sample[0])
            batch_action.append([batch_sample[1]])
            batch_reward.append([batch_sample[2]])
            batch_next_state.append(batch_sample[3])
        return np.asarray(batch_state), np.asarray(batch_action), np.asarray(batch_reward), np.asarray(batch_next_state)

    def get_length(self):
        return len(self._samples)
