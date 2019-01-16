from keras.layers import Input
import numpy as np
    
class CommNet(object):

    def __init__(self, env, agent_network, J, K,
                decoder,
                input_size,
                skip_conn=False,
                temporal=False,
                connexions={}):

        inputs = [Input(input_shape=((input_size,))) for i in range(J)]
        h = []
        c = []
        for j in range(J):
            h.append([inputs[j]])
            c.append([np.zeros((input_shape,))])
        for k in range(0, K):
            for j in range(J):
                h[j].append(agent_network([h[j][-1], c[j][-1]]))
            for j in range(J):
                c[j].append(np.zeros((input_size)))
                for i in connexions[j]:
                    c[j][k] += h[i][k]
                if len(connexions[j]) > 0:
                    c[j][k] /= len(connexions[j])
        
        outputs = [decoder(h[j][-1]) for j in range(J)]

        self.model = Model(inputs, outputs)
        self.env = env
        
        # TODO train method, add skip connection, temporal and baseline

    def get_baseline(self):
        


class LeverTask(object):

    def __init__(self, m, n):
        self.m = m
        self.n = n

    def get_reward(states, actions):
