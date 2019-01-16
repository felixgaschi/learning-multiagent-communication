import numpy as np
import utils

class LeverEnv:
    
    def __init__(self, N, J):
        
        self.J = J
        self.N = N
        
    def reset(self):
        
        state = np.random.choice(self.N, size = self.J, replace = False)
        
        terminal_state = False
        
        return state, terminal_state
    
    def get_reward(self, one_hot_action_seq):        
        
        reward = np.sum(np.sum(one_hot_action_seq, axis = 0) > 0) /self.J
        
        return reward
        
    def step(self, state, action):
        
        next_state = np.random.choice(self.N, size = self.J, replace = False)
        
        one_hot_action_seq = np.zeros((self.J, self.J))
        one_hot_action_seq[range(self.J), action] = 1
        reward = self.get_reward(one_hot_action_seq)
        
        terminal_state = False
        
        return next_state, reward, terminal_state

# correspondance between direction and increment of position (order: NSEW)
move_dict = {
    1: np.array([-1, 0]),
    2: np.array([ 1, 0]),
    3: np.array([ 0, 1]),
    4: np.array([ 0,-1])
}

class PredatorPreyTask():
    """
    Task for predator-prey task. Predators are looking for a prey inside a grid, at each step, they
    can go North, South, East or West so as the prey. Their state is two integers, one between 0 and 
    (grid_size * grid_size - 1) representing their position (x + y * grid_size) and and another between 
    0 and (detection_range * detection_range) equal to 0 if it does not detect the prey and 
    (1 + x' + y' * detection_range) representing the position of the prey inside its detection grid
    of size detection_range. 
    - N: number of agent
    - grid_size: size of the grid (a square of side grid_size)
    - detection_range: half of the size of the detection grid
    - communication_cost: additional cost induced by a single broadcast
    - step_cost: cost induced by any step
    - forbidden_cost: cost induced by forbidden action
    - avoid_closest: if True, the prey avoids the closest agent instead of going random
    - return_absolute: if True, the state of an agent contains the absolute position of the prey (only if
    it can see it) instead of its relative position
    """

    def __init__(self, N=5, grid_size=20, detection_range=2, communication_cost=0.01, step_cost=0.03, avoid_closest=True, forbidden_cost=10., return_absolute=True,
                prey_detection_range=2, uncatched_cost=1., T=50):
        self.N = N
        self.grid_size = grid_size
        self.detection_range = detection_range
        self.communication_cost = communication_cost
        self.step_cost = step_cost
        self.avoid_closest = avoid_closest
        self.return_absolute = return_absolute
        self.forbidden_cost = forbidden_cost
        self.prey_detection_range = prey_detection_range
        self.uncatched_cost = uncatched_cost
        self.T = T

        self.max_pos_index = 1 + grid_size * grid_size
        self.max_det_index = 1 + detection_range * detection_range
    
    def vision(self, pred_coord, prey_coord):
        """
        returns integer representing what the predator sees
        """
        x, y = pred_coord
        xprim, yprim = prey_coord
        a, b = xprim - x + self.detection_range, yprim - y + self.detection_range
        if self.return_absolute:
            if a <= 2 * self.detection_range and b <= 2 * self.detection_range and a >= 0 and b >= 0:
                return utils.encode_pos(xprim, yprim, self.grid_size)
            else:
                return 0
        else:
            return utils.encode_pos(a, b, 2 * self.detection_range)
    
    def is_terminated(self):
        """
        returns true if predators at position pred_positions can see
        the prey at prosition prey_pos
        """
        for p in self.pred_coord:
            if p == self.prey_coord:
                return True
        return False

    def reset(self):
        positions = np.random.choice(range(1, self.max_pos_index), replace=False, size = self.N)
        prey_pos = np.random.randint(1, self.max_pos_index)
        self.pred_coord = [utils.decode_pos(p, self.grid_size) for p in positions]
        self.prey_coord = utils.decode_pos(prey_pos, self.grid_size)
        state = [positions, [self.vision(p, self.prey_coord) for p in self.pred_coord]]
        terminal_state = self.is_terminated()
        self.t = 0
        return state, terminal_state
    
    def step(self, move_action, comm_action):
        self.t += 1
        pred_coord = self.pred_coord
        prey_coord = self.prey_coord

        # move prey
        if self.avoid_closest:
            distances = [np.abs(p[1] - prey_coord[1]) + np.abs(p[0] - prey_coord[0]) for p in pred_coord]
            i = np.argmin(distances)
            a, b = pred_coord[i][0] - prey_coord[0], pred_coord[i][1] - prey_coord[1]
            if np.abs(a) + np.abs(b) > self.prey_detection_range:
                choice = 0
            elif b > a:
                if b > -a:
                    choice = 4
                else:
                    choice = 2
            else:
                if b > -a:
                    choice = 1
                else:
                    choice = 3
        else:
            choice = np.random.choice([1,2,3,4])
        new_prey_coord, _ = utils.move_pos(prey_coord, choice, self.grid_size)
        self.prey_coord = new_prey_coord

        # move predators
        infos = [utils.move_pos(pred_coord[i], move_action[i], self.grid_size) for i in range(self.N)]
        new_coords = [elt[0] for elt in infos]
        is_blocked = [1 if elt[1] else 0 for elt in infos]
        self.pred_coord = new_coords

        # compute terminal_state
        terminal_state = self.is_terminated()
        
        # update reward 
        reward = [ 
            - self.communication_cost * comm_action[i] 
            - self.step_cost 
            - self.forbidden_cost * is_blocked[i]
            - (self.uncatched_cost if not terminal_state and self.t >= self.T else 0)
            for i in range(self.N) 
        ]

        # compute next state
        next_state = [
            [utils.encode_pos(p[0], p[1], self.grid_size) for p in self.pred_coord], 
            [self.vision(p, self.prey_coord) for p in self.pred_coord]
        ]
        
        return next_state, reward, terminal_state

    def render(self):
        pred_coord = self.pred_coord
        prey_coord = self.prey_coord
        print("\n".join([
            "".join([
                "X" if (i,j) in pred_coord and (i,j) == prey_coord
                else (
                    "O" if (i, j) == prey_coord
                    else (
                        "1" if (i, j) in pred_coord else "."
                    )
                )
                for j in range(self.grid_size)
            ])
            for i in range(self.grid_size)
        ]))
