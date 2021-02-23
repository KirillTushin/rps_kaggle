from copy import deepcopy
from kaggle_environments.envs.rps.utils import get_score

from utils import *


class Wrapper:
    def __init__(
            self,
            agent,
            use_agent_type,
            play_random_if_more=220,
            play_random_if_less=-30,
            play_random_proba=0.25,
            first_random_steps=150,
            start_random_after=500,
            verbose=True,
    ):
        self.agent = deepcopy(agent)
        self.use_agent_type = use_agent_type

        self.play_random_if_more = play_random_if_more
        self.play_random_if_less = play_random_if_less
        self.play_random_proba = play_random_proba
        self.first_random_steps = first_random_steps
        self.start_random_after = start_random_after
        self.verbose = verbose

        self.steps = 0
        self.reward = 0
        self.my_action = None
        self.history = {
            'my': [],
            'opponent': [],
            'both': [],
        }

    def update(self, opponent_action):
        self.steps += 1
        if opponent_action is not None:
            self.reward += get_score(self.my_action, opponent_action)

            self.history['my'].append(self.my_action)
            self.history['opponent'].append(opponent_action)
            self.history['both'].append(both2index[f'{self.my_action}{opponent_action}'])
        self.agent.generate_moves(self.history)

    def is_random_step(self):
        is_start = self.steps < self.first_random_steps
        is_enough = self.steps > self.start_random_after
        is_win = self.reward > self.play_random_if_more
        is_lose = self.reward < self.play_random_if_less
        random_step = np.random.rand() < self.play_random_proba
        random_step = is_win or is_lose or random_step
        random_step = is_enough and random_step
        play_random = is_start or random_step
        return play_random

    def step(self, opponent_action):
        self.update(opponent_action)
        self.my_action = int(self.agent.moves[self.use_agent_type][-1])

        if self.is_random_step():
            self.my_action = np.random.randint(3)
            if self.verbose:
                print(f'Random action at step = {self.steps}, reward = {self.reward}')

        return self.my_action
