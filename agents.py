from sklearn.model_selection import ParameterGrid
from collections import defaultdict, OrderedDict
from copy import deepcopy
from kaggle_environments.envs.rps.utils import get_score

from utils import *


class Agent:
    def __init__(
            self,
            agent_name,
            look_at='opponent',
            norm_range=('norm_sum', 'softmax'),
            shift_range=(0, 1, 2),
            deterministic_range=(True, False),
            lag_range=(0, 1, 2, 3, 4),
    ):
        self.agent_name = agent_name
        self.look_at = look_at
        self.reverse_look_at = 'my' if self.look_at == 'opponent' else 'opponent'
        self.norm_range = norm_range
        self.shift_range = shift_range
        self.deterministic_range = deterministic_range
        self.lag_range = lag_range

        self.param_grid = self.generate_params_grid()
        self.name_space = self.generate_name_space()

        self.moves = {n: [] for n in self.name_space}
        self.last_moves = {n: None for n in self.name_space}
        self.rewards = {n: [] for n in self.name_space}

        self.weights = []
        self.actual_weights = None
        self.random_weights = generate_actual_weights([1, 1, 1])
        self.step = 0

    def generate_params_grid(self):
        return ParameterGrid(
            OrderedDict(
                norm_type=self.norm_range,
                deterministic_type=self.deterministic_range,
                shift_type=self.shift_range,
                lag_type=self.lag_range,
            )
        )

    def name_from_params(self, params):
        params_name = '__'.join([f'{key}:{value}' for key, value in sorted(params.items(), key=lambda x: x[0])])
        return f'{self.agent_name}__{params_name}'

    def generate_name_space(self):
        return [
            self.name_from_params(params)
            for params in self.param_grid
        ]

    def update_values(self, history):
        self.actual_weights = self.random_weights

    def get_action(self, norm_type, deterministic_type, shift_type, lag_type, agent_name):
        if 0 == lag_type:
            w = self.actual_weights
        elif len(self.weights) >= lag_type:
            action = self.moves[agent_name.replace(f"lag_type:{lag_type}", "lag_type:0")][-lag_type]
            return action
        else:
            w = self.random_weights

        probas = w[norm_type]
        action = np.argmax(probas) if deterministic_type else np.random.choice(3, p=probas)
        action = (action + shift_type) % 3
        return action

    def generate_range_moves(self):
        for params in self.param_grid:
            agent_name = self.name_from_params(params)
            action = self.get_action(**params, agent_name=agent_name)
            self.last_moves[agent_name] = action
            self.moves[agent_name].append(action)

    def generate_moves(self, history):
        self.update_values(history)
        self.weights.append(self.actual_weights)
        self.generate_range_moves()
        self.step += 1


class Markov(Agent):
    def __init__(
            self,
            feature,
            window_range=range(2, 26),
            *args,
            **kwargs,
    ):
        self.feature = feature
        self.window_range = window_range
        super().__init__(*args, **kwargs)

        self.max_window = max(window_range)
        self.key = ''
        self.table = defaultdict(lambda: np.array([0.0, 0.0, 0.0]))
        self.actual_weights = {
            i: generate_actual_weights([1, 1, 1])
            for i in window_range
        }

    def generate_params_grid(self):
        return ParameterGrid(
            OrderedDict(
                window_type=self.window_range,
                norm_type=self.norm_range,
                deterministic_type=self.deterministic_range,
                shift_type=self.shift_range,
                lag_type=self.lag_range,
            )
        )

    def get_action(self, window_type, norm_type, deterministic_type, shift_type, lag_type, agent_name):
        if 0 == lag_type:
            w = self.actual_weights[window_type]
        elif len(self.weights) >= lag_type:
            action = self.moves[agent_name.replace(f"lag_type:{lag_type}", "lag_type:0")][-lag_type]
            return action
        else:
            w = self.random_weights

        probas = w[norm_type]
        action = np.argmax(probas) if deterministic_type else np.random.choice(3, p=probas)
        action = (action + shift_type) % 3
        return action

    def __get_weights(self, key):
        use_range = range(1, len(key) + 1)
        weights_key = np.array(norm_sum(use_range))

        weights_table = np.array([self.table[key[-i:]] for i in use_range])
        proba = weights_key @ weights_table
        return proba

    def generate_weights(self):
        for i in self.window_range:
            key = self.key[-i:]
            if len(key):
                self.actual_weights[i] = generate_actual_weights(self.__get_weights(key))

    def update_values(self, history):
        if len(history[self.look_at]) > 0:
            len_key = len(self.key)
            use_range = range(1, min(len_key, self.max_window) + 1)

            target = history[self.look_at][-1]
            for i in use_range:
                self.table[self.key[-i:]][target] += 1
            self.key += str(history[self.feature][-1])

        self.generate_weights()


class RPSContestBot(Agent):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(path) as f:
            string_code = ''.join(f.readlines())
        self.code = compile(string_code, '<string>', 'exec')
        self.gg = dict()

    def update_values(self, history):
        if self.gg == dict():
            self.gg['input'] = ''
            self.gg['output'] = ''
        else:
            self.gg['input'] = 'RPS'[history[self.look_at][-1]]
            self.gg['output'] = 'RPS'[history[self.reverse_look_at][-1]]

        exec(self.code, self.gg)

        action = {'R': 0, 'P': 1, 'S': 2}[self.gg['output']]
        proba = proba_from_action(action)
        self.actual_weights = generate_actual_weights(proba)


class SklearnAgent(Agent):
    def __init__(self, estimators, k=5, min_samples=25, *args, **kwargs):
        self.estimators = estimators
        super().__init__(*args, **kwargs)
        self.k = k
        self.min_samples = min_samples

        self.rollouts_hist = {
            'steps': [],
            'actions': [],
            'opp-actions': [],
        }
        self.data = {
            'x': [],
            'y': [],
        }
        self.test_sample = None
        self.actual_weights = {
            est.__repr__(): generate_actual_weights([1, 1, 1])
            for est in self.estimators
        }

    def update_values(self, history):
        if len(history[self.look_at]) > 0:
            self.rollouts_hist = update_rollouts_hist(
                self.rollouts_hist,
                self.step,
                history[self.reverse_look_at][-1],
                history[self.look_at][-1],
            )

        if len(history[self.look_at]) <= self.min_samples + self.k:
            return

        if len(self.data['x']) == 0:
            self.data, self.test_sample = init_training_data(self.data, self.rollouts_hist, self.k)
        else:
            short_stat_rollouts = {key: self.rollouts_hist[key][-self.k:] for key in self.rollouts_hist}
            features = construct_features(short_stat_rollouts, self.rollouts_hist)
            self.data['x'].append(self.test_sample[0])
            self.data['y'] = self.rollouts_hist['opp-actions'][self.k:]
            self.test_sample = features.reshape(1, -1)

        for est in self.estimators:
            proba = predict_opponent_move(est, self.data, self.test_sample)
            self.actual_weights[est.__repr__()] = generate_actual_weights(proba)

    def generate_params_grid(self):
        return ParameterGrid(
            OrderedDict(
                estimator_type=self.estimators,
                norm_type=self.norm_range,
                deterministic_type=self.deterministic_range,
                shift_type=self.shift_range,
                lag_type=self.lag_range,
            )
        )

    def get_action(self, estimator_type, norm_type, deterministic_type, shift_type, lag_type, agent_name):
        if 0 == lag_type:
            w = self.actual_weights[estimator_type.__repr__()]
        elif len(self.weights) >= lag_type:
            action = self.moves[agent_name.replace(f"lag_type:{lag_type}", "lag_type:0")][-lag_type]
            return action
        else:
            w = self.random_weights

        probas = w[norm_type]
        action = np.argmax(probas) if deterministic_type else np.random.choice(3, p=probas)
        action = (action + shift_type) % 3
        return action


class Iocaine(Agent):
    def __init__(
            self,
            agent_name,
            ages=(1000, 100, 10, 5, 2, 1),
            *args,
            **kwargs,
    ):

        super().__init__(
            agent_name,
            *args,
            **kwargs,
        )

        self.predictors = []
        self.predict_history = self.predictor((len(ages), 2, 3))
        self.predict_frequency = self.predictor((len(ages), 2))
        self.predict_fixed = self.predictor()
        self.predict_random = self.predictor()
        self.predict_meta = [Predictor() for _ in range(len(ages))]
        self.stats = [Stats() for _ in range(2)]
        self.histories = [[], [], []]
        self.ages = ages

    def predictor(self, dims=None):
        if dims:
            return [self.predictor(dims[1:]) for _ in range(dims[0])]
        self.predictors.append(Predictor())
        return self.predictors[-1]

    def update_values(self, history):
        them = -1
        if len(history[self.look_at]) > 0:
            self.histories[0].append(history[self.reverse_look_at][-1])
            them = history[self.look_at][-1]
            self.histories[1].append(them)
            self.histories[2].append((self.histories[0][-1], them))
            for watch in range(2):
                self.stats[watch].add(self.histories[watch][-1], 1)

        rand = np.random.randint(3)
        self.predict_random.add_guess(them, rand)
        self.predict_fixed.add_guess(them, 0)

        for a, age in enumerate(self.ages):
            best = [recall(age, hist) for hist in self.histories]
            for mimic in range(2):
                for watch, when in enumerate(best):
                    if not when:
                        move = rand
                    else:
                        move = self.histories[mimic][when]
                    self.predict_history[a][mimic][watch].add_guess(them, move)
                most_freq, score = self.stats[mimic].max(age, rand, -1)
                self.predict_frequency[a][mimic].add_guess(them, most_freq)

        for meta, age in enumerate(self.ages):
            best = (-1, -1)
            for predictor in self.predictors:
                best = predictor.best_guess(age, best)
            self.predict_meta[meta].add_guess(them, best[0])

        best = (-1, -1)
        for meta in range(len(self.ages)):
            best = self.predict_meta[meta].best_guess(len(self.histories[0]), best)

        action = best[0]
        proba = proba_from_action(action)
        self.actual_weights = generate_actual_weights(proba)


class PiBot(Agent):
    def update_values(self, history):
        history_len = len(history[self.look_at])
        action = int(PI[history_len]) % 3
        proba = proba_from_action(action)
        self.actual_weights = generate_actual_weights(proba)


class MultiArmedAgent(Agent):
    def __init__(
            self,
            base_agents,
            window,
            best_n_agents_range=range(5, 26, 5),
            *args,
            **kwargs,
    ):
        self.best_n_agents_range = best_n_agents_range
        super().__init__(*args, **kwargs)

        self.base_agents = deepcopy(base_agents)
        self.base_agents_names = self.all_base_agents_names()

        self.window = window
        self.max_best_n_agents = max(best_n_agents_range)

        self.base_agents_last_moves = {n: None for n in self.base_agents_names}
        self.base_agents_rewards = {n: [] for n in self.base_agents_names}
        self.base_agents_scores = {n: 0 for n in self.base_agents_names}
        self.top_n_agents = None

        self.actual_weights = {
            i: generate_actual_weights([1, 1, 1])
            for i in best_n_agents_range
        }

    def generate_params_grid(self):
        return ParameterGrid(
            OrderedDict(
                best_n_agents_type=self.best_n_agents_range,
                norm_type=self.norm_range,
                deterministic_type=self.deterministic_range,
                shift_type=self.shift_range,
                lag_type=self.lag_range,
            )
        )

    def all_base_agents_names(self):
        names = []
        for agent in self.base_agents:
            names.extend(agent.name_space)
        return names

    def update_values(self, history):

        for base_agent in self.base_agents:

            if self.step > 1:
                for base_agent_name in base_agent.name_space:
                    base_agent_action = self.base_agents_last_moves[base_agent_name]
                    opponent_action = history[self.look_at][-1]
                    res = get_score(base_agent_action, opponent_action)
                    self.base_agents_rewards[base_agent_name].append(res)

            base_agent.generate_moves(history)

            for base_agent_name in base_agent.name_space:
                self.base_agents_last_moves[base_agent_name] = base_agent.last_moves[base_agent_name]

        self.base_agents_scores = {
            key: score_from_rewards(value[-self.window:])
            for key, value in self.base_agents_rewards.items()
        }
        self.generate_weights()

    def get_top_n(self):
        top_n = sorted(
            self.base_agents_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:self.max_best_n_agents]

        self.top_n_agents = dict(top_n)
        return zip(*top_n)

    def __get_weights(self, best_agent_names, best_agent_weights):
        best_agent_actions = [self.base_agents_last_moves[agent_name] for agent_name in best_agent_names]
        action = np.random.choice(best_agent_actions, p=softmax(best_agent_weights))
        proba = proba_from_action(action)
        return proba

    def generate_weights(self):
        best_agent_names, best_agent_weights = self.get_top_n()
        for top_n in self.best_n_agents_range:
            if self.step > 0:
                best_agent_names_n = best_agent_names[:top_n]
                best_agent_weights_n = best_agent_weights[:top_n]
                proba = self.__get_weights(best_agent_names_n, best_agent_weights_n)
                self.actual_weights[top_n] = generate_actual_weights(proba)

    def get_action(self, best_n_agents_type, norm_type, deterministic_type, shift_type, lag_type, agent_name):

        if 0 == lag_type:
            w = self.actual_weights[best_n_agents_type]

        elif len(self.weights) >= lag_type:
            action = self.moves[agent_name.replace(f"lag_type:{lag_type}", "lag_type:0")][-lag_type]
            return action
        else:
            w = self.random_weights

        probas = w[norm_type]
        action = np.argmax(probas) if deterministic_type else np.random.choice(3, p=probas)
        action = (action + shift_type) % 3
        return action
