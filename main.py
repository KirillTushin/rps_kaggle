import os
import sys
from glob import glob
from sklearn.tree import DecisionTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from agents import *
from wrapper import *

RANDOM = True

sys.path.append("/kaggle_simulations/agent")
rpscontest_paths = glob(f'/kaggle_simulations/agent/rpscontest_bots/*.py')

base_agents = []

for look_at in ['opponent', 'my']:
    for feature in ['opponent', 'my', 'both']:
        agent_name = f'Markov__look_at={look_at}__feature={feature}'
        base_agents.append(
            Markov(
                agent_name=agent_name,
                look_at=look_at,
                feature=feature,
            )
        )

    for path in rpscontest_paths:
        agent_name = f'RPS__rpsbot={os.path.basename(path)}__look_at={look_at}'
        base_agents.append(
            RPSContestBot(
                agent_name=agent_name,
                look_at=look_at,
                path=path,
            )
        )
    agent_name = f'sklearn__look_at={look_at}'
    base_agents.append(
        SklearnAgent(
            agent_name=agent_name,
            look_at=look_at,
            estimators=[
                DecisionTreeClassifier(random_state=42),
            ],
        )
    )
    agent_name = f'Iocaine_base__look_at={look_at}'
    base_agents.append(
        Iocaine(
            agent_name=agent_name,
            look_at=look_at,
        )
    )

base_agents.append(
    PiBot(agent_name='pibot')
)


Meta_Maa = MultiArmedAgent(
    agent_name='meta_maa',
    base_agents=base_agents,
    window=200,
    best_n_agents_range=(10,),
    deterministic_range=(False,),
    lag_range=(0,),
    norm_range=('softmax',),
    shift_range=(0,),
)


wrapper = Wrapper(
    agent=Meta_Maa,
    use_agent_type='meta_maa__best_n_agents_type:10__deterministic_type:False__lag_type:0__norm_type'
                   ':softmax__shift_type:0',
    play_random_if_more=300 if RANDOM else 999,
    play_random_if_less=-20 if RANDOM else -999,
    play_random_proba=0.25 if RANDOM else 0,
    first_random_steps=100 if RANDOM else 0,
    start_random_after=400 if RANDOM else 999,
    verbose=True,
)


def agent(observation, configuration):
    global wrapper

    step = observation.step
    opponent_action = None
    if step:
        opponent_action = observation.lastOpponentAction

    wrapper.step(opponent_action)

    if observation.step == configuration.episodeSteps - 2:
        print(wrapper.agent.top_n_agents)

    return wrapper.my_action
