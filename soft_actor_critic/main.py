import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='Implementation of SAC')
#Different enviroments can be cofigured to it like HalfCheetah-v2| 0.2|
#| Hopper-v2  Walker2d-v2 Ant-v2  Humanoid-v2 #/
parser.add_argument('--env-name', default="Reacher-v2",
                    help='Default is "Reacher-v2"')
#So we can have deterministic or Stochastic picking up of actions
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#evaluates the policy after every 10 episodes in reacher v2 kept each episode of 10 steps
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
#it is the discount factor or importance given to future step rewards
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
#tau – (float) the soft update coefficient (“polyak update”, between 0 and 1)
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient (default: 0.005)')
#learning rate depends on the step size
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
#determines the realtive importance of entropy of reward , depends on the number of visits
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter determines the relative importance of the entropy term against the reward (default: 0.2)')
#it is just to automatically checks for entropy tunning i am not implementing it here as it is very sensitive to changes and needs expertise or more time
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust  (default: False)')
#I am generating this random seed to maintain the same random generations of actions of and states in multiple episodes
parser.add_argument('--seed', type=int, default=100000, metavar='N',
                    help='random seed (default: 100000)')
# keeping the batch size of 256 for getting to train from reply buffer
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
#number of the steps before stop training goal or failure
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000001)')
# the number of hidden layers as implemented by the original paper 256
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
# the steps after which update is given
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
#intial number of steps before we start taking actions from Actor
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
#value target update after every step
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
#the max size of reply buffer by default kept as 10000000
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
## I put default True because I have RTX 1650 GPU
parser.add_argument('--cuda', action="store_true",  default=True,
                    help='run on CUDA (default: True)')
#packed all these intial conditions in parser.
args = parser.parse_args()

# This is accepting the enivorment which by default i have Reacher-V2
env = gym.make(args.env_name)
#Seed is used to keep the actions/states same to decrease the randomness
#Not necessary step but can decrease the training time
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Declarint the agent and passing it enviroment of reacher-v2
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow
#can be used for visualization
#'numbers/2021-02-20_20-43-50_SAC_Reacher-v2_Gaussian_\\events.out.tfevents.1613850230.LAPTOP-NT413I90.6888.0' #example output
writer = SummaryWriter('numbers/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Intializing memory object
memory = ReplayMemory(args.replay_size, args.seed)

# From here Training starts
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
#while will be true till env.step(action) timeout or have goal
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # to sample the random actions by randomly
        else:
            action = agent.select_action(state)  # sample the actions by Gussian untill given deterministic

        if len(memory) > args.batch_size:
            # The update given as the size increases then batch
            for i in range(args.updates_per_step):
                # updating the parameter for all networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha , value_loss= agent.update_parameters(memory, args.batch_size, updates)
                # updating the event writer to save the scaler
                writer.add_scalar('loss of critic_1', critic_1_loss, updates)
                writer.add_scalar('loss of critic_2', critic_2_loss, updates)
                writer.add_scalar('loss of policy', policy_loss, updates)
                writer.add_scalar('loss of entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy of temprature alpha', alpha, updates)
                writer.add_scalar('loss of value', value_loss, updates)
                updates += 1
#added to the tensorboard
        next_state, reward, done, _ = env.step(action)
# Step
        if done:
            print('done')
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

# Ignore the "done" signal if it comes from hitting the time horizon.

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
# Append transition to memory in replay buffer
        memory.push(state, action, reward, next_state, mask)


        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
#evaluating after every 10 episodes
    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

#adding again the more comments to tensorboard

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
agent.save_model( args.env_name, suffix="", actor_path=None, critic_path=None,value_path=None)
env.close()

