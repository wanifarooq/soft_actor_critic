# This is the implementaion of the "SAC paper"

# Reinforcement Learning

# Implementation Project

# Algorithm: Soft Actor-Critic

# Environment: Reacher-v

## Table of Contents

- Introduction ------------- page
- Algorithm ------------- page 4-
- Environment ------------- page 9-
- Implementation -------------- page 12-
- Conclusions ------------- page
- References ------------- page


**Introduction**

In this project, we are going to implement the Soft Actor-Critic algorithm in the

“Reacher-v2” environment.


**Algorithm: Soft Actor-Critic**

**Actor-Critic Methods**

The Actor-Critic method is a category of policy gradient methods aiming to

reduce the variance during training [31]. Its main idea is to maintain two

models, the policy 𝜋𝜃(𝑎|𝑠) and the action-value function (or the state-value

function) Qw(s, a), which are the actor and the critic respectively. The critic,

Qw(s, a), is used to estimate the action-value function under the policy 𝜋𝜃.

The actor, 𝜋𝜃(𝑎|𝑠), is updated in the direction suggested by the critic, Qw(s, a).

Equation

In addition to the "vanilla" actor-critic method, the advantage function is often

used to even further reduce the variance during training. That is, Qw(s, a)

in the equation is replaced by the advantage function

Aw(s, a) = Qw(s, a) − Vw(s)

**Soft Actor-Critic**

In this method here are used function approximators for both the Q-function and the
policy, and instead of running evaluation and improvement to convergence, alternate

between optimizing both networks with stochastic gradient descent. We will consider a
parameterized state value function Vψ(st), soft Q-function Qθ(st, at), and a tractable
policy πφ(at|st). The parameters of these networks are ψ, θ, and φ.


The value functions can be modeled as expressive neural networks and the policy as a

Gaussian with mean and covariance given by neural networks.

The updated rules will be derived next for these parameter vectors. The state value
function approximates the soft value. There is no need in principle to include a separate
function approximator for the state value since it is related to the Q-function and policy

according to the Equation below:

```
Equation 1
```
This quantity can be estimated from a single action sample from the current policy
without introducing a bias, but in practice, including a separate function approximator
for the soft value can stabilize training and is convenient to train simultaneously with
the other networks. The soft value function is trained to minimize the squared residual
error

```
Equation 2
```
Where **D** is the distribution of previously sampled states and actions, or a replay buffer.

The gradient of Equation 2 can be estimated with an unbiased estimator where the
actions are sampled according to the current policy, instead of the replay buffer. The
soft Q-function parameters can be trained to minimize the soft Bellman residual

With


which again can be optimized with stochastic gradients

The update makes use of a target value network V ̄ψ, where ̄ψ can be an exponentially
moving average of the value network weights, which has been shown to stabilize
training.

Alternatively, we can update the target weights to match the current value function
weights periodically.

Finally, the policy parameters can be learned by directly minimizing the expected KL-
divergence in Equation below

There are several options for minimizing Jπ. A typical solution for policy gradient
methods is to use the likelihood ratio gradient estimator which does not require
backpropagating the gradient through the policy and the target density networks.
However, in our case, the target density is the Q-function, which is represented by a
neural network an can be differentiated, and it is thus convenient to apply the
reparameterization trick instead, resulting in a lower variance estimator. To that end,
we reparameterize the policy using a neural network transformation

Where 𝜖𝑡 an input noise vector, sampled from some fixed distribution, such as a
spherical Gaussian.


We can now rewrite the objective in Equation below

as:

Where πφ is defined implicitly in terms of fφ, and we have noted that the partition
function is independent of φ and can thus be omitted.

We can approximate the gradient of Equation with:


Where at is evaluated at. This unbiased gradient estimator extends the DDPG
style policy gradients to any tractable stochastic policy.

This algorithm also makes use of two Q-functions to mitigate positive bias in the policy
improvement step that is known to degrade performance of value based. In particular,
in it there are parameterized two Q-functions, with parameters θi, and train them
independently to optimize JQ(θi).

Then the minimum of the Q-functions is use for the value gradient.

The algorithm can learn challenging tasks, using just a single Q-function, it was found
that two Q-functions significantly speedup training, especially on harder tasks.

The method alternates between collecting experience from the environment with the
current policy and updating the function approximators using the stochastic gradients
from batches sampled from a replay buffer.

Using off-policy data from a replay buffer is feasible because both value estimators and
the pol-icy can be trained entirely on off-policy data. The algorithm is agnostic to the
parameterization of the policy, as long as it can be evaluated for any arbitrary state-
action tuple.


**Environment : Reacher-v**

**OpenAI Gym and MuJoCo**

The environment used in this project is a MuJoCo environment, meaning
the dynamic models are built based on MuJoCo. OpenAI Gym provides

a shared interface, allowing the users to write and test deep reinforcement
learning algorithms without worrying about the connection with MuJoCo.

**OpenAI Gym**

OpenAI Gym is a toolkit aimed towards reinforcement learning research.

It contains a collection of benchmark problems, or environments, which are
commonly used in this domain. Its goal is to become a standardized simulated
environment and benchmark which can help researchers and practitioners
evaluate and compare the performance of RL algorithms based on the same

physics models. It also comprises an online community which allows the practitioners
to present their results and facilitates the discussions. In addition,
Its open-source nature allows its users to not only adapt the environments

to their specific needs but also create a brand-new environments or
customize the existing ones.

**MuJoCo**

MuJoCo is the acronym of Multi-Joint dynamics with Contacts. It is a physics
engine aiming to simulate model-based control tasks, and the areas it
facilitates range from robotics and biomechanics to graphics, animation and
any other areas that require fast and accurate simulation of complex dynamical
systems. MuJoCo has outperformed other existing physics engines when it

comes to computation speed and accuracy especially in robotics-related tasks.
MuJoCo is known for its user-friendly design and yet retain computational
efficiency.
The runtime simulation module, which implemented in C, is fine-tuned in order to

maximize the performance.
Unlike the other engines, MuJoCo is dedicated to providing better modelling
of contact dynamics instead of ignoring the contacts or using simple

spring-dampers to represent the contact mechanisms.


The contact models provided in MuJoCo include tangential, rotational resistance and

elliptic and pyramidal friction cones.
The elements in a MuJoCo model are body, joint, Degrees of freedom (DOF), geom, site,
constraint, tendon and acturator.
Bodies are elements to create dynamic components in the model. Joints are elements

to defined the motion of a particular body. In contrary to other physics
engines, such as ODE , the joint elements enable motion between the current
body and its parent instead of confining motion. Moreover, there are socalled
composite objects which are collections of existing model elements.

They are designed to simulate particle systems, rope, cloths, etc
One important feature provided by MuJoCo is the automatic inference
of the body inertial properties such as the masses and the moments of
inertia. The inertias of the bodies are derived automatically from the shape of

geoms and the material density.

In this work, we have used **Reacher-v2** environment.
It can be deemed as a robotic arm reaching task.

Reacher-v2 environment is one of the built-in environments provided by OpenAI
Gym. In this simulated environment, there is a square arena where a
random target and a 2 DOF robotic arm are located. As shown in Figure below,

the robotic arm consists of two linkages with equal length and two revolute
joints.


The **Reacher-v2** environment.

The target, which denotes by a red sphere, is randomly placed at the
beginning of each episode. For better visualization, the end-effector is pinpointed
by a sphere in light blue colour.

The goal of this learning environment is to make the end-effector touch the randomly-
placed target in each episode.
In this project we implement **SAC** (Soft Actor Critic) to enable the agent (the robotic
arm) to learn the optimal or suboptimal policy which is to decide the actions based on

the state at each time step to reach the target.
Initially, the only terminal state in the Reacher-v2 environment is the state
when the elapsed time reaches the limit of the specified time horizon.

To improve the sample efficiency, a minor change is made to the reward signal.
As shown in Algorithm 4, a tolerance of **δ** is specified to create a terminal state
when the end-effector approximately reaches the target.

The tolerance creates a circular area centred at the target with a radius of ). The end-
effector is not necessarily to stay within the area to be deemed as a success.
Specifically, the tolerance **δ** in the experiments is set to 0.0001.

Note that **Rcontrol** is a penalty term to prevent the reacher from obtaining a fast spinning
behavior. It is also reasonable to add this term concerning the energy cost in a real
world scenario.


Implementation

In the Soft actor critic algorithm we want to solve the problem of how to get robust and
stable learning in continuous action space environments.

The basic idea and Soft Actor Critic is that we want to use something called the
maximum entropy framework entropy just means disorder in this case so it's going to
add a parameter to the cost function or rather it's going to scale the cost function in
such a way that that it encourages exploration but it does so in a way that is robust to
random seeds for your environment as well as episode to episode variation and starting
conditions.

The Soft Actor Critic tends to be a little bit more smooth and has less of a problem with
the episode episode variation and that's due to the fact that it is maximizing not just the
total reward overtime but the stochasticity , the randomness, the entropy of how the
agent behaves as far as the neural networks we're going to need we're going to need a
network to handle the actor which is the policy and we're going to be handling how the
agents looks actions differently in this particular technique.

In SAC we are going to output a mean and standard deviation for a normal distribution
that we will then sample to get the actions for our agent will also need a critic network it
takes a state and action as input and tells to the actor actor whether the action it took
was really good or that action was terrible and then we're also going to have a value
network that says whether a state is valuable or not valuable and there will be an
interplay between the three to figure out for any given state we know what is the best
action what is the best sequence of states to access so we can know what actions to
take overtime.

We do need three networks and we were going we are going to define those.

We will start with the **critic network**

We will take a learning rate we will call it beta, a number of input dimensions from our
environment dimensions for the first the second fully connected layers and we are going
to default these two 256 because that's what they use in the paper.

The next thing we want to do is define our neural network and that will take input dims
of zero plus an action as input, the reason we have to do this is because the critic
evaluates the value of a state and action pair and so we want to incorporate the action
right from the very beginning of the input to the neural network, then our second fully


connected layer goes from FC-1 to FC-2 dimms and then we have an output which is a q
and that is a scalar quantity.

Now we need our optimizer what are we going to optimize the parameters of our deep
neural network , the learning rate of beta.

Next we have to deal with the feed. That will take a state and action as input and so we
will say that the action value is the feedforward of the concatenation of the state and
action along the batch dimension through our first fully connected layer. Then we can
pass that through the 2nd fully connected layer activated and then pass it through the
**x2** layer to get the final value for the state action.

Next we define our **value network**

Our initializer is going to take a few parameters we need a learning rate beta, input dims
FC-1 and FC-2 dimms.

The value function just estimates the value of a particular state or set of states it doesn't
care about what action you took or are taking.

Then we have our network , is a very simple neural network we're going to go from
input dims an output layer and that will be a scalar and we also need an optimizer this
network.

The feedforward function is pretty straightforward , we need a state value and then we
want to activate it and pass it through our second fully connected layer activate it again
we get our output and return it.

Then we want our **actor network**

This is the relatively hard part of the problem because we're going to have to handle
sampling a probability distribution instead of just doing a simple feedforward.

It takes learning rate Alpha, input dims we are going to default this to what they use in
the paper.

We also need reparameterization noise now the need for three parameterization noise
will be a little bit more apparent when we handle the calculation of the policy it's going
to be there to serve a number of functions first of all it's going to make sure that we


don't try to take the log of 0 which is undefined that will break the software so we need
a noise factor for that.

We define our deep neural network by 2 FC and then we have two outputs of μ which is
going to be the mean of the distribution for our policy and that will have as many
outputs as we have components to our actions and then a standard deviation.

We need an optimizer, next our feet forward that takes a state as input and then the
next step is to clamp our Sigma just as they did in the paper.

Now we can handle the actual agent itself

We will need an Alpha beta you know what let's do this Alpha we are just going to use
the default from the paper right away these are default 0.0003, gamma (discount
factor) of 0.99 , tau is the factor by which we are going to modulate the parameters of
our target value network so we're going to have a value network in a target value
network.

We are going to use **two critics** and we are going to take the minimum of the evaluation
of the state for those two networks in the calculation of our loss function for the value
and actor networks.

A value network is needed and we need our target value network as well as aninterface
function (replay buffer) between the agent and its memory.

After these steps:

Now we have to worry about the actual learning functionality of our agent and so the
first thing we want to do is see if we have filled up at least batch size of our memory and
if not we're not going to start learning we're just going to go back to the main loop of
the program and if we have filled up our backsides we are ready to go ahead and start
learning ,

next we have to calculate the values of the States and new states according to the value
and target value networks respectively.

What we also need to do we have to do is we have to get the actions and log
probabilities for the states according to the new policy and not for the actions that were
actually sampled from our buffer we use those later but in the calculation of the loss for
our value network and the actor network we want the value of the actions according to
the new policy.


Then we are going to need the Q values critic values are going to be is then going to be
the minimum of the two Q. Wdo this because you actually improve the stability of
learning because there is a problem of the overestimation bias.

Now we are ready to deal with the actor network loss and so we start by doing another
feedforward to get the actions and log probs.

Now we have to deal with the critic loss so this one is a little bit more straightforward.

This quantity called Q-hat ,scaling factor handles the inclusion of the entropy in our loss
function so that is what helps to encourage exploration.

To recap the fundamental idea behind Soft Actor Critic is that we are using a maximum
entropy reinforcement learning framework but the entropy comes from scaling the
reward factor this encourages exploration or encourages exploitation depending on that
scale factor as that scale factor grows then the signal to exploit grows as the reward
scale decreases then the tendency to explore increases so it is a way of fine tuning the
performance of the model.


**Conclusions**

After implementing the algorithm in our chosen environment we have observed
satisfying results considering the number of episodes for which we have trained it.

So the results are as follows:

Episode: 1, total numsteps: 50, episode steps: 50, reward: -39.

Episode: 2, total numsteps: 100, episode steps: 50, reward: -49.

Episode: 1539, total numsteps: 76950, episode steps: 50, reward: -12.

Episode: 1540, total numsteps: 77000, episode steps: 50, reward: -11.

Episode: 2519, total numsteps: 125950, episode steps: 50, reward: -4.

Episode: 2520, total numsteps: 126000, episode steps: 50, reward: -4.

We have all the plots and graphs in tensorboardx.


**References**

[1].

Greg Brockman et al. “Openai gym”. In: arXiv preprint arXiv:1606.01540 (2016).

[2].

Tom Erez, Yuval Tassa, and Emanuel Todorov. “Simulation tools for model-based
robotics: Comparison of bullet, havok, mujoco, ode and physx”. In: 2015 IEEE
international conference on robotics and automation (ICRA). IEEE. (2015), pp. 4397–
4404.

[3].

https://towardsdatascience.com/in-depth-review-of-soft-actor-critic-91448aba63d

[ 4 ].

[http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)

[5].

https://www.diva-portal.org/smash/get/diva2:1415901/FULLTEXT01.pdf

[6].

https://gym.openai.com/envs/Reacher-v2/


