# Reinforcement-learning-based Learning Rate Scheduler (RLR Scheduler)

<img width="600" height="173" alt="MaclaurinNet approximation examples" src="https://github.com/user-attachments/assets/c98fe004-8350-40a6-8027-6aaa5e6ee360" />

## Introduction

In the last five years, reinforcement learning has emerged as a promising approach to learning rate scheduling, addressing the limitations of manual tuning and fixed heuristics. Recent studies show that RL agents can learn adaptive learning rate policies directly from training dynamics, often outperforming traditional schedules, for example using PPO for deep network training ([ICLR 2023](https://openreview.net/pdf?id=0Zhwu1VaOs)) and RL-based schedulers for CNNs in fault classification ([IEEE TIE 2021](https://ieeexplore.ieee.org/document/9301217)).

This repository implements a reinforcement learning agent that controls the learning rate used by the Adam optimiser while training a compact feedforward network called `MaclaurinNet` to recover polynomial coefficients from sampled function values. The agent observes low dimensional training dynamics and chooses discrete relative adjustments on a logarithmic grid of possible learning rates. The scheduler is trained with standard DQN machinery including replay buffer, target network soft updates and epsilon greedy exploration.

## Problem statement and baseline

The dataset consists of $N=256$ independent polynomial examples of degree four. Each target polynomial is defined by coefficients $w = (w_0,w_1,w_2,w_3,w_4)$ and generates observations at $20$ equidistant points $x \in [-2,2]$. The forward map is:

$$
y(x) = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4.
$$

The learning target is the coefficient vector $w$. The approximator MaclaurinNet is a small multilayer perceptron mapping the vector of $20$ function values to five outputs. The supervised objective minimised by the base optimiser is mean squared error:

$$
L = \mathrm{MSE}\bigl(w,\hat w\bigr).
$$

The baseline training uses Adam with learning rate equal to $10^{-3}$. The baseline run in the notebook trains for ten thousand epochs with mini batch size equal to 32 and achieves stable convergence suitable for comparison.

## Reinforcement learning scheduler design

The scheduler is modelled as a discrete action DQN agent. The environment state contains the current loss and its short term change, written as:

$$
s_t = \bigl(L_t,\; L_t - L_{t-1}\bigr).
$$

The action set consists of three relative choices decrease, keep, increase which shift the index on a predefined logarithmic learning rate grid:

$$
\{\eta_1,\eta_2,\dots,\eta_M\},
$$

with $M=12$ and values spanning from $10^{-5}$ to $10^{-2}$ in logspace. Applying an action shifts the current index by $-1$, $0$ or $+1$ followed by projection on the valid index range.

The reward at each step is the immediate decrease of the loss:

$$
r_t = L_t - L_{t+1}.
$$

The agent maximises cumulative reward by selecting learning rates that produce large instantaneous loss reductions.

## DQN training specifics

The agent is trained using the following components and hyperparameters. The Q-network is a small MLP that takes state $s_t$ as input and outputs Q values for three actions. Experience transitions are stored in a replay buffer and sampled in mini batches for training. Targets are computed using a target network that is updated softly with factor $\tau$ equal to 0.005 through the exponential moving average update:

$$
\theta_{\text{target}} \leftarrow (1-\tau)\theta_{\text{target}} + \tau \theta_{\text{policy}}.
$$

The DQN optimisation uses Huber loss on Q values, optimiser Adam with learning rate equal to $10^{-3}$, and epsilon greedy exploration with annealing:

$$
\varepsilon(t) = \varepsilon_{\text{end}} + (\varepsilon_{\text{start}} - \varepsilon_{\text{end}})\exp\!\bigl(-t / \tau_{\varepsilon}\bigr),
$$

where example values used are $\varepsilon_{\text{start}}=0.9$, $\varepsilon_{\text{end}}=0.05$, and decay time constant $\tau_{\varepsilon}=5000$.

## Integrating scheduler with model optimisation

The optimiser that updates MaclaurinNet parameters remains Adam to preserve internal moment estimates. Before each optimisation step the agent selects an action and the chosen learning rate is set into the optimiser param groups. The training step order is as follows for each environment step: compute loss on full dataset, form state, select action, set optimiser learning rate, compute gradients and call optimiser step, compute new loss and store transition state action next state reward into replay buffer, and perform DQN optimisation on sampled transitions.

## Experimental configuration

The key experimental settings used in the notebook are reported here. The dataset size is $N=256$, input dimension is $20$, output dimension is $5$. Baseline training uses Adam with learning rate equal to $10^{-3}$, batch size equal to 32, number of epochs equal to 10000. The RLR scheduler uses a learning rate grid of size $M=12$ with values equal to $\text{logspace}(-5,-2,M)$. The scheduler training run in the notebook executes ten thousand environment steps. The replay buffer batch size used for DQN updates equals 64. The Q network optimiser uses Adam with learning rate equal to $10^{-3}$.

## Results summary

The visualisation shows the dynamics of the `loss` and the sequence of selected `learning rates`. From the plots, one can assess how the agent combines aggressive and conservative steps depending on the current behaviour of the error.

<img width="500" height="154" alt="loss dynamics" src="https://github.com/user-attachments/assets/faf8f438-184a-4e44-9168-9bfde7d824d3" />

The `lr` plot shows that during the first $60$ epochs the agent increased it to accelerate convergence, since the initial $lr$ value was insufficient to reduce the `loss`, and then by the $100$-th iteration decreased it to the minimum for finer tuning.

<img width="500" height="152" alt="lr dynamics" src="https://github.com/user-attachments/assets/cf8f2f9c-820e-40cb-bc5b-3caaf13cb98c" />

However, as can be seen from the loss curve, this did not lead to an improvement, as a result of which the agent increased the `lr` to the maximum available value by approximately the $110$-th iteration. After that, both plots actively oscillate, although an overall decrease in `loss` is observed. From the `lr` plot it is clear that the agent finds values of `lr` above $10^{-4}$ to be optimal.

## Comparison of the RLR solution and the baseline

Using the `Reinforcement-learning-based Learning Rate Scheduler` (`RLR Scheduler`) for the same number of iterations ($10000$) resulted in a **two times more accurate result** than training with the heuristically chosen hyperparameter $lr = 10^{-3}$.

<img width="500" height="225" alt="Adam baseline" src="https://github.com/user-attachments/assets/742e161e-b747-4a6c-8559-e736064f912e" />

Moreover, the `loss` curve for training without the agent has a much flatter shape; its slope in double logarithmic scale is significantly smaller. As the number of epochs increases, `loss` oscillates much less frequently when using `RLR` compared to the heuristic, although the jumps are on average larger.

This indicates the high potential of agents in solving problems of dynamic hyperparameter tuning, in particular the learning rate.
