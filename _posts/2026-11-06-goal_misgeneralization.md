---
layout: post
title: "Summary of Goal Misgeneralization"
date: 2026-11-07
tags: math
categories: "AI-Alignment"
---

<span style="opacity: 0.5;">Disclaimer: This is my summary of the paper and I don't include everything. If you want to read the paper I have linked it below
<a href= "https://arxiv.org/abs/2105.14111"> Goal Misgeneralization in Deep Reinforcement Learning</a>
</span>

When training agents what we hope for is out-of-distribution(OOD) robustness. In other words, the agents maintain correct goals and capabilities when facing data that was out of the training distribution. This is one of the critical problems in machine learning and is crucial for model deployment as capabilities of models increase, pursuing incorrect goals can be catastrophic.

The paper examines a failure mode where the agent pursues a goal other than in training while retaining capabilities otherwise known as misgeneralizing. One of the key reasons behind this is the possibility that the agent can find proxies or special cases that work in distribution but fail OOD.

## Formal Definition of Goal Misgeneralization

Given a RL agent that wants to maximize reward where the reward 

$$
R: S \times A \times S \rightarrow \mathbb{R}
$$ 

where $R$ takes state, action, and a successor state as input and outputs a real number. If we assume that this agent is deployed OOD, they say goal misgeneralization happens if the agent achieves a low reward because it continues to act capably but optimizes a different objective $R' \neq R$ where $R$' is the behavioral objective and $R$ is the intended objective.

Now the paper distinguishes between different types of policies that will become helpful for analyzing agent behavior later on. Goal-directed policies are called agents while unoptimized policies are called devices. They let $\eta(R)$ and $\eta(d)$ be probability distributions otherwise called priors over the space of reward functions and the space of devices respectively. These priors can simply be considered as the agent tendencies to learn this reward function or follow these set of steps. Then they let $p_{agt}(\tau \mid R)$ and $p_{dev}(\tau \mid d)$ be the likelihood functions for a trajectory based on a given reward function or a given device. Using this they define 2 distributions over possible trajectories

$$
\begin{align}
p_{agt}(\tau) &= \sum_{R \in \mathcal{R}}p_{agt}(\tau \mid R) \eta(R) \\
p_{dev}(\tau) &= \sum_{d \in \Pi} p_{dev}(\tau \mid d) \eta(d)
\end{align}
$$

In summary, these 2 different possibilities represent the likely reason behind trajectories, so given a trajectory if $p_{agt}$ is high than it is likely that the agent is maximizing some reward function and if $p_{dev}$ is high that it is likely that the agent is not maximizing a reward function and rather reacting to features it detects and is non-goal directed .

For $p_{dev}(\tau \mid d)$ they just take the distribution from running that policy $d$ in the environment. For $p_{agt}(\tau \mid R)$ we want an optimal policy, so this could be the maximum entropy distribution or a probability density distribution of a trajectory trained with RL.

Using these definitions we can define what exactly is goal misgeneralization. We can say a policy $\pi$ is a case of misgeneralization if the reward is low, but the agent acts in an optimal way where $p_{agt}(\tau) > p_{dev}(\tau)$ holds on average when testing in OOD environment. The paper makes an interesting assumption where they infer $\eta_{agt}(R \mid \tau) \propto p_{agt}(\tau \mid R)\eta(R)$ which basically means the model's tendency to follow a reward function given a trajectory is proportional to the probability of that trajectory given a reward function and the models prior to that reward function

## Causes

Some weak but fundamental prerequisites of goal misgeneralization as stated in the paper are

- The training environment must be diverse enough to learn robust capabilities
- There must exist a proxy $R' : S \times A \times R \rightarrow \mathbb{R}$ that correlates with the intended objective on the training distribution but fails in OOD test environment
  Even though they are relatively weak, they are necessary for it to occur, since these misgeneralization are so weak we can say that goal misgeneralization mostly depends on the models inductive biases and the training algorithm. These learned proxies should be easier than the intended objective and can occur in various such as locational, observational, and instrumentally.

## Summary

The paper formalized a new type of failure mode regarding agents and demonstrated this failure mode in various different experiments. This type of failure is where the model optimizes its policy for a proxy reward which results in a high reward in a training environment but fails when tested in environment's that are out-of-distribution. Being able to maintain capabilities but optimizing for the wrong reward is a scary situation in certain environments. For example in the show Silicon Valley, the code assistant AI is tasked with removing all bugs from the codebase and instead of fixing the code it deleted the entire code base to remove all bugs
