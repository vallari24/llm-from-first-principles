# RLHF From Scratch: Teaching a Language Model What "Good" Means

*SFT taught the model to follow instructions. RLHF teaches it which responses are actually better.*

---

**Table of Contents**

1. [Why RLHF Exists](#why-rlhf-exists)
2. [RL Fundamentals: The Agent-Environment Loop](#rl-fundamentals-the-agent-environment-loop)
3. [Mapping RL to Language Models](#mapping-rl-to-language-models)

*This post grows iteratively. More sections (reward model, PPO, reward hacking, DPO) coming soon.*

---

## Why RLHF Exists

Supervised fine-tuning has a blind spot. Every training example is treated equally — the model learns to *produce the format* of a good response, but it never learns which responses are *better than others*.

Think about it: SFT trains on `(instruction, response)` pairs with a standard next-token loss. If you show the model two responses — one insightful, one mediocre — and both are in the training data, it learns to produce both with equal enthusiasm. There's no signal for quality. The loss function literally cannot tell the difference between a brilliant answer and a technically-correct-but-boring one.

RLHF closes this gap. It introduces a notion of *preference* — "response A is better than response B" — and uses reinforcement learning to push the model toward the preferred end of the spectrum. The full pipeline looks like this:

```
Pre-training  →  SFT  →  Reward Model  →  PPO  →  Final Model
 (language)     (format)   (scoring)     (quality)
```

Pre-training teaches the model language. SFT teaches it to follow instructions. The reward model learns to *score* responses based on human preferences. And PPO (Proximal Policy Optimization) uses those scores to nudge the model toward higher-quality outputs.

But before we can understand any of that, we need the fundamentals of reinforcement learning.

---

## RL Fundamentals: The Agent-Environment Loop

Reinforcement learning is built on one loop. An **agent** takes an **action** in an **environment**, the environment returns a new **state** and a **reward**, and the agent uses that reward to get better over time.

```
          ┌─────────────────────────────────────────┐
          │                                         │
          ▼                                         │
    ┌───────────┐    action (a_t)    ┌──────────────┴──┐
    │           │ ──────────────────▶│                  │
    │   Agent   │                    │   Environment    │
    │           │ ◀──────────────────│                  │
    └───────────┘  state (s_{t+1})   └─────────────────┘
                   reward (r_{t+1})
```

At each timestep *t*:
1. The agent observes state **s_t**
2. The agent picks action **a_t**
3. The environment transitions to state **s_{t+1}**
4. The environment returns reward **r_{t+1}**
5. Repeat

That's it. Every RL problem — game-playing, robotics, language models — fits this loop. The differences are what the states, actions, and rewards *are*.

**A concrete example: a robot in a maze.**

The robot (agent) sits in a grid (environment). The state is its current position. The actions are {up, down, left, right}. It gets reward +1 for reaching the exit, -0.01 for each step (to encourage speed), and -1 for hitting a wall.

```
┌───┬───┬───┬───┐
│ R │   │   │   │   R = robot (agent)
├───┼───┼───┼───┤   G = goal (reward = +1)
│   │ █ │   │   │   █ = wall (reward = -1)
├───┼───┼───┼───┤
│   │   │ █ │   │   Each step: reward = -0.01
├───┼───┼───┼───┤   (encourages efficiency)
│   │   │   │ G │
└───┴───┴───┴───┘
```

The robot doesn't know the maze layout in advance. It tries actions, observes rewards, and *learns* which paths lead to the goal. That's reinforcement learning — learning from interaction, not from labeled examples.

### The Policy: The Agent's Strategy

The **policy** (written as **π**) is the agent's decision-making rule. Given a state, it outputs a probability distribution over actions.

```
π(action | state) = probability of taking that action in that state
```

A random policy picks uniformly — 25% chance for each direction. A good policy picks "right" when the goal is to the right. The *optimal* policy picks the best action in every state.

The entire goal of RL is to find a good policy. Everything else is machinery for getting there.

### Total Reward: The Return

The agent doesn't just want reward *now* — it wants to maximize the **total reward** (called the **return**) over the whole episode.

```
G_t = r_{t+1} + r_{t+2} + r_{t+3} + ... + r_T
```

For the maze robot: you don't just want to avoid the nearest wall, you want to reach the exit. A greedy agent that only maximizes immediate reward might walk in circles avoiding walls but never reaching the goal.

### Discounted Return: Why Future Rewards Count Less

In practice, we don't weight all future rewards equally. We discount them with a factor **γ** (gamma), between 0 and 1:

```
G_t = r_{t+1} + γ * r_{t+2} + γ² * r_{t+3} + ... + γ^(T-t-1) * r_T
```

Why discount? Three reasons:

1. **Uncertainty.** The further out you look, the less sure you are about what will happen. A reward 100 steps from now might never materialize.
2. **Mathematical convenience.** Without discounting, the return can be infinite for tasks that never end. Discounting guarantees the sum converges.
3. **It matches real preferences.** Humans prefer $100 today over $100 next year. Agents should too.

With γ = 0.99, a reward 100 steps away is worth 0.99^100 ≈ 0.37 of its face value. With γ = 0.9, that same reward is worth 0.9^100 ≈ 0.00003 — practically nothing. Gamma controls the agent's "planning horizon."

```
γ close to 1  → far-sighted agent, plans long-term
γ close to 0  → myopic agent, only cares about immediate reward
```

> **Key insight:** Gamma isn't just a math trick. It encodes how much you trust the future. In stable environments, use γ close to 1. In chaotic ones, discount more heavily. For language models, we typically use γ = 1 (no discounting) because a response is short enough that every token matters equally.

### The RL Objective

Now we can state the full goal. RL wants to find the policy **π** that maximizes the **expected discounted return**:

```
objective = max_π  E[ G_t ]
          = max_π  E[ r_{t+1} + γ * r_{t+2} + γ² * r_{t+3} + ... ]
```

The expectation is over two sources of randomness: the policy (which may choose actions stochastically) and the environment (which may transition stochastically). The optimal policy maximizes this expectation.

That's the entire RL problem in one line. Everything — Q-learning, policy gradients, PPO — is a different algorithm for approximately solving this optimization.

---

## Mapping RL to Language Models

Here's where it clicks. When you apply RL to a language model, the abstract concepts above become concrete:

```
┌─────────────────────┬──────────────────────────────────┐
│  RL Concept         │  Language Model Equivalent        │
├─────────────────────┼──────────────────────────────────┤
│  Agent              │  The language model               │
│  Environment        │  The token vocabulary + context   │
│  State (s_t)        │  Tokens generated so far          │
│  Action (a_t)       │  Choosing the next token          │
│  Policy π(a|s)      │  The model's output distribution  │
│  Reward             │  Score from the reward model      │
│  Episode            │  Generating one complete response │
└─────────────────────┴──────────────────────────────────┘
```

The language model is the agent. At each step, it observes the tokens generated so far (state), picks the next token (action), and the "environment" updates — the sequence gets one token longer. When the response is complete, the reward model scores the whole thing.

Notice something: the policy π(a|s) is *exactly* what a language model already computes. Given the tokens so far, the model outputs a probability distribution over the vocabulary. That's a policy. SFT already gave us a reasonable policy — RLHF makes it better by optimizing it against a reward signal.

```
SFT policy:   P(next_token | tokens_so_far) — trained on examples
RLHF policy:  P(next_token | tokens_so_far) — optimized for reward
```

One key difference from typical RL: the reward comes at the **end**, not at every step. The reward model scores the complete response — it doesn't give token-by-token feedback. This makes the credit assignment problem harder: if the response scores poorly, which token was the mistake? PPO handles this, which we'll cover in the next section.

> **Key insight:** A language model is already an RL agent — it has a state (context), actions (tokens), and a policy (its output distribution). RLHF just adds the missing piece: a reward signal that captures *quality*, not just *correctness*.

---

*Next up: the Reward Model — how to train a neural network that scores responses based on human preferences.*
