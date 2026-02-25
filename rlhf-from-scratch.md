# RL Fundamentals: The Building Blocks Behind RLHF

*Before you can understand how RLHF teaches a language model what "good" means, you need to understand reinforcement learning.*

---

**Table of Contents**

1. [Why RLHF Exists](#why-rlhf-exists)
2. [RL Fundamentals: The Agent-Environment Loop](#rl-fundamentals-the-agent-environment-loop)
3. [Mapping RL to Language Models](#mapping-rl-to-language-models)

*This post covers RL fundamentals — the foundation you need before understanding RLHF. Future posts will cover the reward model, PPO, reward hacking, and DPO.*

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

### Why Cumulative Reward? The Return

The agent doesn't just want reward *now* — it wants to maximize the **total reward** (called the **return**) over the whole episode. The return is a cumulative sum:

```
G_t = r_{t+1} + r_{t+2} + r_{t+3} + ... + r_T
```

Why cumulative? Because a single action isn't the end of the story. Consider:

```
The agent is at position A. It goes right to B. Then from B, down to the Goal.

Step 1: A → right → B     immediate reward: -0.01 (step cost)
Step 2: B → down  → Goal  immediate reward: +1.00 (goal!)

Total return from A = -0.01 + 1.00 = 0.99
```

If you only looked at the immediate reward, "go right" looks bad (-0.01). But the total return reveals the truth: going right *leads to the goal*. An agent that only maximizes immediate reward might walk in circles avoiding walls but never reaching the exit. The cumulative sum captures **consequences**, not just the immediate result.

### Discounted Return: Why Future Rewards Count Less

In practice, we don't weight all future rewards equally. We discount them with a factor **γ** (gamma), between 0 and 1:

```
G_t = r_{t+1} + γ * r_{t+2} + γ² * r_{t+3} + ... + γ^(T-t-1) * r_T
```

Using the same example with γ = 0.9:

```
From A: -0.01 + 0.9 * (1.00) = 0.89
Instead of 0.99, the goal reward is slightly faded because it's one step away.

If the goal were 10 steps away: 0.9^10 * 1.00 = 0.35
The further the reward, the less the agent cares.
```

Why discount? Three reasons:

1. **Uncertainty.** The further out you look, the less sure you are about what will happen. A reward 100 steps from now might never materialize.
2. **Mathematical convenience.** Without discounting, the return can be infinite for tasks that never end. Discounting guarantees the sum converges.
3. **It matches real preferences.** Humans prefer $100 today over $100 next year. Agents should too.

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

### The Q-Function: Rating State-Action Pairs

The objective says "maximize expected return." But how does an agent actually *decide* which action to take? It needs a way to evaluate: "If I'm in this state and take this action, how good is that?" That's the **Q-function**.

```
Q(s_t, a_t) = E[ G_t | s_t, a_t ]
```

Q(s, a) answers one question: "If I'm in state *s* and take action *a*, what total discounted reward do I expect to get from here until the end?"

**"Future reward" means: what actually happened after this step.** During training, the agent takes an action, then keeps playing, and *eventually* the episode ends. You sum up all the rewards that came after. That's the "future" part — it's not a mystery, it's just what happened next. The tricky part is that the agent needs to *predict* this sum *before* taking the action, based on what happened in previous episodes.

Let's trace how Q gets built, starting from nothing.

**Episode 1 — the agent knows nothing.** All Q-values are zero. Every action looks equally good (or bad). The agent picks randomly.

```
Episode 1 Q-table (knows nothing):
                  up     down    left    right
position (0,0):  [0.0    0.0     0.0     0.0  ]
position (0,1):  [0.0    0.0     0.0     0.0  ]
position (1,2):  [0.0    0.0     0.0     0.0  ]
position (2,3):  [0.0    0.0     0.0     0.0  ]
...every cell is zero. The agent wanders randomly.
```

It stumbles around. Goes right, gets -0.01. Goes down, gets -0.01. Hits a wall, gets -1. Maybe after many random steps it accidentally reaches the goal, gets +1. Now it updates: the cell *right next to the goal* gets a slightly positive Q-value for "move toward goal." Everything else is still mostly zero.

**Episode 100 — the agent has experience.** The cells near the goal have high Q-values (the agent knows how to get there from nearby). Cells near walls have negative Q-values. The agent has learned the landscape.

```
Episode 100 Q-table (has learned):
                  up     down    left    right
position (0,0):  [0.1    0.3     0.0     0.2  ]  ← "down seems good from here"
position (1,2):  [-0.5   0.1     0.2    -0.8  ]  ← "don't go right, wall there"
position (2,3):  [0.1    0.0     0.1     0.9  ]  ← "go right! goal is there!"
```

The learning spreads backward from the goal like a wave. First the cells adjacent to the goal learn good Q-values. Then the cells adjacent to *those*. Then the next ring out.

```
Episode 1:    Only the cell before the goal has a useful Q-value
Episode 10:   Cells 2-3 steps from goal are learning
Episode 100:  Most of the maze has reasonable Q-values
Episode 1000: Q-values are well-calibrated everywhere
```

**How many episodes?** It depends on the problem. A 4x4 maze might need ~100. Atari games need millions. The harder the environment, the more episodes to build good estimates. In practice, you train until the Q-values stop changing much (convergence).

**Why greedy on immediate reward fails.** Say position (1,1) has: going up gives immediate reward -0.2 (step penalty, rough terrain), going down gives +0.2 (bonus tile). A greedy agent always picks down. But going up leads through rough terrain toward the goal (total return: +5.0), while going down leads to a dead end (total return: +0.2, then stuck). Q captures the full picture — Q(up) = 5.0 despite the immediate -0.2.

**The Bellman equation: how Q is actually computed.** You don't literally need to play thousands of episodes from each position. There's a recursive shortcut:

```
Q(s, a) = r + γ * max_a' Q(s', a')

"The value of doing a in state s =
    the immediate reward I get
    + the discounted value of the best thing I can do next"
```

You start with all zeros and keep applying this equation. Each update pulls information from neighboring cells. The update after hitting a wall: Q(pos, right) was 0.0, now it shifts toward -1.0. Next time the agent is here, it knows: don't go right.

**Once you have Q, the optimal policy is trivial:** in any state, pick the action with the highest Q-value.

```
π*(s) = argmax_a Q(s, a)

"In this state, which action has the best expected future?"
```

**Q as a table vs. a neural network.** For small problems like our 4x4 maze, Q is literally a table — one row per state, one column per action. But for large problems (chess has ~10^43 states, a language model has effectively infinite states), a table is impossible. Instead, you use a **neural network** that takes in the state and outputs Q-values. Same idea, but the network *generalizes* — it estimates Q for states it's never seen, based on similar states it has.

```
Small problem (maze):  Q-table    (direct lookup)
Large problem (chess): Q-network  (neural net approximating the table)
LLMs:                  Skip Q entirely — PPO optimizes the policy directly
```

> **Key insight:** Q is not knowledge of the future — it's a *learned prediction* about the future, refined through experience. During training, the agent plays episodes and sees what actually happens. Q averages those outcomes. At decision time, Q tells the agent "based on everything I've seen, going right from here tends to work out well." It's a prediction, not a fact — but with enough episodes, it's a very good one.

### Two Families of RL: Value Learning vs. Policy Learning

We know the goal: find the best action in every state. There are two fundamentally different strategies to get there.

```
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│        VALUE LEARNING           │   │        POLICY LEARNING          │
│                                 │   │                                 │
│   Learn to EVALUATE options     │   │   Learn to ACT directly        │
│                                 │   │                                 │
│   Step 1: Learn Q(s, a)        │   │   Step 1: Learn π(s)           │
│           "score every option"  │   │           "what should I do?"  │
│                                 │   │                                 │
│   Step 2: a = argmax Q(s, a)   │   │   Step 2: a ~ π(s)            │
│           "pick the best score" │   │           "sample from policy" │
│                                 │   │                                 │
│   Examples: Q-learning, DQN    │   │   Examples: REINFORCE, PPO     │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

**Value learning** is the indirect approach. The agent never explicitly learns "what to do." It learns "how good is each option?" — that's Q. Then at decision time, it just picks the highest-scoring option. It's like a restaurant critic who scores every dish, then orders the highest-rated one.

**Policy learning** is the direct approach. The agent learns a function that maps states straight to action probabilities. No scoring, no Q-table. You ask "what should I do?" and the network outputs a distribution. Sample from it. It's like a chef who *knows* what to cook — no menu needed.

Let's see how each one plays out in the maze:

```
VALUE LEARNING (Q-learning):

The agent at (1,1) consults its Q-table:
┌──────────────────────────────┐
│  Q((1,1), up)    = 0.7  ★   │  ← highest score
│  Q((1,1), down)  = 0.3      │
│  Q((1,1), left)  = 0.1      │
│  Q((1,1), right) = -0.2     │
└──────────────────────────────┘
Pick argmax → go up.

What was learned: a SCORE for every action.
How to decide: pick the action with the best score.
```

```
POLICY LEARNING (policy gradient):

The agent at (1,1) consults its policy network:
┌──────────────────────────────┐
│  π((1,1), up)    = 0.60  ★  │  ← highest probability
│  π((1,1), down)  = 0.25     │
│  π((1,1), left)  = 0.10     │
│  π((1,1), right) = 0.05     │
└──────────────────────────────┘
Sample from distribution → most likely go up.

What was learned: a PROBABILITY for every action.
How to decide: sample from the distribution.
```

Both end up going up — but the reasoning is different. Value learning says "up has the best score." Policy learning says "up has the highest probability." The end result is similar, but the path to get there matters.

**Why does this distinction matter for LLMs?**

A language model *already is* a policy. It takes tokens in (state) and outputs a probability distribution over the next token (action). That's π(s). It's already doing policy learning.

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  "The cat sat on the ___"                               │
│        ↓                                                │
│  ┌───────────┐                                          │
│  │    LLM    │  ← this IS π(s)                          │
│  └─────┬─────┘                                          │
│        ↓                                                │
│  P("mat")  = 0.35  ★                                    │
│  P("dog")  = 0.12                                       │
│  P("roof") = 0.08                                       │
│  P("the")  = 0.06                                       │
│  ...50,000 more tokens...                               │
│                                                         │
│  The model outputs action probabilities directly.       │
│  No Q-table needed. The model IS the policy.            │
└─────────────────────────────────────────────────────────┘
```

If you tried value learning for an LLM, you'd need to learn Q(tokens_so_far, next_token) for every possible token at every possible context — and the vocabulary is 50,000+ tokens with effectively infinite possible contexts. Why build a separate scoring system when you already have a network that outputs action probabilities?

```
Value learning for LLMs:  Build Q-network → score 50k tokens → pick max
                          (wasteful — you already have a model that picks tokens)

Policy learning for LLMs: Take existing model → nudge toward higher reward
                          (natural — the model IS the policy, just improve it)
```

This is why RLHF uses **PPO** (a policy learning algorithm). PPO takes the existing language model, generates responses, scores them with the reward model, and nudges the policy so that high-reward responses become more likely. No Q-table, no separate scoring network — just direct improvement of the policy you already have.

> **Key insight:** Value learning and policy learning solve the same problem via different strategies. Value learning asks "how good is each option?" and picks the best. Policy learning asks "what should I do?" directly. Language models are natural policies — they already output action probabilities — which is why RLHF uses policy learning (PPO) rather than value learning (Q-learning).

### Scaling Q to Neural Networks: Deep Q-Networks (DQN)

For a 4x4 maze, the Q-table has 16 rows and 4 columns — 64 numbers. Easy. But what about Atari, where the state is a game screen with 210x160 pixels in color? That's millions of possible states. A table can't hold that.

The fix: replace the table with a **neural network**. Feed in the game screen, get Q-values for every action out.

```
┌───────────┐         ┌────────────┐      Q(s, up)    = 0.2
│ game      │         │            │      Q(s, down)  = 0.4
│ screen    │ ──────▶ │  Deep NN   │ ──▶  Q(s, left)  = 0.1
│ (pixels)  │         │            │      Q(s, right) = 0.73  ★
└───────────┘         └────────────┘
   state s             the agent         one forward pass → all Q-values
```

One forward pass gives you Q-values for every action at once. Pick the max. This is **DQN** — Deep Q-Network — what DeepMind used to play Atari at superhuman level.

But how do you *train* this network? What's the loss function? This is where the Bellman target becomes a training signal.

### The Bellman Target: How Q Actually Learns

The Bellman equation isn't just a formula — it's a **training target**. It tells the network what the Q-value *should* be, so you can compute a loss and backpropagate.

The intuition: **you only ever look one step ahead.**

Say the agent is at position A, goes right, lands at B, and got reward -0.01 for that step. Now it's standing at B. From past experience, the network already estimates that the best Q-value from B is about +0.90.

```
What I actually got (immediate reward):   -0.01   ← I KNOW this, it just happened
What I expect from here (best Q at B):    +0.90   ← I ALREADY ESTIMATED this
                                          ─────
Bellman target:                           +0.89

This is what Q(A, right) SHOULD be.
```

That's it. The Bellman target = immediate reward + discounted best-Q of the next state. No predicting the distant future. Just one step ahead, trusting your existing estimates for the rest.

Think of it like a road trip. You don't need to know the entire route to evaluate a highway choice:

```
Highway A:                              Highway B:
  First segment: 2hrs bad road            First segment: 1hr good road
  Best estimate from junction: 3hrs       Best estimate from junction: 6hrs
  Total: 5hrs  ★                          Total: 7hrs

You only need: (1) how bad is the NEXT segment?
               (2) what's my best estimate FROM THERE?
```

### The Training Loop: Turning Bellman Into a Loss Function

Now it's just supervised learning. The network predicts a Q-value, the Bellman equation gives you the target, and the loss is the squared difference.

```
Training step:

1. Agent plays, records experience:
   (state=A, action=right, reward=-0.01, next_state=B)

2. Network predicts:
   Q(A, right) = 0.50                   ← current prediction

3. Compute Bellman target:
   target = r + γ * max_a' Q(B, a')
         = -0.01 + 0.99 * 0.90
         = 0.88                          ← what it SHOULD be

4. Loss = (prediction - target)²
        = (0.50 - 0.88)²
        = 0.14                           ← how wrong we are

5. Backprop → nudge network weights so Q(A, right) moves toward 0.88

6. Repeat thousands of times.
```

The beautiful thing: this is **bootstrapping**. The target itself uses Q estimates. Early on, those estimates are garbage — so the targets are garbage. But each update makes them slightly better. And the improvements ripple backward from states where you *know* the answer:

```
Episode 1:    Cell next to goal learns    → "Q ≈ 1.0, goal is right here"
Episode 10:   2 steps away improves       → "next cell is ~1.0, so I'm ~0.99"
Episode 50:   5 steps away improves       → "cell ahead is ~0.97, so I'm ~0.96"
Episode 200:  Entire maze is calibrated   → every cell has a good Q-value

Each cell only looks ONE step ahead.
The chain of one-step lookups covers the whole maze.
```

> **Key insight:** The Bellman target turns RL into supervised learning. You don't need to know the future — you only look one step ahead and trust your own (improving) estimates for the rest. The loss is just (prediction - target)². Train long enough, and the one-step estimates chain together to cover the entire problem. This is why RL works — it decomposes an impossible problem (predict total future reward) into a simple one (predict one step ahead, repeatedly).

### Why Q-Learning Breaks Down (and Why Policy Learning Wins for LLMs)

Q-learning works great for mazes and Atari. But it has three fundamental limitations that make it a poor fit for harder problems — and especially for language models.

**Limitation 1: Must score every action.** Q-learning ends with `argmax` — pick the highest-scoring action. That means you need to compute Q for *every* possible action. In a maze, that's 4 actions. In a vocabulary of 50,000 tokens, that's 50,000 scores per step. Possible, but expensive. For a robot arm with continuous joint angles — how many degrees to turn? 23.5°? 23.51°? — there are *infinite* possible actions. You can't argmax over infinity.

```
Maze:        4 actions   → score all 4, pick max. Easy.
Atari:       18 actions  → score all 18, pick max. Fine.
Robot arm:   ∞ actions   → score them all? Impossible.
LLM:         50k tokens  → score all 50k every step? Expensive.
```

Policy learning doesn't have this problem. It outputs probabilities directly — no need to enumerate and score everything. For continuous actions, it outputs distribution parameters (mean=23.5°, std=2°) and samples.

**Limitation 2: Deterministic — same state, same answer, every time.** Q-learning uses `argmax`: same state → same highest Q-value → same action. Always. No randomness. Why is that bad?

Think about Rock-Paper-Scissors. If you always play Rock (deterministic), your opponent plays Paper every time. You lose. The *optimal* strategy is random: 33% each. No deterministic policy can win.

```
Deterministic:   "playing RPS" → always Rock → opponent plays Paper → lose
Stochastic:      "playing RPS" → 33/33/33    → opponent can't exploit you
```

For language models, deterministic = "write a poem about the sea" gives the *exact same poem* every time. That's temperature = 0. Stochastic sampling (temperature > 0) gives variety and creativity. The ability to *sometimes* pick a less-obvious word is what makes language feel natural.

```
Deterministic (temp=0):   "The sea is vast and blue."  Every. Single. Time.
Stochastic (temp=0.7):    "The ocean whispers ancient songs..."  (varied)
```

**Limitation 3: Doesn't explore well.** Because argmax always picks the action it currently thinks is best, the agent can get stuck. If it tried "right" once and got a bad reward, Q(right) goes negative. Now it *never* tries right again — even if that one bad experience was a fluke. Policy learning naturally explores because it samples from a distribution — low-probability actions still get tried occasionally.

```
Q-LEARNING                         POLICY LEARNING
┌────────────────────────┐         ┌────────────────────────┐
│ Must score EVERY action│         │ Outputs probs directly │
│ Can't do continuous    │         │ Handles any action     │
│ Same state = same act  │         │ Naturally stochastic   │
│ Exploration is hard    │         │ Explores by sampling   │
└────────────────────────┘         └────────────────────────┘
      ↓                                   ↓
  Good for: games with               Good for: robotics,
  few discrete actions                continuous control,
  (Atari, board games)                language models
```

> **Key insight:** Q-learning asks "which option is best?" and always picks it. Policy learning asks "what's the probability of each option?" and samples. Sampling means variety, exploration, and the ability to handle continuous and huge action spaces. For language models — which need creative, varied outputs across a 50,000-token vocabulary — policy learning is the natural fit.

### Policy Gradient: Try Stuff, Keep What Works

Policy gradient is the algorithm behind policy learning. The idea is almost too simple: run the policy, see what happens, make good actions more likely and bad actions less likely.

No Q-table. No Bellman equation. Just: **try, evaluate, adjust.**

The architecture is a single neural network that takes in the state and outputs action probabilities directly:

```
                                         ┌──────────────────┐
                                    ┌──▶ │ P(a₁|s) = 0.9  ◀── │  ← most likely
                                    │    └──────────────────┘
┌───────────┐     ┌────────────┐    │    ┌──────────────────┐        ┌──────────┐
│           │     │            │    ├──▶ │ P(a₂|s) = 0.1  ✗ │  ───▶  │ Sample:  │
│  state s  │ ──▶ │  Deep NN   │ ───┤    └──────────────────┘        │ a = a₁   │
│           │     │  (policy)  │    │    ┌──────────────────┐        └──────────┘
└───────────┘     └────────────┘    └──▶ │ P(a₃|s) = 0.0  ✗ │
                                         └──────────────────┘
                                                  ↑
                                         All probabilities sum to 1

Compare with DQN (value learning):
  DQN:              state → NN → Q-values → argmax → always same action
  Policy gradient:  state → NN → probabilities → SAMPLE → stochastic action
```

The network's weights ARE the policy. Training adjusts these weights so that high-reward actions get higher probabilities. No intermediate Q-values — the output is the decision.

```
EPISODE 1 (good outcome, return = +0.98):

  (0,0) policy: P(up)=.25 P(down)=.25 P(left)=.25 P(right)=.25
  Sampled: right → reward -0.01

  (0,1) policy: P(up)=.30 P(down)=.20 P(left)=.20 P(right)=.30
  Sampled: down  → reward -0.01

  (1,1) policy: P(up)=.25 P(down)=.25 P(left)=.25 P(right)=.25
  Sampled: right → reward +1.00 (goal!)

  Total return: +0.98 → GOOD episode
  Update: increase P(right) at (0,0), P(down) at (0,1), P(right) at (1,1)
```

```
EPISODE 2 (bad outcome, return = -2.04):

  (0,0) Sampled: left  → hit wall
  (0,0) Sampled: left  → hit wall again
  ...kept wandering, never found goal.

  Total return: -2.04 → BAD episode
  Update: decrease P(left) at (0,0), decrease every action we took
```

```
AFTER HUNDREDS OF EPISODES:

  Actions that led to the goal:  very likely
  Actions that led to walls:     very unlikely
  The policy learned — without ever computing Q.
```

**The formula.** The policy gradient update has one equation:

```
∇J(θ) = E[ R * ∇log π(a|s) ]
```

Each piece has a plain meaning:

```
R              = how good was this episode? (total return)
log π(a|s)     = how likely was the action I took?
∇log π(a|s)    = direction to push weights to make that action MORE likely
R * ∇log π(a|s)= scale by how good it went
```

The multiplication does all the work:

```
Good episode (R = +5):    push HARD → make these actions more likely
Meh episode  (R = +0.1):  push a little → barely change anything
Bad episode  (R = -3):    negative R FLIPS the gradient
                          → makes these actions LESS likely
```

Visualize the policy parameters as a position on a landscape. Expected reward is the height. Policy gradient computes the slope and walks uphill:

```
Expected                    ★ ← optimal policy
Reward    ╱‾‾‾‾‾‾‾‾‾╲
  ↑      ╱            ╲
  │     ╱              ╲
  │    ╱                ╲
  │   •→                      • = current policy
  │    gradient says:         → = "move this way to
  │    "go right"                  increase reward"
  └─────────────────────→
      policy parameters θ
```

**Why this formulation is powerful:**

**1. Handles both discrete and continuous actions.** For discrete actions (maze directions, vocabulary tokens), the network outputs one probability per option. For continuous actions (steering angle, force), the network outputs the **shape of a bell curve** — a mean and standard deviation — and you sample a number from it.

```
Discrete: "Which direction?"
  Network outputs: P(left)=0.1  P(right)=0.6  P(up)=0.2  P(down)=0.1
  Sample → right

Continuous: "How many degrees to turn the wheel?"
  Network outputs: mean = 23.5°, std = 2.0°
  This defines a bell curve:
           ╱╲
          ╱  ╲
         ╱    ╲
        ╱      ╲
    ───╱────────╲───
      19   23.5   28
            ↑
         most likely value
  Sample → 24.1°
```

Q-learning can't do this — it needs to `argmax` over every possible action. You can't argmax over infinite steering angles. Policy gradient just outputs curve parameters and samples.

**2. Built-in exploration that fades naturally.** Because you're *sampling* from a distribution, the agent automatically explores. Early in training, the distribution is spread out (high uncertainty) — the agent tries diverse actions. As it learns, the distribution narrows (low uncertainty) — actions become precise. The network *learns its own confidence*.

```
Early training (exploring):
  Discrete:   [0.25, 0.25, 0.25, 0.25]  → could pick anything
  Continuous: mean=23°, std=15°
              ╱────────────────╲          Wide. Tries 0° to 45°.
          ───╱──────────────────╲───

Late training (confident):
  Discrete:   [0.95, 0.02, 0.02, 0.01]  → almost always picks action 1
              but 5% of the time tries others (still exploring a little!)
  Continuous: mean=23.5°, std=0.5°
                      ╱╲                  Narrow. Consistently ~23.5°.
          ───────────╱──╲───────────
```

No special exploration strategy needed. Q-learning, by contrast, always picks the argmax — it *never* explores unless you add a hack (like epsilon-greedy: "pick a random action 10% of the time").

**3. Directly optimizes what you care about.** Q-learning optimizes Q-values and *hopes* good actions follow. Policy gradient optimizes the expected reward directly. No middleman.

**4. End-to-end differentiable.** Standard neural network. Gradient flows from reward straight to policy weights. Normal backprop.

**A concrete example: training a self-driving car.** The car drives down a road. At each moment it makes a steering decision. Eventually it either stays on the road (good) or crashes (bad). Let's trace one episode:

```
THE CAR'S JOURNEY:

  ● Step 1: steered slightly right   → on the road ✓   ┐
  ● Step 2: steered slightly left    → on the road ✓   ├── green zone (good)
  ● Step 3: steered right            → on the road ✓   ┘
  ● Step 4: steered hard right       → drifting off...  ┐
  ● Step 5: steered hard right       → off the road     ├── red zone (bad)
  ● Step 6: steered right            → CRASH 💥          ┘

  Total reward: negative
```

After this episode, policy gradient looks at each action:

```
UPDATE:
  Steps 1-3 (kept car on road):  INCREASE probability of those steering actions
  Steps 4-6 (caused crash):     DECREASE probability of those steering actions

  At step 4's state: P(hard right) goes DOWN
  → next time the car approaches that curve, it won't steer so aggressively
```

The full training algorithm:

```
┌─────────────────────────────────────────────────────────────┐
│  POLICY GRADIENT TRAINING ALGORITHM                         │
│                                                             │
│  1. Initialize agent        → random policy, car swerves    │
│  2. Run policy to end       → car drives until crash/done   │
│  3. Record everything       → every state, action, reward   │
│  4. Bad actions → less likely  (negative return flips grad) │
│  5. Good actions → more likely (positive return scales grad)│
│  6. Go to step 2, repeat                                    │
│                                                             │
│  Episode 1:    random swerving, crashes immediately         │
│  Episode 10:   survives a few seconds, crashes at curve     │
│  Episode 100:  makes it past the curve, crashes later       │
│  Episode 1000: drives smoothly, rarely crashes              │
└─────────────────────────────────────────────────────────────┘
```

This maps directly to LLMs:

```
Car:  drove a path  → crashed → decrease P(those steering actions)
LLM:  generated text → low reward → decrease P(those tokens)

Car:  drove a path  → stayed on road → increase P(those steering actions)
LLM:  generated text → high reward   → increase P(those tokens)
```

**The catch: high variance.** Policy gradient depends on *sampled* episodes. Two runs from the same state can give wildly different returns:

```
Episode 1: went right → got lucky  → return +10  → "RIGHT IS AMAZING"
Episode 2: went right → got unlucky → return -5   → "RIGHT IS TERRIBLE"

Same action, same state, opposite signals.
One episode is not enough — you need many to average out the noise.
```

This is why raw policy gradient (called REINFORCE) is slow and unstable. You need thousands of episodes for a reliable signal. More advanced algorithms like PPO (Proximal Policy Optimization) fix this by adding clipping and baselines to reduce variance — but the core idea remains the same.

> **Key insight:** Policy gradient is beautifully direct — try actions, see what happens, reinforce what works. The reward scales the gradient: good outcomes push actions to be more likely, bad outcomes push them to be less likely. For LLMs, this means: generate a response, score it, and nudge the model so high-scoring token sequences become more probable.

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
│  Q(s, a)            │  "How good is generating token a  │
│                     │   given the tokens so far?"       │
│  Episode            │  Generating one complete response │
└─────────────────────┴──────────────────────────────────┘
```

The language model is the agent. At each step, it observes the tokens generated so far (state), picks the next token (action), and the "environment" updates — the sequence gets one token longer. When the response is complete, the reward model scores the whole thing.

Notice something: the policy π(a|s) is *exactly* what a language model already computes. Given the tokens so far, the model outputs a probability distribution over the vocabulary. That's a policy. SFT already gave us a reasonable policy — RLHF makes it better by optimizing it against a reward signal.

```
SFT policy:   P(next_token | tokens_so_far) — trained on examples
RLHF policy:  P(next_token | tokens_so_far) — optimized for reward
```

One key difference from typical RL: the reward comes at the **end**, not at every step. The model generates a complete response, and *then* it gets scored. This makes the credit assignment problem harder: if the response scores poorly, which token was the mistake? This is one of the core challenges that RLHF algorithms need to solve.

> **Key insight:** A language model is already an RL agent — it has a state (context), actions (tokens), and a policy (its output distribution). RLHF just adds the missing piece: a reward signal that captures *quality*, not just *correctness*.

---

## Summary: RL Concepts You Need for RLHF

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL FUNDAMENTALS CHEAT SHEET                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Agent-Environment Loop                                         │
│    Agent observes state → picks action → gets reward → repeat   │
│                                                                 │
│  Policy π(a|s)                                                  │
│    The agent's strategy: given a state, probability of          │
│    each action. An LLM's output distribution IS a policy.       │
│                                                                 │
│  Return (cumulative reward)                                     │
│    Total reward over an episode, not just one step.             │
│    Captures consequences, not just immediate results.           │
│                                                                 │
│  Discounted Return (γ)                                          │
│    Future rewards count less. γ close to 1 = far-sighted.       │
│    For LLMs, γ = 1 (every token matters equally).               │
│                                                                 │
│  Q-function Q(s, a)                                             │
│    "If I'm here and do this, what total reward do I expect?"    │
│    A learned prediction, not a fact. Refined over episodes.     │
│                                                                 │
│  Bellman Equation                                               │
│    Q(s,a) = immediate reward + γ * best Q of next state.        │
│    Only looks one step ahead. Bootstraps from own estimates.    │
│                                                                 │
│  Value Learning vs Policy Learning                              │
│    Value: learn Q, pick argmax. Deterministic, can't do         │
│           continuous actions, poor exploration.                  │
│    Policy: learn π directly, sample from it. Stochastic,        │
│            handles any action space, explores naturally.         │
│    LLMs use policy learning — they already ARE policies.        │
│                                                                 │
│  Policy Gradient                                                │
│    Try actions → see results → good outcome? increase P.        │
│    Bad outcome? decrease P. Reward scales the update.           │
│    High variance → needs many episodes or smarter algorithms.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

These are the building blocks. With this foundation, you can understand how RLHF trains language models: a **reward model** scores responses based on human preferences, and a **policy gradient algorithm** (like PPO) nudges the language model toward higher-scoring outputs — reinforcing the good, suppressing the bad, one episode at a time.

*Next up: the Reward Model — how to train a neural network that learns human preferences.*
