```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Q-Learning
:label:`sec_qlearning`

이전 절에서는 완전한 마르코프 결정 과정(MDP), 예컨대 전이 함수와 보상 함수에 접근해야 하는 Value Iteration 알고리즘을 다뤘습니다. 이 절에서는 Q-Learning :cite:`Watkins.Dayan.1992`을 살펴봅니다. 이 알고리즘은 반드시 MDP를 알지 않고도 가치 함수를 학습하는 알고리즘입니다. 이 알고리즘은 강화 학습의 중심 아이디어를 구현합니다. 즉, 로봇이 자기 자신의 데이터를 얻을 수 있게 해줍니다.
<!-- , instead of relying upon the expert. -->

## Q-Learning 알고리즘

:ref:`sec_valueiter`에서 행동-가치 함수에 대한 value iteration이 다음 업데이트에 해당함을 떠올려 봅시다.

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

논의했듯이, 이 알고리즘을 구현하려면 MDP, 특히 전이 함수 $P(s' \mid s, a)$를 알아야 합니다. Q-Learning의 핵심 아이디어는 위 식의 모든 $s' \in \mathcal{S}$에 대한 합을 로봇이 방문한 상태들에 대한 합으로 대체하는 것입니다. 이로써 전이 함수를 알아야 하는 필요성을 우회할 수 있게 됩니다.

## Q-Learning의 기저에 있는 최적화 문제

로봇이 행동을 취하기 위해 정책 $\pi_e(a \mid s)$를 사용한다고 상상해 봅시다. 이전 장과 마찬가지로, $T$ 시간 단계로 이루어진 $n$개의 궤적 데이터셋 $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$을 수집합니다. value iteration이 사실은 서로 다른 상태와 행동의 행동-가치 $Q^*(s, a)$들을 서로 묶는 제약 조건들의 집합임을 떠올리세요. 로봇이 $\pi_e$를 사용해 수집한 데이터를 사용하여 value iteration의 근사적인 버전을 다음과 같이 구현할 수 있습니다.

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

먼저 이 식과 위의 value iteration 사이의 유사점과 차이점을 살펴봅시다. 만약 로봇의 정책 $\pi_e$가 최적 정책 $\pi^*$와 같고, 무한한 양의 데이터를 수집했다면, 이 최적화 문제는 value iteration의 기저에 있는 최적화 문제와 동일했을 것입니다. 그러나 value iteration이 $P(s' \mid s, a)$를 알아야 하는 것과 달리, 최적화 목적 함수에는 이 항이 없습니다. 저희는 속임수를 쓴 것이 아닙니다. 로봇이 상태 $s_t^i$에서 행동 $a_t^i$를 취하기 위해 정책 $\pi_e$를 사용함에 따라, 다음 상태 $s_{t+1}^i$는 전이 함수로부터 추출된 표본이 됩니다. 따라서 최적화 목적 함수도 전이 함수에 접근하지만, 그 접근은 로봇이 수집한 데이터의 형태로 암묵적으로 이루어집니다.

이 최적화 문제의 변수는 모든 $s \in \mathcal{S}$와 $a \in \mathcal{A}$에 대한 $Q(s, a)$입니다. 경사 하강법을 사용해 이 목적 함수를 최소화할 수 있습니다. 데이터셋의 모든 쌍 $(s_t^i, a_t^i)$에 대해 다음과 같이 쓸 수 있습니다.

$$\begin{aligned}Q(s_t^i, a_t^i) &\leftarrow Q(s_t^i, a_t^i) - \alpha \nabla_{Q(s_t^i,a_t^i)} \ell(Q) \\&=(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),\end{aligned}$$
:eqlabel:`q_learning`

여기서 $\alpha$는 학습률입니다. 일반적으로 실제 문제에서 로봇이 목표 위치에 도달하면 궤적은 종료됩니다. 그러한 종료 상태의 가치는 0인데, 로봇이 이 상태를 넘어 어떤 추가 행동도 취하지 않기 때문입니다. 이러한 상태를 처리하기 위해 업데이트를 다음과 같이 수정해야 합니다.

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

여기서 $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$은 $s_{t+1}^i$가 종료 상태이면 1, 그렇지 않으면 0인 지시 변수입니다. 데이터셋의 일부가 아닌 상태-행동 튜플 $(s, a)$의 가치는 $-\infty$로 설정됩니다. 이 알고리즘이 Q-Learning으로 알려져 있습니다.

이 업데이트의 해 $\hat{Q}$ (이는 최적 가치 함수 $Q^*$의 근사값입니다)가 주어지면, 이 가치 함수에 대응하는 최적 결정론적 정책을 다음과 같이 쉽게 얻을 수 있습니다.

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

동일한 최적 가치 함수에 대응하는 결정론적 정책이 여러 개일 수도 있는 상황도 있을 수 있습니다. 그러한 동점은 모두 같은 가치 함수를 가지므로 임의로 깨도 됩니다.

## Q-Learning에서의 탐험

데이터를 수집하기 위해 로봇이 사용하는 정책 $\pi_e$는 Q-Learning이 잘 동작하도록 보장하는 데 매우 중요합니다. 결국, 저희는 전이 함수 $P(s' \mid s, a)$를 사용한 $s'$에 대한 기댓값을 로봇이 수집한 데이터로 대체했기 때문입니다. 만약 정책 $\pi_e$가 상태-행동 공간의 다양한 부분에 도달하지 못한다면, 우리의 추정값 $\hat{Q}$가 최적의 $Q^*$에 대한 빈약한 근사가 될 것임을 쉽게 상상할 수 있습니다. 그러한 상황에서는 $\pi_e$가 방문한 상태만이 아니라 *모든 상태* $s \in \mathcal{S}$에서 $Q^*$의 추정값이 좋지 않게 된다는 점에 유의하는 것 또한 중요합니다. Q-Learning 목적 함수(또는 value iteration)는 모든 상태-행동 쌍의 가치를 서로 묶는 제약 조건이기 때문입니다. 따라서 데이터를 수집하기 위해 올바른 정책 $\pi_e$를 선택하는 것이 매우 중요합니다.

행동을 $\mathcal{A}$에서 균등하게 무작위로 표집하는 완전히 무작위인 정책 $\pi_e$를 선택함으로써 이러한 우려를 완화할 수 있습니다. 그러한 정책은 모든 상태를 방문하겠지만, 그렇게 되기까지 많은 수의 궤적이 필요할 것입니다.

이로써 저희는 Q-Learning의 두 번째 핵심 아이디어인 탐험(exploration)에 도달합니다. Q-Learning의 일반적인 구현은 $Q$의 현재 추정값과 정책 $\pi_e$를 함께 묶어 다음과 같이 설정합니다.

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$
:eqlabel:`epsilon_greedy`

여기서 $\epsilon$은 "탐험 파라미터"라 불리며 사용자가 선택합니다. 정책 $\pi_e$는 탐험 정책이라 부릅니다. 이 특정한 $\pi_e$는 $\epsilon$-탐욕 탐험 정책($\epsilon$-greedy exploration policy)이라 부릅니다. 확률 $1-\epsilon$로 (현재 추정값 $\hat{Q}$ 하에서의) 최적 행동을 선택하지만, 나머지 확률 $\epsilon$로는 무작위로 탐험하기 때문입니다. 이른바 소프트맥스 탐험 정책도 사용할 수 있습니다.

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/T}}{\sum_{a'} e^{\hat{Q}(s, a')/T}};$$

여기서 초매개변수 $T$는 온도(temperature)라 부릅니다. $\epsilon$-탐욕 정책에서 큰 $\epsilon$ 값은 소프트맥스 정책에서 큰 온도 값 $T$와 유사하게 동작합니다.

행동-가치 함수의 현재 추정값 $\hat{Q}$에 의존하는 탐험을 선택할 때, 주기적으로 최적화 문제를 다시 풀어야 한다는 점에 유의해야 합니다. Q-Learning의 일반적인 구현은 $\pi_e$를 사용해 매 행동을 취한 후, 수집된 데이터셋의 일부 상태-행동 쌍(일반적으로 로봇의 이전 시간 단계에서 수집된 것들)을 사용해 한 번의 미니배치 업데이트를 수행합니다.

## Q-Learning의 "자체 교정(self-correcting)" 속성

Q-Learning 동안 로봇이 수집하는 데이터셋은 시간이 지남에 따라 늘어납니다. 탐험 정책 $\pi_e$와 추정값 $\hat{Q}$는 모두 로봇이 더 많은 데이터를 수집함에 따라 진화합니다. 이는 Q-Learning이 잘 동작하는 이유에 대한 핵심적인 통찰을 제공합니다. 어떤 상태 $s$를 생각해 봅시다. 만약 어떤 특정 행동 $a$가 현재 추정값 $\hat{Q}(s,a)$ 하에서 큰 값을 가진다면, $\epsilon$-탐욕과 소프트맥스 탐험 정책 모두 이 행동을 선택할 확률이 더 큽니다. 만약 이 행동이 실제로 이상적인 행동이 *아니라면*, 이 행동에서 발생하는 미래 상태들은 좋지 않은 보상을 가질 것입니다. 따라서 다음 Q-Learning 목적 함수 업데이트는 가치 $\hat{Q}(s,a)$를 줄일 것이고, 이는 다음번에 로봇이 상태 $s$를 방문할 때 이 행동을 선택할 확률을 줄일 것입니다. 좋지 않은 행동(예컨대 $\hat{Q}(s,a)$에서 가치가 과대평가된 행동)은 로봇이 탐험하지만, 다음 Q-Learning 목적 함수 업데이트에서 그 가치가 교정됩니다. 좋은 행동(예컨대 가치 $\hat{Q}(s, a)$가 큰 행동)은 로봇이 더 자주 탐험하고, 그렇게 함으로써 강화됩니다. 이 속성을 사용해 Q-Learning이 무작위 정책 $\pi_e$로 시작하더라도 최적 정책에 수렴할 수 있다는 것을 보일 수 있습니다 :cite:`Watkins.Dayan.1992`.

새로운 데이터를 수집하는 것뿐 아니라 올바른 종류의 데이터를 수집하는 이 능력은 강화 학습 알고리즘의 핵심 특징이며, 지도 학습과 강화 학습을 구별 짓는 부분입니다. (나중에 DQN 장에서 보게 될) 심층 신경망을 사용한 Q-Learning은 강화 학습의 부활에 결정적인 역할을 했습니다 :cite:`mnih2013playing`.

## Q-Learning의 구현

이제 [Open AI Gym](https://gym.openai.com)의 FrozenLake에 Q-Learning을 어떻게 구현하는지 보여드리겠습니다. 이는 :ref:`sec_valueiter` 실험에서 고려한 것과 동일한 설정임에 유의하세요.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 256  # Number of iterations
alpha   = 0.9  # Learing rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

FrozenLake 환경에서 로봇은 $4 \times 4$ 격자(이것들이 상태입니다) 위를 이동하며, 행동은 "위쪽"($\uparrow$), "아래쪽"($\rightarrow$), "왼쪽"($\leftarrow$), "오른쪽"($\rightarrow$)입니다. 환경은 여러 개의 구멍(H) 칸과 얼어붙은(F) 칸, 그리고 목표(G) 칸을 포함하고 있으며, 이 모두는 로봇에게 알려져 있지 않습니다. 문제를 단순하게 유지하기 위해, 로봇이 신뢰할 수 있는 행동을 가진다고 가정합니다. 즉, 모든 $s \in \mathcal{S}, a \in \mathcal{A}$에 대해 $P(s' \mid s, a) = 1$입니다. 로봇이 목표에 도달하면 시행이 종료되고 로봇은 행동과 무관하게 $1$의 보상을 받습니다. 그 외 다른 상태에서의 보상은 모든 행동에 대해 $0$입니다. 로봇의 목적은 주어진 시작 위치(S)(이것이 $s_0$입니다)에서 목표 위치(G)에 도달하는 정책을 학습하여 *리턴*을 극대화하는 것입니다.

먼저 $\epsilon$-탐욕 방법을 다음과 같이 구현합니다.

```{.python .input}
%%tab all

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

이제 Q-learning을 구현할 준비가 되었습니다.

```{.python .input}
%%tab all

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # Reset environment
        state, done = env.reset(), False
        while not done:
            # Select an action for a given state and acts in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-update:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

이 결과는 Q-learning이 대략 250번의 반복 후에 이 문제에 대한 최적 해를 찾을 수 있음을 보여줍니다. 그러나 이 결과를 Value Iteration 알고리즘의 결과(:ref:`subsec_valueitercode` 참조)와 비교하면, Value Iteration 알고리즘이 이 문제에 대한 최적 해를 찾는 데 훨씬 더 적은 반복이 필요함을 알 수 있습니다. 이는 Value Iteration 알고리즘이 완전한 MDP에 접근할 수 있는 반면, Q-learning은 그렇지 않기 때문에 발생합니다.


## 요약
Q-learning은 가장 기본적인 강화 학습 알고리즘 중 하나입니다. 강화 학습의 최근 성공의 진앙에 있으며, 특히 비디오 게임을 학습으로 플레이하는 것에서 두드러집니다 :cite:`mnih2013playing`. Q-learning을 구현하는 데에는 마르코프 결정 과정(MDP), 예컨대 전이 함수와 보상 함수를 완전히 알아야 할 필요가 없습니다.

## 연습문제

1. 격자의 크기를 $8 \times 8$로 늘려보세요. $4 \times 4$ 격자와 비교했을 때 최적 가치 함수를 찾는 데 몇 번의 반복이 걸리나요?
1. 위 코드의 $\gamma$ ("gamma")가 각각 $0$, $0.5$, $1$일 때 Q-learning 알고리즘을 다시 실행하고 그 결과를 분석해 보세요.
1. 위 코드의 $\epsilon$ ("epsilon")이 각각 $0$, $0.5$, $1$일 때 Q-learning 알고리즘을 다시 실행하고 그 결과를 분석해 보세요.

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12103)
:end_tab:
