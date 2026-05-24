```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Value Iteration
:label:`sec_valueiter`

이 절에서는 궤적의 *리턴*을 극대화하기 위해 각 상태에서 로봇이 취할 최선의 행동을 어떻게 선택할지 논의합니다. Value Iteration이라는 알고리즘을 설명하고, 얼어붙은 호수 위를 이동하는 시뮬레이션 로봇에 대해 구현해 보겠습니다.

## 확률적 정책

확률적 정책(policy), 즉 $\pi(a \mid s)$ (줄여서 정책)는 상태 $s \in \mathcal{S}$가 주어졌을 때 행동 $a \in \mathcal{A}$에 대한 조건부 분포로, $\pi(a \mid s) \equiv P(a \mid s)$입니다. 예를 들어, 로봇이 네 개의 행동 $\mathcal{A}=$ {왼쪽으로 가기, 아래로 가기, 오른쪽으로 가기, 위로 가기}을 가지고 있다고 합시다. 그러한 행동 집합 $\mathcal{A}$에 대해 상태 $s \in \mathcal{S}$에서의 정책은 범주형 분포(categorical distribution)이며, 네 개의 행동에 대한 확률은 $[0.4, 0.2, 0.1, 0.3]$이 될 수도 있습니다. 또 다른 상태 $s' \in \mathcal{S}$에서는 같은 네 개의 행동에 대한 확률 $\pi(a \mid s')$가 $[0.1, 0.1, 0.2, 0.6]$이 될 수도 있습니다. 임의의 상태 $s$에 대해 $\sum_a \pi(a \mid s) = 1$이 성립해야 한다는 점에 유의하세요. 결정론적 정책은 확률적 정책의 특수한 경우로, 분포 $\pi(a \mid s)$가 오직 한 개의 특정 행동에만 0이 아닌 확률을 부여하는 경우입니다. 예를 들어, 네 개의 행동 예제에서는 $[1, 0, 0, 0]$이 그러한 경우입니다.

표기를 더 간결하게 하기 위해, $\pi(a \mid s)$ 대신 $\pi(s)$로 조건부 분포를 표기하는 경우가 많습니다.

## 가치 함수

이제 로봇이 상태 $s_0$에서 시작하여 매 시간 순간마다 먼저 정책으로부터 행동을 표집(sample)해 $a_t \sim \pi(s_t)$로 얻고, 이 행동을 취하여 다음 상태 $s_{t+1}$이 되는 상황을 상상해 봅시다. 궤적 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$는 중간 순간에 특정 행동 $a_t$가 어떻게 표집되는가에 따라 달라질 수 있습니다. 그러한 모든 궤적의 평균 *리턴* $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$를 다음과 같이 정의합니다.
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$

여기서 $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$는 로봇의 다음 상태이고, $r(s_t, a_t)$는 시간 $t$에 상태 $s_t$에서 행동 $a_t$를 취하여 얻은 즉시 보상입니다. 이를 정책 $\pi$에 대한 "가치 함수(value function)"라고 부릅니다. 간단히 말해, 정책 $\pi$에 대한 상태 $s_0$의 가치 $V^\pi(s_0)$는 로봇이 상태 $s_0$에서 시작하여 매 시간 순간마다 정책 $\pi$에서 행동을 취할 때 얻을 것으로 기대되는 $\gamma$-할인된 *리턴*입니다.

다음으로 궤적을 두 단계로 분해합니다. (i) 행동 $a_0$을 취해 $s_0 \to s_1$이 되는 첫 번째 단계, 그리고 (ii) 그 이후의 궤적 $\tau' = (s_1, a_1, r_1, \ldots)$에 해당하는 두 번째 단계입니다. 강화 학습의 모든 알고리즘의 핵심 아이디어는, 상태 $s_0$의 가치를 첫 번째 단계에서 얻은 평균 보상과 가능한 모든 다음 상태 $s_1$에 대해 평균낸 가치 함수의 합으로 쓸 수 있다는 점입니다. 이는 매우 직관적이며 우리의 마르코프 가정에서 비롯됩니다. 현재 상태에서의 평균 리턴은 다음 상태에서의 평균 리턴과 다음 상태로 가는 평균 보상의 합입니다. 수학적으로 두 단계를 다음과 같이 씁니다.

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$
:eqlabel:`eq_dynamic_programming`

이 분해는 매우 강력합니다. 모든 강화 학습 알고리즘의 기초가 되는 동적 프로그래밍(dynamic programming) 원리의 토대이기 때문입니다. 두 번째 단계에서 두 개의 기댓값이 나타나는 점에 주목하세요. 하나는 확률적 정책을 사용해 첫 번째 단계에서 취하는 행동 $a_0$의 선택에 대한 것이고, 다른 하나는 선택된 행동에서 얻어지는 가능한 상태 $s_1$에 대한 것입니다. :eqref:`eq_dynamic_programming`를 마르코프 결정 과정(MDP)의 전이 확률을 사용해 다음과 같이 쓸 수 있습니다.

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$
:eqlabel:`eq_dynamic_programming_val`

여기서 주목해야 할 중요한 점은, 위 항등식이 모든 상태 $s \in \mathcal{S}$에 대해 성립한다는 것입니다. 그 상태에서 시작하는 임의의 궤적을 생각하고 그 궤적을 두 단계로 분해할 수 있기 때문입니다.

## 행동-가치 함수

구현 시에는 가치 함수와 밀접하게 관련된 "행동 가치(action value)" 함수라는 양을 유지하는 것이 종종 유용합니다. 이는 $s_0$에서 시작하는 궤적의 평균 *리턴*으로 정의되되, 첫 번째 단계의 행동이 $a_0$으로 고정된 경우의 평균 리턴입니다.

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

이 경우 첫 번째 단계의 보상이 고정되어 있기 때문에 기댓값 안의 합이 $t=1,\ldots, \infty$ 범위라는 점에 유의하세요. 다시 궤적을 두 부분으로 나누어 다음과 같이 쓸 수 있습니다.

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$
:eqlabel:`eq_dynamic_programming_q`

이 버전은 행동 가치 함수에 대한 :eqref:`eq_dynamic_programming_val`의 대응에 해당합니다.

## 최적 확률적 정책

가치 함수와 행동-가치 함수 모두 로봇이 선택하는 정책에 의존합니다. 다음으로 최대 평균 *리턴*을 달성하는 "최적 정책(optimal policy)"을 생각해 보겠습니다.
$$\pi^* = \underset{\pi}{\mathrm{argmax}} V^\pi(s_0).$$

로봇이 취할 수 있었던 모든 가능한 확률적 정책 중에서, 최적 정책 $\pi^*$는 상태 $s_0$에서 시작하는 궤적에 대해 가장 큰 평균 할인 *리턴*을 달성합니다. 최적 정책의 가치 함수와 행동-가치 함수를 $V^* \equiv V^{\pi^*}$, $Q^* \equiv Q^{\pi^*}$로 표기합시다.

임의의 주어진 상태에서 정책 하에 가능한 행동이 단 하나뿐인 결정론적 정책의 경우를 살펴봅시다. 이때 다음을 얻습니다.

$$\pi^*(s) = \underset{a \in \mathcal{A}}{\mathrm{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big].$$

이를 기억하기 좋은 방법은, 상태 $s$에서의 (결정론적 정책에 대한) 최적 행동이 첫 번째 단계의 보상 $r(s, a)$와, 두 번째 단계의 가능한 모든 다음 상태 $s'$에 대해 평균낸, 다음 상태 $s'$에서 시작하는 궤적의 평균 *리턴*의 합을 최대화하는 행동이라는 점입니다.

## 동적 프로그래밍의 원리

이전 절의 :eqref:`eq_dynamic_programming` 또는 :eqref:`eq_dynamic_programming_q`의 전개를 알고리즘으로 바꾸어 최적 가치 함수 $V^*$ 또는 행동-가치 함수 $Q^*$를 각각 계산할 수 있습니다. 다음을 살펴봅시다.
$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi^*(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

결정론적 최적 정책 $\pi^*$의 경우, 상태 $s$에서 취할 수 있는 행동이 단 하나뿐이므로 다음과 같이 쓸 수도 있습니다.

$$V^*(s) = \mathrm{argmax}_{a \in \mathcal{A}} \Big\{ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big\}$$

모든 상태 $s \in \mathcal{S}$에 대해 성립합니다. 이 항등식을 "동적 프로그래밍의 원리(principle of dynamic programming)"라 합니다 :cite:`BellmanDPPaper,BellmanDPBook`. 1950년대에 Richard Bellman이 정식화한 것으로, "최적 궤적의 나머지 부분도 최적이다"라고 기억할 수 있습니다.

## Value Iteration

동적 프로그래밍의 원리를 최적 가치 함수를 찾는 알고리즘인 value iteration으로 변환할 수 있습니다. value iteration의 핵심 아이디어는 이 항등식을 서로 다른 상태 $s \in \mathcal{S}$에서의 $V^*(s)$를 함께 묶는 제약 조건의 집합으로 생각하는 것입니다. 모든 상태 $s \in \mathcal{S}$에 대해 가치 함수를 임의의 값 $V_0(s)$로 초기화합니다. $k^{\textrm{th}}$ 반복에서 Value Iteration 알고리즘은 가치 함수를 다음과 같이 업데이트합니다.

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \Big\{ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big\};\ \textrm{for all } s \in \mathcal{S}.$$

$k \to \infty$일 때 Value Iteration 알고리즘으로 추정된 가치 함수는 초기화 $V_0$과 무관하게 최적 가치 함수에 수렴한다는 사실이 알려져 있습니다.
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

동일한 Value Iteration 알고리즘은 행동-가치 함수를 사용해 동치적으로 다음과 같이 쓸 수 있습니다.
$$Q_{k+1}(s, a) = r(s, a) + \gamma \max_{a' \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) Q_k (s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

이 경우 모든 $s \in \mathcal{S}$와 $a \in \mathcal{A}$에 대해 $Q_0(s, a)$를 임의의 값으로 초기화합니다. 마찬가지로 모든 $s \in \mathcal{S}$와 $a \in \mathcal{A}$에 대해 $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$가 성립합니다.

## 정책 평가

Value Iteration을 통해 최적 가치 함수, 즉 최적 결정론적 정책 $\pi^*$의 $V^{\pi^*}$를 계산할 수 있습니다. 유사한 반복적 업데이트를 사용해, 잠재적으로 확률적일 수 있는 임의의 다른 정책 $\pi$에 연관된 가치 함수도 계산할 수 있습니다. 모든 상태 $s \in \mathcal{S}$에 대해 $V^\pi_0(s)$를 임의의 값으로 다시 초기화하고, $k^{\textrm{th}}$ 반복에서 다음과 같은 업데이트를 수행합니다.

$$    V^\pi_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi_k(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

이 알고리즘은 정책 평가(policy evaluation)로 알려져 있으며, 주어진 정책의 가치 함수를 계산하는 데 유용합니다. 마찬가지로, $k \to \infty$일 때 이 업데이트는 초기화 $V_0$과 무관하게 올바른 가치 함수에 수렴한다는 것이 알려져 있습니다.

$$V^\pi(s) = \lim_{k \to \infty} V^\pi_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

정책 $\pi$의 행동-가치 함수 $Q^\pi(s, a)$를 계산하는 알고리즘도 유사합니다.

## Value Iteration의 구현
:label:`subsec_valueitercode`
다음으로 [Open AI Gym](https://gym.openai.com)의 FrozenLake라는 탐색 문제에 대해 Value Iteration을 어떻게 구현하는지 보여드리겠습니다. 먼저 다음 코드와 같이 환경을 설정해야 합니다.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 10  # Number of iterations
random.seed(seed)  # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

FrozenLake 환경에서 로봇은 $4 \times 4$ 격자(이것들이 상태입니다) 위를 이동하며, 행동은 "위쪽"($\uparrow$), "아래쪽"($\rightarrow$), "왼쪽"($\leftarrow$), "오른쪽"($\rightarrow$)입니다. 환경은 여러 개의 구멍(H) 칸과 얼어붙은(F) 칸, 그리고 목표(G) 칸을 포함하고 있으며, 이 모두는 로봇에게 알려져 있지 않습니다. 문제를 단순하게 유지하기 위해, 로봇이 신뢰할 수 있는 행동을 가진다고 가정합니다. 즉, 모든 $s \in \mathcal{S}, a \in \mathcal{A}$에 대해 $P(s' \mid s, a) = 1$입니다. 로봇이 목표에 도달하면 시행이 종료되고 로봇은 행동과 무관하게 $1$의 보상을 받습니다. 그 외 다른 상태에서의 보상은 모든 행동에 대해 $0$입니다. 로봇의 목적은 주어진 시작 위치(S)(이것이 $s_0$입니다)에서 목표 위치(G)에 도달하는 정책을 학습하여 *리턴*을 극대화하는 것입니다.

다음 함수는 Value Iteration을 구현합니다. 여기서 `env_info`는 MDP 및 환경 관련 정보를 담고 있고, `gamma`는 할인 인자입니다.

```{.python .input}
%%tab all

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D array shows what each item means
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Next state
                    reward = pxrds[reward_idx]  # Reward
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

위 그림들은 정책(화살표가 행동을 나타냄)과 가치 함수(색상의 변화는 어두운 색으로 표시된 초기 값에서 밝은 색으로 표시된 최적 값으로 가치 함수가 시간에 따라 어떻게 변하는지를 보여줍니다)를 보여줍니다. 보시는 것처럼, Value Iteration은 10번의 반복 후에 최적 가치 함수를 찾고, H 칸이 아닌 한 어떤 상태에서 시작하더라도 목표 상태(G)에 도달할 수 있습니다. 구현의 또 다른 흥미로운 측면은, 최적 가치 함수를 찾는 것 외에 이 가치 함수에 대응하는 최적 정책 $\pi^*$도 자동으로 찾았다는 점입니다.


## 요약
Value Iteration 알고리즘의 주된 아이디어는 동적 프로그래밍의 원리를 사용하여 주어진 상태에서 얻을 수 있는 최적 평균 리턴을 찾는 것입니다. Value Iteration 알고리즘을 구현하려면 마르코프 결정 과정(MDP), 예를 들어 전이 함수와 보상 함수를 완전히 알아야 한다는 점에 유의하세요.


## 연습문제

1. 격자의 크기를 $8 \times 8$로 늘려보세요. $4 \times 4$ 격자와 비교했을 때 최적 가치 함수를 찾는 데 몇 번의 반복이 걸리나요?
1. Value Iteration 알고리즘의 계산 복잡도는 무엇인가요?
1. 위 코드의 $\gamma$ ("gamma")가 각각 $0$, $0.5$, $1$일 때 Value Iteration 알고리즘을 다시 실행하고 그 결과를 분석해 보세요.
1. $\gamma$의 값은 Value Iteration이 수렴하는 데 걸리는 반복 횟수에 어떻게 영향을 미치나요? $\gamma=1$일 때는 어떤 일이 일어나나요?

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12005)
:end_tab:
