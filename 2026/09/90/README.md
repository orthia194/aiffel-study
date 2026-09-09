# Transformer

---

## 1. Overall Structure (전체 구조)

* **Encoder:** 입력 문장(Source)의 전체 문맥을 이해하고 고차원 특징 표현(Feature Vectors)으로 압축합니다.
* **Decoder:** 인코더의 정보와 이전에 자신이 생성한 단어들을 바탕으로 타겟 문장(Target)을 순차적으로 생성합니다.

---

## 2. Key Components & Roles (주요 용어 및 역할)

### Input & Output Processing
| 용어 | 역할 |
| :--- | :--- |
| **Input / Output Embedding** | 단어(토큰 ID)를 고차원 연속 벡터(Continuous Vector)로 변환합니다. |
| **Positional Encoding** | 순차 처리(RNN)를 하지 않는 트랜스포머를 위해, 단어의 위치(순서) 정보를 사인/코사인 주기함수 값으로 더해줍니다. |
| **Outputs (shifted right)** | 디코더 입력 시 타겟 문장의 시작 토큰(`<BOS>`)을 맨 앞에 두고 오른쪽으로 한 칸씩 밀어넣어 학습을 진행합니다. |

---

### Attention Layers
| 용어 | 역할 |
| :--- | :--- |
| **Multi-Head Attention (Self-Attention)** | 문장 내부 단어들 간의 연관성($Q, K, V$)을 여러 시선(Head)에서 동시에 분석하여 가중치를 계산합니다. |
| **Masked Multi-Head Attention** | 디코더에서 사용하며, 추론 시점 이후의 미래 단어(Future Tokens)를 미리 보지 못하도록 마스킹 처리하여 정답 유출을 막습니다. |
| **Multi-Head Attention (Encoder-Decoder Cross)** | 디코더 중단에 위치하며, **$Q$는 디코더**에서, **$K, V$는 인코더의 출력**에서 가져와 원문 문맥과 번역문의 관계를 연결합니다. |

---

### Feed-Forward & Residual Connections
| 용어 | 역할 |
| :--- | :--- |
| **Feed Forward (FFN)** | 각 토큰 벡터를 독립적으로 두 개의 Linear 레이어와 활성화 함수(ReLU/GELU)에 통과시켜 비선형 특징을 학습합니다. |
| **Add & Norm (Residual Connection + LayerNorm)** | **Add:** 입력값을 출력에 그대로 더해주는 잔차 연결로 기울기 소실(Gradient Vanishing)을 방지합니다.<br>**Norm:** 레이어 정규화(Layer Normalization)를 적용하여 학습을 안정화합니다. |

---

### Final Output
| 용어 | 역할 |
| :--- | :--- |
| **Linear Layer** | 디코더의 최종 벡터를 전체 단어장 크기(Vocabulary Size) 차원으로 차원 확장합니다. |
| **Softmax** | 각 단어별 등장 확률(Output Probabilities)을 0~1 사이 값(총합 1)으로 변환하여 가장 높은 확률의 단어를 최종 예측합니다. |

---

## Data Flow Summary (데이터 흐름 요약)

1. **[입력]** `Inputs` $\rightarrow$ `Embedding` + `Positional Encoding`
2. **[인코더]** `Self-Attention` $\rightarrow$ `Add & Norm` $\rightarrow$ `FFN` $\rightarrow$ `Add & Norm` (인코더 출력 생성)
3. **[디코더]** `Masked Self-Attention` $\rightarrow$ `Add & Norm` $\rightarrow$ `Encoder-Decoder Cross Attention` $\rightarrow$ `FFN` $\rightarrow$ `Add & Norm`
4. **[출력]** `Linear` $\rightarrow$ `Softmax` $\rightarrow$ `다음 단어 예측`

---

### Input & Output Processing

| 용어 | 역할 및 특징 |
| :--- | :--- |
| **Input / Output Embedding** | 단어(토큰 ID)를 고차원 연속 벡터(Continuous Vector)로 변환합니다. |
| **Positional Encoding** | **[순서 정보 부여]** RNN과 달리 모든 단어를 한 번에 병렬 처리하는 트랜스포머의 특성상 단어의 위치/순서를 알 수 없는 한계를 극복하기 위해, 위치 정보를 가진 벡터를 단어 임베딩에 더해주는($+$) 역할을 합니다. |
| **Outputs (shifted right)** | 디코더 입력 시 타겟 문장의 시작 토큰(`<BOS>`)을 맨 앞에 두고 오른쪽으로 한 칸씩 밀어넣어 학습을 진행합니다. |

---

#### 💡 Deep Dive: Positional Encoding vs Positional Embedding

* **왜 필수적인가?**
  * 순서 정보가 없다면 `"나는 사과를 먹었다"`와 `"사과가 나를 먹었다"`를 완전히 동일한 단어 집합으로 인식하는 치명적 단점이 발생합니다.
* **동작 방식:**
  * $\text{[최종 입력 벡터]} = \text{[단어 의미 벡터 (Embedding)]} + \text{[위치 정보 벡터 (Positional)]}$
* **Positional Encoding (원조 Transformer 방식):**
  * 삼각함수(사인 $\sin$, 코사인 $\cos$) 주기함수 공식을 사용하여 고정된 위치 값을 계산해 더해줍니다.
  * 별도의 학습 파라미터가 필요 없어 문장 길이가 길어져도 가변적으로 위치를 계산할 수 있습니다.
* **Positional Embedding (BERT 등 최신 모델 방식):**
  * 위치 정보 자체도 학습 가능한 파라미터(Weight)로 만들어 모델이 학습 데이터로부터 위치 표현을 직접 학습하도록 구성합니다.

---
 ## Scaled Dot-Product Attention

* **개념:** $Q$(Query) 단어가 문장 내 다른 단어들($K$, Key)과 얼마나 깊은 관련이 있는지를 계산하고, 그 유사도 비율만큼 정보($V$, Value)를 가져와 풍부한 맥락 벡터를 만드는 트랜스포머의 핵심 연산 엔진입니다.
* **예시:** `"나는 선생이고..."` 문장에서 주어 `"나는"`($Q$)이 보어 `"선생이고"`($K$)와 65%의 높은 관련성을 가짐을 파악하여, `"나는"` 벡터에 `"선생"`이라는 정체성 맥락($V$)을 주입합니다.

---

### ⚙️ Step-by-Step Data Flow (1~5 단계)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

#### 1. MatMul (첫 번째 행렬 곱: $QK^T$)
* **역할:** $Q$(Query)와 $K$(Key)를 행렬 곱하여 각 단어 쌍 간의 관련성 점수(Attention Score)를 구합니다.
* **의미:** *"단어 A가 단어 B와 얼마나 관계가 깊은가?"*를 측정하는 유사도 계산 단계입니다.

#### 2. Scale (스케일링: $\div \sqrt{d_k}$)
* **역할:** 점수를 차원의 크기($d_k$)의 제곱근으로 나눠줍니다.
* **의미:** 차원이 커질수록 $QK^T$ 연산 결과값이 극도로 커지는 현상을 막아, Softmax 단계에서 기울기 소실(Gradient Vanishing)이 발생하는 것을 방지합니다.

#### 3. Mask (opt.) (마스킹)
* **역할:** 선택적(Optional) 단계로, 불필요하거나 접근해서는 안 되는 위치의 점수를 `-1e9`(매우 작은 음수)로 가립니다.
* **주요 용도:**
  * **Padding Mask:** 문장 길이를 맞추기 위한 의미 없는 `PAD` 토큰 무시
  * **Look-ahead Mask (디코더):** 문장 생성 시 미래의 정답 단어를 미리 훔쳐보지 못하도록 차단

#### 4. SoftMax (소프트맥스)
* **역할:** 마스킹된 점수들을 **합이 1(100%)이 되는 확률값(가중치 비율)**으로 변환합니다.
* **의미:** 큰 값은 더 강조하고 작은 값은 낮추어, 어떤 단어의 정보를 얼마만큼의 비율(예: 65%, 20%, 15%)로 가져올지 가중치를 정합니다.

#### 5. MatMul (두 번째 행렬 곱: $\times V$)
* **역할:** Softmax로 계산된 가중치 비율과 진짜 정보가 담긴 $V$(Value)를 행렬 곱합니다.
* **의미:** 각 단어의 의미 정보($V$)를 관련성 비율대로 가중합(Weighted Sum)하여, 주변 문맥 정보가 깊게 반영된 **최종 어텐션 벡터**를 완성합니다.

---
### Causality Masking (Look-ahead Masking)

* **개념:** 디코더(Decoder)가 문장을 생성할 때, 현재 위치보다 **미래에 나올 정답 단어를 미리 훔쳐보지 못하도록 미래 토큰들의 어텐션 점수를 가리는(Masking) 기법**입니다.
* **필요성:** 
  * 기존 RNN은 순차적으로(Autoregressive) 단어를 처리하지만, 트랜스포머는 전체 문장을 **병렬(Parallel) 처리**합니다.
  * 병렬 처리 특성상 아무런 제약이 없다면 모델이 미래 단어(정답)를 참조해 버리는 부작용이 발생하므로, 인위적으로 시점의 연속성과 인과 관계를 보장해 주기 위해 필수적입니다.

---

#### ⚙️ 동작 원리 (How it works)

1. **하삼각 행렬 (Lower Triangular Matrix) 생성**
   * 현재 시점 기준 **자신과 과거 위치는 `1`(허용)**, **미래 위치는 `0`(차단)**으로 구성된 마스크 행렬을 생성합니다.
   
2. **어텐션 점수 차단 (Mask Application)**
   * $QK^T$ 연산 결과에서 차단해야 할 미래 위치에 **`-1e9`(음수 무한대)** 값을 더해줍니다.
   
3. **Softmax를 통한 확률 0%화**
   * Softmax 연산을 거치면 `-1e9`가 가해진 미래 단어들의 가중치는 **`0` (0%)**이 되어, 어텐션 정보($V$)를 가져올 때 완전히 배제됩니다.

---

#### 📊 데이터 흐름 시각화 (4개 단어 예시)

| 시점 (Query) | 참조 가능 단어 (Key) | 마스킹 처리 |
| :--- | :--- | :--- |
| **1번째 단어** | `[ 1번째 단어,   가림,     가림,     가림   ]` | 오직 첫 단어만 보고 다음 단어 예측 |
| **2번째 단어** | `[ 1번째 단어, 2번째 단어,   가림,     가림   ]` | 1~2번째 단어만 보고 3번째 단어 예측 |
| **3번째 단어** | `[ 1번째 단어, 2번째 단어, 3번째 단어,   가림   ]` | 1~3번째 단어만 보고 4번째 단어 예측 |
| **4번째 단어** | `[ 1번째 단어, 2번째 단어, 3번째 단어, 4번째 단어 ]` | 전체 단어를 참고하여 문맥 완성 |

---
## Position-wise Feed-Forward Networks (FFN)

* **개념:** Attention Layer를 통과하며 수집된 문맥 정보(Context)를 바탕으로, **각 단어 벡터의 표현력(Feature Representation)을 복잡하고 비선형적인 공간으로 확장 및 재구성하는 2층 신경망(MLP)**입니다.
* **특징 (Position-wise):** 
  * 문장 전체를 묶어서 연산하는 Attention과 달리, 문장 내 **각 위치(Position)의 단어 벡터들에 완전히 독립적이면서 동일한 가중치(Shared Weights)로 적용**됩니다.
  * Attention이 *"단어 간 상호작용(대화)"*이라면, FFN은 *"각 단어가 얻은 정보를 스스로 재정리하는 과정"*입니다.

---

### ⚙️ 동작 원리 및 수식 (How it works)

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

> 최근 LLM(LLaMA 등) 구현체에서는 $\text{ReLU}(\max(0, \cdot))$ 대신 **GELU** 또는 **SwiGLU** 활성화 함수를 주로 사용합니다.

1. **차원 확장 (Linear 1: Expansion)**
   * 입력 벡터 $x$ ($d_{\text{model}} = 512$)에 첫 번째 가중치 행렬 $W_1$을 곱해 차원을 **4배 크기($d_{ff} = 2048$)로 뻥튀기**합니다.
   * 차원을 크게 넓혀 복잡하고 다양한 특징(Feature)들을 수용할 수 있는 고차원 공간을 확보합니다.

2. **비선형 활성화 (Activation Function)**
   * 확장된 고차원 공간에 **ReLU** 등의 활성화 함수를 적용하여 비선형성(Non-linearity)을 추가합니다.
   * 단순 선형 결합으로는 학습할 수 없는 복잡한 패턴을 파악할 수 있게 만듭니다.

3. **차원 축소 (Linear 2: Compression)**
   * 두 번째 가중치 행렬 $W_2$를 곱해 다시 **원래 차원($d_{\text{model}} = 512$)으로 압축**합니다.
   * 다음 Residual Connection(Add & Norm) 연산을 위해 입력값과 동일한 모양(Shape)으로 맞춰줍니다.

---

### 📊 입출력 차원 변화 요약

| 단계 | 연산 | 차원 (Shape) |
| :--- | :--- | :--- |
| **입력 (Input)** | Attention 및 Add&Norm을 거친 벡터 | $(batch\_size, seq\_len, 512)$ |
| **Step 1** | $xW_1 + b_1$ (차원 4배 확장) | $(batch\_size, seq\_len, 2048)$ |
| **Step 2** | $\text{ReLU}(\cdot)$ (비선형 활성화) | $(batch\_size, seq\_len, 2048)$ |
| **Step 3** | $(\cdot)W_2 + b_2$ (원래 차원 축소) | $(batch\_size, seq\_len, 512)$ |

---
## Add & Norm (Residual Connection & Layer Normalization)

* **개념:** 트랜스포머의 모든 서브레이어(Multi-Head Attention, FFN) 뒤에 위치하여, **깊은 신경망 구조에서도 기울기 소실 없이 안정적이고 빠르게 학습되도록 돕는 보호막 장치**입니다.
* **수식:** 
  $$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

---

### ⚙️ 구성 요소별 동작 원리

#### 1. Add (Residual Connection / 잔차 연결)
* **연산:** $\text{Input}(x) + \text{SubLayerOutput}(F(x))$
* **역할:** 
  * 입력값 $x$를 고속도로(Shortcut/Skip Connection)를 통해 출력단에 그대로 더해줍니다.
  * 역전파 시 기울기(Gradient)가 가중치 연산을 거치지 않고 최소 $1$ 이상 보장되어, **레이어가 아무리 깊어져도 기울기 소실(Gradient Vanishing)이 발생하지 않습니다.**

#### 2. Norm (Layer Normalization / 레이어 정규화)
* **연산:** 각 토큰 벡터의 요소(Feature) 차원에 대해 평균($\mu$)과 분산($\sigma^2$)을 구하여 정규화합니다.
  $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
* **역할:** 
  * 각 단어 벡터 내부의 값 분포를 평균 $0$, 표준편차 $1$로 맞춰 주어 연산의 폭주를 막습니다.
  * 학습 속도를 크게 향상시키고 internal covariate shift(내부 공변량 변화) 현상을 완화합니다.

### 💡 Deep Dive: Layer Normalization (LN)

* **개념:** 단일 토큰 벡터 내부의 Feature(특징) 차원을 기준으로 평균을 0, 분산을 1로 정규화하는 기법입니다.
* **왜 Transformer는 BatchNorm 대신 LayerNorm을 쓸까?**
  * **Batch-independent:** 문장 생성(LLM 추론) 시에는 배치 크기가 1이 되는 경우가 많은데, LayerNorm은 배치 크기에 영향을 받지 않고 독자적으로 정규화를 수행할 수 있습니다.
  * **Variable Length:** 입력 문장의 길이가 제각각이어도 안정적인 정규화가 가능합니다.
    
* 💡 **왜 BatchNorm 대신 LayerNorm을 사용하는가?**
  * **Batch-independent:** `BatchNorm`은 여러 데이터(배치) 축으로 정규화하므로 배치 크기가 작거나 $1$일 때(LLM 실시간 추론 시) 분산이 $0$이 되어 에러가 발생합니다.
  * `LayerNorm`은 단어 벡터 **내부($d_{\text{model}}$ 차원)** 수치들만 이용해 정규화하므로, 단어 1개($\text{Batch Size}=1$)만 들어와도 완벽하게 동작합니다.
---

#### ⚙️ 계산 과정 (Vector-level)

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$\text{Output} = \gamma \odot \hat{x} + \beta$$

* $\mu, \sigma^2$: 단일 토큰 벡터의 Feature 차원에 대한 평균과 분산
* $\gamma, \beta$: 모델이 학습을 통해 최적화하는 파라미터 (Scale & Shift)
* $\epsilon$: Divide-by-zero 방지용 소수 ($1e-5$)
---

### 💡Residual Connection (Skip Connection)

* **개념:** 입력 신호 $x$를 두 갈래(Path)로 나누어 전달함으로써, 레이어가 아무리 깊어져도 원본 정보 손실과 기울기 소실(Gradient Vanishing)을 방지하는 메커니즘입니다.

---

#### ⚙️ 데이터의 2가지 전달 경로 (Two-Path Signal Flow)
1. **계산하는 길 (SubLayer Path):**
   * 입력 $x$가 어텐션(Attention)이나 FFN 레이어를 지나며 복잡한 연산($F(x)$)을 수행합니다.
   * *"새로운 맥락과 특징(Feature)을 추출하는 경로"*

2. **건네주는 길 (Skip / Residual Path):**
   * 입력 $x$가 아무런 가중치 연산 없이 고속도로처럼 그대로 출력단까지 넘어갑니다.
   * *"기존 정보와 원본 신호를 손실 없이 보존하여 전달하는 경로"*

---

#### 📊 왜 두 개로 나누어 전달하는가? (핵심 이점)

* **순전파 (Forward Pass):** 모델이 $x$ 전체를 새로 만들어낼 필요 없이, 변화량(잔차, $F(x) = y - x$)만 집중해서 학습하므로 학습 안정성과 속도가 비약적으로 상승합니다.
* **역전파 (Backward Pass):** 미분 시 $1 + F'(x)$ 형태가 되어, "건네주는 길"을 타고 기울기가 최단거리($1$)로 역전파됩니다. 이로 인해 레이어가 수십~수백 개로 깊어져도 기울기가 $0$으로 소실되지 않습니다.

---

#### 🔍 용어 정리 (Residual Connection vs Skip Connection)

* **Skip Connection:** 신호가 중간 연산을 건너뛰는(Skip) **구조적/형태적 관점**의 명칭
* **Residual Connection:** 모델이 입력과의 차이인 잔차(Residual)만 학습한다는 **수학적/학습적 관점**의 명칭 (두 용어는 완전히 동일한 구현 기법을 가리킴)
---

### 💡 Deep Dive: Optimizer & Learning Rate Scheduler

#### 1. Optimizer: 왜 AdamW가 표준인가?
* **AdamW (Adam with Decoupled Weight Decay):** 기본 Adam의 Weight Decay 연산 오류를 보완한 버전으로, 트랜스포머 학습 시 가장 널리 쓰이는 표준 옵티마이저입니다.
* **메모리 효율화 버전:** 거대 언어 모델(LLM) 학습 시 GPU 메모리 절약을 위해 **Adafactor**나 **8-bit Adam (bitsandbytes)**이 활용됩니다.

---

#### 2. Learning Rate Scheduler (Warmup Strategy)
트랜스포머는 초기 가중치 발산 방지 및 안정적인 수렴을 위해 **Warmup + Decay** 스케줄러를 필수적으로 결합하여 사용합니다.

* **Warmup Phase (초반):** 지정한 `warmup_steps`까지 학습률을 $0$에서 선형적으로 크게 올려 가중치가 급격히 흔들리는 것을 방지합니다.
* **Decay Phase (후반):** 스텝이 진행됨에 따라 학습률을 서서히 줄여 손실 함수의 최적점(Global Minimum)에 정밀하게 수렴하도록 유도합니다.
---
### 💡 Deep Dive: Weight Sharing (Tied Embeddings)

* **개념:** 입력 단어 임베딩(Embedding) 레이어와 출력층의 선형(Linear) 레이어가 동일한 가중치 행렬($W$)을 전치(Transpose)하여 공유하는 테크닉입니다.

$$\text{Embedding Layer}: W \in \mathbb{R}^{V \times d_{\text{model}}}$$
$$\text{Output Linear Layer}: W^T \in \mathbb{R}^{d_{\text{model}} \times V}$$

---

#### ⚙️ 주요 특징 및 이점

1. **파라미터 절감:** 단어장(Vocabulary) 크기가 클수록 수천만 개의 파라미터를 아낄 수 있어 메모리 효율이 급증합니다.
2. **과적합 방지:** 입력과 출력 공간을 통일시켜 강력한 정규화(Regularization) 효과를 제공합니다.
3. **Scaling Factor ($\sqrt{d_{\text{model}}}$):** 
   * 공유된 가중치는 수치 크기(Scale)가 작아지는 경향이 있습니다.
   * 이에 따라 임베딩 벡터에 $\sqrt{d_{\text{model}}}$을 곱해 스케일을 키워줌으로써, 뒤이어 더해지는 Positional Encoding 신호에 단어 고유의 의미 정보가 파묻히는 것을 방지합니다.
