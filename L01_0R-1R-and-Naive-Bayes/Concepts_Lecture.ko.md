# 강의 01 — 0-R, 1-R, Naive Bayes 분류

> **최종 수정일:** 2026-10-08
>
> Data Mining: Practical Machine Learning Tools and Techniques, Witten and Frank - Ch 4

> **학습 목표**:
> 1. Weather 데이터셋에서 instance, feature, feature value, class를 직접 지목할 수 있다
> 2. 0-R 분류기가 baseline으로 쓰이는 이유를 설명하고 그 정확도를 계산할 수 있다
> 3. 1-R 알고리즘으로 attribute별 rule과 error를 계산하고, Outlook의 total error가 4/14인 이유를 설명할 수 있다
> 4. 수치형 feature를 discretization하는 방법과, 구간을 너무 잘게 나누면 training error는 줄고 overfitting은 커지는 이유를 설명할 수 있다
> 5. Bayes 정리에서 prior, likelihood, posterior를 구분할 수 있다
> 6. Naive Bayes로 nominal feature와 numeric feature를 가진 새 instance를 분류하고, P(feature | class)를 곱하는 이유를 설명할 수 있다

---

## 목차

- [1. Classification 문제와 Weather 데이터셋](#1-classification-문제와-weather-데이터셋)
  - [1.1 출발 질문](#11-출발-질문)
  - [1.2 데이터셋의 모양과 용어](#12-데이터셋의-모양과-용어)
- [2. 0-R: feature를 전혀 보지 않는 분류](#2-0-r-feature를-전혀-보지-않는-분류)
- [3. 1-R: feature 하나만 사용하는 분류](#3-1-r-feature-하나만-사용하는-분류)
  - [3.1 알고리즘](#31-알고리즘)
  - [3.2 Weather 데이터셋에 적용하기](#32-weather-데이터셋에-적용하기)
- [4. 1-R과 수치형 feature](#4-1-r과-수치형-feature)
  - [4.1 Discretization](#41-discretization)
  - [4.2 구간을 정하는 방법](#42-구간을-정하는-방법)
  - [4.3 Overfitting이 문제인 이유](#43-overfitting이-문제인-이유)
- [5. Bayes 정리](#5-bayes-정리)
  - [5.1 정리와 용어](#51-정리와-용어)
  - [5.2 예: meningitis와 stiff neck](#52-예-meningitis와-stiff-neck)
- [6. Naive Bayesian Classifier](#6-naive-bayesian-classifier)
  - [6.1 Naive 가정](#61-naive-가정)
  - [6.2 Weather 데이터의 count와 확률](#62-weather-데이터의-count와-확률)
- [7. Weather 예제: 새 instance 분류](#7-weather-예제-새-instance-분류)
- [8. Promotion 데이터 예제](#8-promotion-데이터-예제)
- [9. Naive Bayes와 수치형 feature](#9-naive-bayes와-수치형-feature)
  - [9.1 Gaussian 확률 밀도](#91-gaussian-확률-밀도)
  - [9.2 Numeric weather 데이터](#92-numeric-weather-데이터)
  - [9.3 새 instance 분류](#93-새-instance-분류)
  - [9.4 Naive Bayes의 가정과 장단점](#94-naive-bayes의-가정과-장단점)
- [10. Spam email 분류 예제](#10-spam-email-분류-예제)
- [11. 정리: feature를 어떻게 사용하는가](#11-정리-feature를-어떻게-사용하는가)
- [요약](#요약)
- [점검 문제](#점검-문제)

---

<br>

## 1. Classification 문제와 Weather 데이터셋

### 1.1 출발 질문

**Classification(분류)** 은 데이터에서 규칙을 찾아 새 instance의 class를 정하는 일이다. 다음 질문에서 출발하자.

> **출발 질문:** 아래와 같은 training dataset이 있다. 새로운 날(new instance)이 들어왔을 때 Play의 class를 Yes 또는 No 중 어떻게 판별할 것인가?

**Weather dataset**

| Outlook | Temperature | Humidity | Windy | Play (class) |
|:--------|:------------|:---------|:------|:-------------|
| sunny | hot | high | false | no |
| sunny | hot | high | true | no |
| overcast | hot | high | false | yes |
| rainy | mild | high | false | yes |
| rainy | cool | normal | false | yes |
| rainy | cool | normal | true | no |
| overcast | cool | normal | true | yes |
| sunny | mild | high | false | no |
| sunny | cool | normal | false | yes |
| rainy | mild | normal | false | yes |
| sunny | mild | normal | true | yes |
| overcast | mild | high | true | yes |
| overcast | hot | normal | false | yes |
| rainy | mild | high | true | no |

14일 중 Play = yes는 9일, Play = no는 5일이다. 이 장의 0-R, 1-R, Naive Bayes는 모두 이 표를 training data로 삼아 설명한다.

### 1.2 데이터셋의 모양과 용어

| 용어 | 이 데이터에서의 의미 |
|:-----|:---------------------|
| Instance / Example | 한 행(row): 하루의 관측값 하나 |
| Feature / Attribute | Outlook, Temperature, Humidity, Windy |
| Feature value | sunny, hot, high, false 등 |
| Class / Target | Play = yes 또는 no |
| Classifier | feature들을 보고 새 instance의 class를 결정하는 규칙 또는 모델 |

이 장에서는 분류기를 **feature 정보를 얼마나, 어떤 방식으로 사용하는가** 에 따라 세 단계로 나누어 살펴본다. 0-R은 feature를 하나도 쓰지 않고, 1-R은 하나만 쓰며, Naive Bayes는 모두 쓴다.

---

<br>

## 2. 0-R: feature를 전혀 보지 않는 분류

가장 단순한 생각은 training data에서 **가장 많이 나타난 class(majority class)** 를 모든 새 instance에 대해 예측하는 것이다.

| Class | Count | Rule |
|:------|:-----:|:-----|
| Play = yes | 9 | 항상 Yes로 예측 |
| Play = no | 5 | 사용하지 않음 |

> **0-R rule:** 새 instance의 Outlook, Temperature, Humidity, Windy 값이 무엇이든 Play = Yes.

| Accuracy | Error rate |
|:--------:|:----------:|
| 9/14 = 64.3% | 5/14 = 35.7% |

**0-R부터 시작하는 이유**

- 0-R은 이후의 classifier가 넘어야 할 가장 기본적인 **baseline** 이다.
- feature를 추가했을 때 실제로 성능이 좋아졌는지를 비교할 기준이 된다.

> **생각해 보기:** 새 모델의 accuracy가 65%라면, 0-R의 64.3%보다 충분히 좋은 모델이라고 할 수 있을까?
>
> 정확도의 절댓값만 보면 안 된다. 아무 feature도 보지 않고 다수 class만 예측해도 64.3%가 나오므로, 65%는 feature를 사용해 얻은 이득이 거의 없다는 뜻이다. 성능은 항상 baseline과 비교해 판단한다.

---

<br>

## 3. 1-R: feature 하나만 사용하는 분류

### 3.1 알고리즘

**1-R(One-R)** 은 각 attribute를 하나씩 시험한다. 그 attribute의 각 value에 대해 가장 많이 나타나는 class를 rule로 만든 뒤, **total error가 가장 작은 attribute** 를 선택한다. 결과는 attribute 하나에 대한 1단계 rule 집합이다.

```text
For each attribute:
    For each value of that attribute:
        count how often each class appears
        find the most frequent class
        make the rule assign that class to this value
    calculate the error rate of the rules
Choose the rules with the smallest error rate.
```

### 3.2 Weather 데이터셋에 적용하기

| Attribute | Rules | Errors | Total Errors |
|:----------|:------|:------:|:------------:|
| outlook | sunny → no | 2/5 | **4/14** |
| | overcast → yes | 0/4 | |
| | rainy → yes | 2/5 | |
| temperature | hot → no | 2/4 | 5/14 |
| | mild → yes | 2/6 | |
| | cool → yes | 1/4 | |
| humidity | high → no | 3/7 | **4/14** |
| | normal → yes | 1/7 | |
| windy | false → yes | 2/8 | 5/14 |
| | true → no | 3/6 | |

Outlook의 error는 다음처럼 센다. sunny인 날 5일 중 yes 2일, no 3일이므로 rule은 sunny → no이고 틀리는 날은 2일이다. overcast 4일은 모두 yes라서 틀리는 날이 없다. rainy 5일 중 yes 3일, no 2일이므로 rule은 rainy → yes이고 2일이 틀린다. 합하면 2 + 0 + 2 = 4, 즉 **4/14** 이다.

> **선택된 1-R rule (Outlook):** sunny → no, overcast → yes, rainy → yes
>
> Training accuracy = 10/14 = 71.4%

0-R의 64.3%에서 1-R의 71.4%로, feature 하나를 사용하는 것만으로 training data에서는 오류가 줄었다.

> **참고:** Outlook과 Humidity의 total error는 모두 4/14로 같다. 동점일 때는 둘 중 하나를 임의로 고르며, 여기서는 Outlook을 택했다. Temperature의 hot(2 yes, 2 no)과 Windy의 true(3 yes, 3 no)처럼 value 안에서 class 수가 같을 때도 임의로 하나를 고르므로 rule이 no로 정해져 있다.

---

<br>

## 4. 1-R과 수치형 feature

### 4.1 Discretization

Temperature가 숫자로 주어지면 각 숫자를 모두 서로 다른 value로 취급할 수도 있다. 그러나 그렇게 하면 value 수가 너무 많아지고, training data를 외우는 rule이 생길 수 있다.

**Weather 데이터의 Temperature 값** (값을 정렬하고 각 값의 class를 적은 것)

| 64 | 65 | 68 | 69 | 70 | 71 | 72 | 72 | 75 | 75 | 80 | 81 | 83 | 85 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| yes | no | yes | yes | yes | no | no | yes | yes | yes | no | yes | yes | no |

**Discretization(이산화)** 은 numerical value를 몇 개의 interval로 묶어 categorical value처럼 사용하는 방법, 즉 numerical data를 categorical data로 변환하는 방법이다. 예를 들어 나이처럼 0부터 120 사이에서 25, 39, 12, 58, 8, …과 같이 다양한 값을 갖는 feature는 다음과 같이 네 구간으로 나누고 각 구간을 하나의 범주로 바꿀 수 있다.

| 구간 | 범주 |
|:----:|:----:|
| [0, 25] | y1 |
| (25, 35) | y2 |
| [35, 60] | y3 |
| [60, ∞) | y4 |

그러면 나이 열은 숫자 대신 y1~y4 중 하나의 값을 갖는 범주형 feature가 된다. 각 구간에는 여러 instance가 모이므로, 구간마다 majority class를 정하는 1-R rule이 의미를 갖는다.

Temperature에 대해 구간 수를 달리하면 다음과 같다.

![그림 1. 구간 수에 따른 Temperature의 discretization (A 구간 하나, B 77.5 기준 두 구간, C 지나치게 많은 구간) (4쪽)](../images/L01_p04.png)

*그림 1. 구간 수에 따른 Temperature의 discretization (A 구간 하나, B 77.5 기준 두 구간, C 지나치게 많은 구간) (4쪽)*

- **A. No split:** 구간이 하나뿐이라 매우 단순하다. 0-R과 같다.
- **B. Two intervals:** threshold = 77.5에서 나눈다. 최종 1-R rule이다.
- **C. Too many intervals:** 구간이 너무 많아 training data를 외울 수 있다(overfitting).

> **핵심:** 구간이 너무 적으면 중요한 차이를 놓칠 수 있고(underfitting), 너무 많으면 training instance를 거의 하나씩 외우게 되어 overfitting이 생길 수 있다.

### 4.2 구간을 정하는 방법

교재의 방법은 각 partition에서 **majority class의 사례 수가 일정 수 이상** 이 되도록 최소 제한을 두고, **인접 partition의 majority class가 같으면 합치는** 방식이다.

**예: minimum = 3**

1. **초기 class sequence:** class가 바뀔 때마다 나누면 다음과 같다.

   `yes | no | yes yes yes | no no | yes yes yes | no | yes yes | no`

2. **partition을 만들고 인접한 동일 majority class를 합치면:** majority class가 적어도 3개가 되도록 partition을 넓힌 뒤, majority가 같은 이웃 partition을 합친다.

   `yes no yes yes yes no no yes yes yes | no yes yes no`

3. **최종 discretization:** 경계는 75와 80 사이의 중간값이다.

   - temperature ≤ 77.5 → yes
   - temperature > 77.5 → no

> **참고:** 마지막 partition `no yes yes no` 는 yes와 no가 2개씩이라 동점이다. 교재는 이 partition에 no를 준다.

### 4.3 Overfitting이 문제인 이유

1-R은 한 attribute의 value로 feature space를 띠 모양으로 나누고, 각 띠의 error rate로 attribute를 고른다. Weather 데이터의 nominal attribute는 value가 outlook 3가지, temperature 3가지, humidity 2가지, windy 2가지뿐이라 띠 하나에 여러 instance가 모인다. 그런데 temperature의 value가 10가지라면 공간이 아주 좁은 띠로 나뉘고, 띠 하나에 instance가 한두 개만 남는다. 이렇게 value가 많은 attribute(multi-valued feature)는 rule이 training data를 외우게 만든다.

- **ID code** 나 주민번호처럼 instance를 거의 유일하게 식별하는 feature는 training set에서 error 0인 rule을 만들 수 있다. instance마다 값이 다르므로 이런 feature는 분류에 가장 부적합한 feature이다.
- 하지만 test data의 새로운 value에는 적용할 규칙이 없어 일반화가 매우 나쁘다.
- 1-R에서는 attribute가 많은 것이 문제가 아니라, **한 attribute가 지나치게 많은 distinct values를 가질 때** 특히 조심해야 한다.

이 차이는 모델의 경계로도 볼 수 있다. 두 class를 직선 하나로 나누는 모델은 **일반화된(generalized) 모델** 이다. 반면 training data의 점 하나하나를 피해 구불구불하게 경계를 긋는 모델은 **training data에 지나치게 맞춘(overfitted) 모델** 이다. 이 모델은 training error가 0에 가깝지만, 새 data는 그 구불구불한 경계의 엉뚱한 쪽에 떨어지기 쉽다.

> **Discretization의 trade-off:** 구간 수 증가 → training error는 내려가기 쉬움 → model complexity 증가 → test data에서의 generalization은 오히려 나빠질 수 있음.

---

<br>

## 5. Bayes 정리

### 5.1 정리와 용어

1-R은 한 feature만 선택했다. 이제 Outlook, Temperature, Humidity, Windy를 모두 함께 사용해 class를 판단하고 싶다. 이를 위해 먼저 **Bayes theorem** 을 살펴본다.

$$
P(H \mid E) = \frac{P(E \mid H)\,P(H)}{P(E)}
$$

| 용어 | 표기 | 의미 |
|:-----|:----:|:-----|
| Prior probability | P(H) | evidence를 보기 전 hypothesis의 확률 |
| Conditional probability (likelihood) | P(E \| H) | H가 참일 때 evidence가 나타날 확률 |
| Posterior probability | P(H \| E) | evidence를 본 뒤 H의 확률 |

### 5.2 예: meningitis와 stiff neck

Bayes 정리의 고전적인 예로 뇌수막염(meningitis)과 목 경직(stiff neck)을 보자.

- M: meningitis patient (뇌수막염 환자)
- S: stiff neck if patient (목이 뻣뻣함)
- P(M) = 1/50000, P(S) = 1/20, P(S | M) = 0.5

P(S | M) = 0.5가 주어졌을 때 구하려는 것은 posterior probability, 즉 **stiff neck일 때 meningitis일 확률** 이다.

$$
P(M \mid S) = \frac{P(S \mid M)\,P(M)}{P(S)} = \frac{0.5 \times 1/50000}{1/20} = 0.0002
$$

> **해석:** 목이 뻣뻣하다는 evidence가 있어도 meningitis의 posterior probability는 0.0002이다. 흔하지 않은 사건에서는 prior가 매우 중요하다.

이처럼 Bayes theorem을 이용하여 classification을 하는 모델이 **Bayes Classifier** 이며, 이때 분류의 근거가 되는 규칙을 **Bayes rule** 이라 한다.

---

<br>

## 6. Naive Bayesian Classifier

### 6.1 Naive 가정

Bayes theorem을 classification에 적용하면, 새 instance의 evidence E가 주어졌을 때 각 class H의 posterior probability를 계산하고 가장 큰 class를 선택할 수 있다. evidence가 feature 값 x₁, …, xₙ으로 이루어지면 다음과 같다.

$$
P(C \mid x_1, \ldots, x_n) \propto P(C) \prod_{i=1}^{n} P(x_i \mid C)
$$

> **Naive assumption:** class가 주어졌을 때 각 feature는 서로 **independent** 하다고 가정한다. 따라서 여러 feature의 likelihood를 곱으로 분해할 수 있다.

분모 P(E)는 모든 class에 공통이므로 class를 고를 때는 계산하지 않아도 된다. 그래서 "="가 아니라 비례(∝)로 쓴다.

### 6.2 Weather 데이터의 count와 확률

**교재 Table 4.2: The weather data, with counts and probabilities**

<table>
<thead>
<tr><th></th><th colspan="3">outlook</th><th colspan="3">temperature</th><th colspan="3">humidity</th><th colspan="3">windy</th><th colspan="2">play</th></tr>
<tr><th></th><th></th><th>yes</th><th>no</th><th></th><th>yes</th><th>no</th><th></th><th>yes</th><th>no</th><th></th><th>yes</th><th>no</th><th>yes</th><th>no</th></tr>
</thead>
<tbody>
<tr><th rowspan="3">개수</th><td>sunny</td><td>2</td><td>3</td><td>hot</td><td>2</td><td>2</td><td>high</td><td>3</td><td>4</td><td>false</td><td>6</td><td>2</td><td>9</td><td>5</td></tr>
<tr><td>overcast</td><td>4</td><td>0</td><td>mild</td><td>4</td><td>2</td><td>normal</td><td>6</td><td>1</td><td>true</td><td>3</td><td>3</td><td></td><td></td></tr>
<tr><td>rainy</td><td>3</td><td>2</td><td>cool</td><td>3</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th rowspan="3">확률</th><td>sunny</td><td>2/9</td><td>3/5</td><td>hot</td><td>2/9</td><td>2/5</td><td>high</td><td>3/9</td><td>4/5</td><td>false</td><td>6/9</td><td>2/5</td><td>9/14</td><td>5/14</td></tr>
<tr><td>overcast</td><td>4/9</td><td>0/5</td><td>mild</td><td>4/9</td><td>2/5</td><td>normal</td><td>6/9</td><td>1/5</td><td>true</td><td>3/9</td><td>3/5</td><td></td><td></td></tr>
<tr><td>rainy</td><td>3/9</td><td>2/5</td><td>cool</td><td>3/9</td><td>1/5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>

위쪽 세 줄은 count이고, 아래쪽 세 줄은 그 count를 class별 개수(yes 9, no 5)로 나눈 확률이다. 예를 들어 outlook = sunny의 count는 yes 2, no 3이고, 확률은 2/9와 3/5이다. play 열의 9/14와 5/14는 class prior이다.

> **주의:** P(yes | sunny)가 아니라 **P(sunny | yes)** 를 곱한다. 표에서 outlook = sunny일 때 yes가 2번이다. 이때 확률은 2/5가 아니고 2/9이다. 즉, sunny일 때 yes일 확률이 아니라, **yes일 때 sunny일 확률** 이다. sunny이고 yes인 사례는 yes 9개 중 2개이므로 P(sunny | yes) = 2/9이다.

---

<br>

## 7. Weather 예제: 새 instance 분류

> **A new day:** Outlook = sunny, Temperature = cool, Humidity = high, Windy = true, Play = ???

**Yes의 likelihood × prior**

$$
\frac{2}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{9}{14} = 0.0053
$$

**No의 likelihood × prior**

$$
\frac{3}{5} \times \frac{1}{5} \times \frac{4}{5} \times \frac{3}{5} \times \frac{5}{14} = 0.0206
$$

**정규화한 posterior probability**

$$
P(yes \mid E) = \frac{0.0053}{0.0053 + 0.0206} = 20.5\%, \qquad P(no \mid E) = 79.5\%
$$

> **Prediction:** No (79.5%)

분류만 할 때는 두 class의 공통 분모 P(E)를 직접 구하지 않고, P(E | class)P(class)를 비교해도 같은 class가 선택된다. 위 계산은 P(play = yes | E)를 다음과 같이 풀어 쓴 것이다.

$$
P(yes \mid E) = \frac{2/9 \times 3/9 \times 3/9 \times 3/9 \times 9/14}{P(E)}
$$

Bayes 식 P(H | e) = P(e | H)P(H) / P(e)에서 P(e | H)가 **likelihood**, P(H)가 **prior probability** 이다. 이렇게 구한 사후 확률을 비교하여 play를 결정하는 것이 조건부 확률에 대한 Bayes 규칙(**Bayes's rule of conditional probability**)의 적용이다.

---

<br>

## 8. Promotion 데이터 예제

여러 binary attribute를 관측하고 Sex class를 분류하는 예를 보자.

**Training data**

| Magazine Promotion | Watch Promotion | Life Insurance Promotion | Credit Card Insurance | Sex |
|:------------------:|:---------------:|:------------------------:|:---------------------:|:---:|
| Yes | No | No | No | Male |
| Yes | Yes | Yes | Yes | Female |
| No | No | No | No | Male |
| Yes | Yes | Yes | Yes | Male |
| Yes | No | Yes | No | Female |
| No | No | No | No | Female |
| Yes | Yes | Yes | Yes | Female |
| No | No | No | No | Male |
| Yes | No | No | No | Male |
| Yes | Yes | No | No | Female |

> **예측할 instance (Evidence):** Magazine Promotion = Yes, Watch Promotion = Yes, Life Insurance Promotion = No, Credit Card Insurance = No, Sex = ?

각 class에 대해 위 evidence의 conditional probabilities와 prior를 곱한 뒤 더 큰 쪽을 선택한다. 이 데이터를 attribute별 count와 비율로 정리한 표(Table 10.5)는 다음과 같다.

**Table 10.5: Counts and Probabilities for Attribute Sex**

<table>
<thead>
<tr><th rowspan="2">Sex</th><th colspan="2">Magazine Promotion</th><th colspan="2">Watch Promotion</th><th colspan="2">Life Insurance Promotion</th><th colspan="2">Credit Card Insurance</th></tr>
<tr><th>Male</th><th>Female</th><th>Male</th><th>Female</th><th>Male</th><th>Female</th><th>Male</th><th>Female</th></tr>
</thead>
<tbody>
<tr><td>Yes</td><td>4</td><td>3</td><td>2</td><td>2</td><td>2</td><td>3</td><td>2</td><td>1</td></tr>
<tr><td>No</td><td>2</td><td>1</td><td>4</td><td>2</td><td>4</td><td>1</td><td>4</td><td>3</td></tr>
<tr><td>Ratio: yes/total</td><td>4/6</td><td>3/4</td><td>2/6</td><td>2/4</td><td>2/6</td><td>3/4</td><td>2/6</td><td>1/4</td></tr>
<tr><td>Ratio: no/total</td><td>2/6</td><td>1/4</td><td>4/6</td><td>2/4</td><td>4/6</td><td>1/4</td><td>4/6</td><td>3/4</td></tr>
</tbody>
</table>

Table 10.5에서 Male은 6명, Female은 4명이다. 이를 이용해 계산하면 다음과 같다.

$$
P(E \mid Male)\,P(Male) = \frac{4}{6} \times \frac{2}{6} \times \frac{4}{6} \times \frac{4}{6} \times \frac{6}{10} \approx 0.0593
$$

$$
P(E \mid Female)\,P(Female) = \frac{3}{4} \times \frac{2}{4} \times \frac{1}{4} \times \frac{3}{4} \times \frac{4}{10} \approx 0.0281
$$

0.0593 > 0.0281이므로 **Sex = Male** 로 분류한다. 정규화하면 Male일 확률은 약 67.8%이다.

> **참고:** 위 training data 표는 Table 10.5와 두 행이 다르다. Table 10.5와 맞으려면 7번째 행의 Sex가 Male, 10번째 행의 Life Insurance Promotion이 Yes여야 한다. 위 표를 그대로 세면 Male과 Female이 5명씩이고, 결과도 Female(0.0576 > 0.0384)로 뒤집힌다. 위 계산은 Table 10.5의 count를 따랐다.

---

<br>

## 9. Naive Bayes와 수치형 feature

### 9.1 Gaussian 확률 밀도

Temperature, Humidity가 실제 숫자라면 각 숫자의 frequency를 직접 세는 대신 **class별 Gaussian distribution** 을 가정하고 **probability density** 를 사용한다. 즉 확률을 구하기 위하여 확률 분포함수(PDF: Probability Density Function)를 사용한다.

$$
f(x \mid C) = \frac{1}{\sigma_C \sqrt{2\pi}} \exp\!\left( -\frac{(x - \mu_C)^2}{2\sigma_C^2} \right)
$$

- e: the exponential function (지수 함수)
- μ: the class mean for the given numerical attribute (해당 수치형 attribute의 class별 평균)
- σ: the class standard deviation for the attribute (해당 attribute의 class별 표준편차)
- x: the attribute value (attribute 값)

### 9.2 Numeric weather 데이터

교재의 Table 1.3은 Temperature와 Humidity가 숫자로 주어진 weather 데이터이다.

**교재 Table 1.3: Weather data with some numeric attributes**

| outlook | temperature | humidity | windy | play |
|:--------|:-----------:|:--------:|:-----:|:----:|
| sunny | 85 | 85 | false | no |
| sunny | 80 | 90 | true | no |
| overcast | 83 | 86 | false | yes |
| rainy | 70 | 96 | false | yes |
| rainy | 68 | 80 | false | yes |
| rainy | 65 | 70 | true | no |
| overcast | 64 | 65 | true | yes |
| sunny | 72 | 95 | false | no |
| sunny | 69 | 70 | false | yes |
| rainy | 75 | 80 | false | yes |
| sunny | 75 | 70 | true | yes |
| overcast | 72 | 90 | true | yes |
| overcast | 81 | 75 | false | yes |
| rainy | 71 | 91 | true | no |

numeric data의 summary(교재 Table 4.4)를 보면, nominal attribute는 앞과 같이 count와 확률로, numeric attribute는 class별 값의 목록과 평균, 표준편차로 정리된다.

**교재 Table 4.4: The numeric weather data with summary statistics**

<table>
<thead>
<tr><th colspan="3">outlook</th><th colspan="3">temperature</th><th colspan="3">humidity</th><th colspan="3">windy</th><th colspan="2">play</th></tr>
<tr><th></th><th>yes</th><th>no</th><th></th><th>yes</th><th>no</th><th></th><th>yes</th><th>no</th><th></th><th>yes</th><th>no</th><th>yes</th><th>no</th></tr>
</thead>
<tbody>
<tr><td>sunny</td><td>2</td><td>3</td><td></td><td>83</td><td>85</td><td></td><td>86</td><td>85</td><td>false</td><td>6</td><td>2</td><td>9</td><td>5</td></tr>
<tr><td>overcast</td><td>4</td><td>0</td><td></td><td>70</td><td>80</td><td></td><td>96</td><td>90</td><td>true</td><td>3</td><td>3</td><td></td><td></td></tr>
<tr><td>rainy</td><td>3</td><td>2</td><td></td><td>68</td><td>65</td><td></td><td>80</td><td>70</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td>64</td><td>72</td><td></td><td>65</td><td>95</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td>69</td><td>71</td><td></td><td>70</td><td>91</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td>75</td><td></td><td></td><td>80</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td>75</td><td></td><td></td><td>70</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td>72</td><td></td><td></td><td>90</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td>81</td><td></td><td></td><td>75</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>sunny</td><td>2/9</td><td>3/5</td><td>mean</td><td>73</td><td>74.6</td><td>mean</td><td>79.1</td><td>86.2</td><td>false</td><td>6/9</td><td>2/5</td><td>9/14</td><td>5/14</td></tr>
<tr><td>overcast</td><td>4/9</td><td>0/5</td><td>std dev</td><td>6.2</td><td>7.9</td><td>std dev</td><td>10.2</td><td>9.7</td><td>true</td><td>3/9</td><td>3/5</td><td></td><td></td></tr>
<tr><td>rainy</td><td>3/9</td><td>2/5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>

표 아래쪽의 mean과 std dev가 Gaussian 식의 μ와 σ로 쓰인다. 예를 들어 Temperature는 yes일 때 μ = 73, σ = 6.2이고 no일 때 μ = 74.6, σ = 7.9이다.

### 9.3 새 instance 분류

> **A new day:** Outlook = sunny, Temperature = 66, Humidity = 90, Windy = true, Play = ???

class가 yes일 때 Temperature = 66의 probability density는 다음과 같다.

$$
f(temperature = 66 \mid yes) = \frac{1}{\sqrt{2\pi} \cdot 6.2} \, e^{-\frac{(66 - 73)^2}{2 \cdot 6.2^2}} = 0.0340
$$

같은 방법으로 humidity가 90일 때 yes의 probability density는 다음과 같다.

$$
f(humidity = 90 \mid yes) = 0.0221
$$

따라서 nominal attribute의 확률과 numeric attribute의 density를 함께 곱한다.

$$
P(E \mid yes)\,P(yes) = \frac{2}{9} \times 0.0340 \times 0.0221 \times \frac{3}{9} \times \frac{9}{14} = 0.000036
$$

$$
P(E \mid no)\,P(no) = \frac{3}{5} \times 0.0291 \times 0.0380 \times \frac{3}{5} \times \frac{5}{14} = 0.000136
$$

$$
P(yes \mid E) = \frac{0.000036}{0.000036 + 0.000136} = 20.9\%, \qquad P(no \mid E) = 79.1\%
$$

> **Prediction:** No

> **참고:** 식에 직접 넣어 보면 f(temperature = 66 | no) ≈ 0.0279이다. 0.0279로 계산한 P(E | no)P(no)가 0.000136과 맞고, 0.0291을 쓰면 약 0.000142가 나오므로 0.0291은 교재의 오기로 보인다. 어느 값을 써도 예측은 No이다.

evidence e에 따라 두 class의 사후 확률 P(Yes | e)와 P(No | e)를 곡선으로 그려 보면, 주어진 e에서 H ∈ {Yes, No} 가운데 P(H | e)가 큰 쪽의 class를 고르는 것이 Naive Bayes의 결정이다. 위 예에서는 그 값이 20.9%와 79.1%이므로 No를 고른다.

### 9.4 Naive Bayes의 가정과 장단점

Naive Bayes의 성격은 다음 질문으로 정리할 수 있다.

> **Naive Bayes는 feature 사이의 관계를 모델링하는가?**
>
> Naive Bayes는 확률 모델로서 **모든 feature가 동등한 비중으로 기여** 하고, class가 주어졌을 때 **feature들이 서로 독립** 이라고 가정한다.
>
> 반면 일반적인 기계학습 모델은 feature마다 중요도가 다르다고 보며, feature 사이의 독립을 가정하지 않는다.

이 가정으로부터 다음과 같은 장단점이 나온다.

- **장점:** count와 평균, 표준편차만 구하면 되므로 학습이 매우 빠르고 단순하다. feature가 많아도 확률을 곱하기만 하면 되며, 확률값 자체를 결과로 준다.
- **단점:** 실제 데이터의 feature는 서로 독립이 아닌 경우가 많고, 중요한 feature와 덜 중요한 feature를 구분하지 않는다. 또 한 feature의 확률이 0이면 곱 전체가 0이 된다(10절 참고).

---

<br>

## 10. Spam email 분류 예제

단어(word)를 attribute로 보고 spam과 valid email(ham)을 분류하는 예를 보자.

![그림 2. 단어를 attribute로 사용하는 spam 예제 (18쪽)](../images/L01_p18.png)

*그림 2. 단어를 attribute로 사용하는 spam 예제 (18쪽)*

| Doc | Email text | Class |
|:---:|:-----------|:-----:|
| D1 | send us your password | spam |
| D2 | send us your review | ham |
| D3 | review your password | ham |
| D4 | review us | spam |
| D5 | send your password | spam |
| D6 | send us your account | spam |

> **New email:** "review us now"

**Prior**

| P(spam) | P(ham) |
|:-------:|:------:|
| 4/6 | 2/6 |

**Word probabilities**

| Word | P(word \| spam) | P(word \| ham) |
|:-----|:---------------:|:--------------:|
| password | 2/4 | 1/2 |
| review | 1/4 | 2/2 |
| send | 3/4 | 1/2 |
| us | 3/4 | 1/2 |
| your | 3/4 | 1/2 |
| account | 1/4 | 0/2 |

> **생각해 보기:** 새 email "review us now"를 분류하려면 어떤 단어의 probability를 사용해야 할까? training data에 없던 "now"는 어떻게 처리해야 할까?

이 예제는 Naive Bayes가 텍스트 분류에서 **"단어를 feature로 보고 여러 evidence를 확률적으로 결합"** 하는 방식을 보여준다.

**풀이.** 각 단어를 binary attribute로 보고, 단어가 포함되면 yes, 포함되지 않으면 no로 둔다. 어휘(vocabulary)는 training data에 나온 여섯 단어이고, training data에 없던 "now"는 어떤 class에도 정보를 주지 않으므로 계산에서 뺀다. 따라서 evidence는 다음과 같다.

E = (password = no, review = yes, send = no, us = yes, your = no, account = no)

단어가 없을 확률은 1에서 있을 확률을 빼서 구한다.

$$
P(E \mid spam)\,P(spam) = \frac{2}{4} \times \frac{1}{4} \times \frac{1}{4} \times \frac{3}{4} \times \frac{1}{4} \times \frac{3}{4} \times \frac{4}{6} = \frac{18}{4096} \times \frac{4}{6} \approx 0.00293
$$

$$
P(E \mid ham)\,P(ham) = \frac{1}{2} \times \frac{2}{2} \times \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} \times \frac{2}{2} \times \frac{2}{6} = \frac{1}{16} \times \frac{2}{6} \approx 0.02083
$$

$$
P(spam \mid E) = \frac{0.00293}{0.00293 + 0.02083} \approx 0.123
$$

P(spam | E) ≈ 12.3%이므로 "review us now"는 **ham** 으로 분류한다.

> **참고:** 단어 확률표는 D3을 "review your password"로 본 것이다(D3에 us가 있으면 P(us | ham)이 2/2가 된다). 또 문서를 직접 세면 your는 ham 두 문서에 모두 있으므로 P(your | ham) = 2/2인데, 확률표에는 1/2로 적혀 있다. 2/2를 쓰면 P(your = no | ham) = 0이 되어 ham의 곱 전체가 0이 된다. 이처럼 count가 0인 값 하나가 다른 모든 evidence를 무시하게 만드는 것을 **zero-frequency problem** 이라 하며, 보통 모든 count에 1을 더하는 **Laplace estimator** 로 해결한다(교재 4.2절).

---

<br>

## 11. 정리: feature를 어떻게 사용하는가

| Classifier | 사용하는 feature | 판단 방법 | 핵심 이슈 |
|:-----------|:----------------:|:----------|:----------|
| 0-R | 0개 | majority class | baseline |
| 1-R | 1개 | value별 majority rule + error 최소화 | discretization, overfitting |
| Naive Bayes | 여러 개 | prior × feature likelihoods | conditional independence assumption |

> **한 문장으로:** 0-R → 1-R → Naive Bayes는 "데이터에서 class를 예측할 때 feature 정보를 얼마나, 어떤 방식으로 사용할 것인가"를 단계적으로 보여준다.

**확인 질문**

- 새 instance, feature, feature value, class를 Weather dataset에서 직접 지목할 수 있는가?
- 1-R 계산표에서 왜 Outlook의 error가 4/14인지 설명할 수 있는가?
- 수치형 feature를 너무 잘게 discretize하면 왜 training error는 줄고 overfitting은 커지는가?
- Bayes 식에서 prior, likelihood, posterior를 구분할 수 있는가?
- Naive Bayes에서 왜 P(feature | class)를 곱하는지 설명할 수 있는가?

---

<br>

## 요약

| 개념 | 핵심 요약 |
|:-----|:----------|
| Classification | training data의 feature와 class로 규칙을 찾아 새 instance의 class를 정한다. |
| 0-R | feature를 보지 않고 majority class를 예측한다. Weather에서 accuracy 9/14 = 64.3%이며 모든 모델의 baseline이다. |
| 1-R | attribute마다 value별 majority rule을 만들고 total error가 가장 작은 attribute를 고른다. Weather에서는 Outlook(4/14)이 선택되어 accuracy 71.4%가 된다. |
| Discretization | 수치형 값을 구간으로 묶어 범주형처럼 쓴다. 각 구간의 majority class가 최소 개수 이상이 되게 하고 같은 majority의 이웃 구간은 합친다. |
| Overfitting | 구간이나 value가 너무 많으면 training error는 줄지만 일반화가 나빠진다. ID처럼 instance마다 다른 feature가 극단적인 예이다. |
| Bayes 정리 | P(H \| E) = P(E \| H)P(H) / P(E). 드문 사건에서는 prior가 결과를 크게 좌우한다. |
| Naive Bayes | class가 주어지면 feature가 독립이라고 가정하고 P(C)∏P(xᵢ \| C)가 가장 큰 class를 고른다. P(class \| value)가 아니라 P(value \| class)를 곱한다. |
| 수치형 feature | class별 평균과 표준편차로 Gaussian density를 구해 확률 대신 곱한다. |
| 텍스트 분류 | 단어를 binary attribute로 보고 확률을 곱한다. 확률이 0인 값은 Laplace estimator로 보정한다. |

---

<br>

## 점검 문제

1. **Baseline:** 어떤 데이터셋에서 class가 A 70개, B 30개이다. 새 모델의 accuracy가 72%라면 이 모델을 어떻게 평가해야 하는가?

   > **정답:** 0-R은 항상 A로 예측해 accuracy 70%를 얻는다. 72%는 baseline보다 2%p 높을 뿐이므로 feature를 사용해 얻은 이득이 작다. 모델의 성능은 0-R baseline과 비교해서 판단해야 한다.

2. **1-R error:** Weather 데이터에서 Humidity의 total error가 4/14인 이유를 설명하라.

   > **정답:** humidity = high인 7일은 yes 3, no 4이므로 rule은 high → no이고 3일이 틀린다. normal인 7일은 yes 6, no 1이므로 rule은 normal → yes이고 1일이 틀린다. 합은 3 + 1 = 4이므로 4/14이다.

3. **Overfitting:** 각 instance에 서로 다른 ID 번호가 붙어 있을 때, 1-R이 ID attribute를 고르면 training error와 test 성능은 어떻게 되는가?

   > **정답:** ID의 value마다 instance가 하나뿐이므로 각 value의 majority class가 곧 정답이 되어 training error는 0이다. 그러나 test data의 새 ID에는 적용할 rule이 없으므로 일반화 성능은 매우 나쁘다. 1-R은 distinct value가 지나치게 많은 attribute를 조심해야 한다.

4. **Bayes 정리:** P(M) = 1/50000, P(S) = 1/20, P(S | M) = 0.5일 때 P(M | S)를 구하고 그 의미를 설명하라.

   > **정답:** P(M | S) = 0.5 × (1/50000) / (1/20) = 0.0002이다. stiff neck이라는 evidence가 있어도 meningitis의 prior가 매우 작아서 posterior도 매우 작다.

5. **확률의 방향:** Weather 데이터에서 P(sunny | yes)는 얼마이며, P(yes | sunny)와 어떻게 다른가?

   > **정답:** P(sunny | yes)는 yes 9일 중 sunny인 2일의 비율이므로 2/9이다. P(yes | sunny)는 sunny 5일 중 yes인 2일의 비율인 2/5이다. Naive Bayes가 곱하는 것은 likelihood인 P(sunny | yes)이다.

6. **Naive Bayes 계산:** Weather 데이터에서 Outlook = overcast, Temperature = mild, Humidity = high, Windy = false인 날을 분류하라.

   > **정답:** yes: 4/9 × 4/9 × 3/9 × 6/9 × 9/14 ≈ 0.0282. no: 0/5 × 2/5 × 4/5 × 2/5 × 5/14 = 0이다. 따라서 yes로 분류한다. P(overcast | no) = 0이라서 no의 곱 전체가 0이 된 경우이며, Laplace estimator를 쓰면 0을 피할 수 있다.

7. **수치형 feature:** 수치형 attribute를 Naive Bayes에 넣을 때 무엇을 class별로 계산해 두어야 하며, 새 값의 확률은 어떻게 얻는가?

   > **정답:** class별로 그 attribute 값의 평균 μ와 표준편차 σ를 계산해 둔다. 새 값 x가 들어오면 Gaussian density f(x | C) = (1 / (σ√(2π))) × exp(−(x − μ)² / (2σ²))를 구해 nominal attribute의 확률과 함께 곱한다.

8. **텍스트 분류:** spam 예제에서 training data에 없는 단어 "now"를 계산에서 빼도 되는 이유는 무엇인가?

   > **정답:** "now"는 어휘에 없는 단어라 spam과 ham 어느 쪽에서도 count가 없다. 두 class에 같은 정보를 주므로 posterior의 비교에 영향을 주지 않는다. 따라서 어휘에 있는 여섯 단어만으로 evidence를 구성한다.

---
