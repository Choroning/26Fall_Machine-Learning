# [2026학년도 가을학기] 기계학습

![Last Commit](https://img.shields.io/github/last-commit/Choroning/26Fall_Machine-Learning)
![Languages](https://img.shields.io/github/languages/top/Choroning/26Fall_Machine-Learning)

이 레포지토리는 대학 강의 및 과제를 위해 작성된 학습 노트와 예제 코드를 체계적으로 정리하고 보관합니다.

*작성자: 박철원 (고려대학교(서울), 소프트웨어기술벤처융합전공) - 2026년 기준 3학년*
<br><br>

## 📑 목차

- [레포지토리 소개](#about-this-repository)
- [강의 정보](#course-information)
- [사전 요구사항](#prerequisites)
- [레포지토리 구조](#repository-structure)
- [라이선스](#license)

---


<br><a name="about-this-repository"></a>
## 📝 레포지토리 소개

이 레포지토리에는 대학 수준의 기계학습 과목을 위해 작성된 이중 언어 학습 자료와 코드가 포함되어 있습니다:

- 각 강의자료마다 한국어(`.ko.md`)와 영어(`.md`)로 작성된 이중 언어 개념 정리 노트가 있습니다.
- 각 과제에는 솔루션과 함께 상세한 설명 문서가 포함되어 있습니다.
- 디렉토리는 강의자료 번호(`L01`, `L02` 등)를 기준으로 구성하며, 주차별 계획 및 진도 표에 각 주차에 진행한 강의자료를 정리합니다.

> **🤖 AI 에이전트 활용**
> [Claude Code](https://claude.ai/download)와 [Codex](https://github.com/openai/codex)를 강의 내용 정리를 위한 학습 보조 도구로 활용하였습니다.

<br><a name="course-information"></a>
## 📚 강의 정보

- **학기:** 2026학년도 가을학기 (9월 - 12월)
- **소속:** 고려대학교(서울)

|학수번호      |강의명    |이수구분|교수자|개설학과|
|:----------:|:-------|:----:|:------:|:----------------|
|`COSE362-01`|기계학습|전공선택|김현철 교수|컴퓨터학과|

### 과목 개요

컴퓨터 프로그램이 경험을 통해 스스로 성능을 개선할 수 있는 알고리즘을 다룬다. 특히 개념 학습(Concept Learning), 베이즈 학습(Bayesian Learning), 은닉 마르코프 모델(Hidden Markov Models), 의사결정나무(Decision Trees), 인공 신경망(Artificial Neural Networks), 신뢰 네트워크(Belief Networks), 커널 머신(Kernel Machines), 비지도 학습(Unsupervised Learning), 회귀(Regression), 다중 학습기(Multiple Learners) 등을 학습한다. 데이터를 기반으로 하는 다양한 기계학습 알고리즘의 개념을 이해하고 사용하는 방법을 익히며, 그 결과를 분석할 수 있도록 한다. 수업은 주어진 데이터에 대하여 다양한 모델을 만들고 분석하는 개별 텀 프로젝트로 진행한다.

### 교수자

- **교수자:** 김현철 교수 (Prof. Hyeoncheol Kim)
- **면담:** 수업 후 1시간, 교수 연구실

### 수업 일정 및 형식

- **학점:** 3학점
- **수업 시간:** 화요일 5교시, 목요일 5교시 (15:00 ~ 16:15)
- **강의실:** 애기능생활관 301호
- **수업 형식:** 대면 강의와 실습을 병행한다. 강의노트는 PDF로 제공되며, 주어진 문제는 Python 프로그래밍 언어로 해결한다.

### 평가

| 항목 | 비율 |
|:-----|-----:|
| 중간고사 | 30% |
| 기말고사 | 30% |
| 숙제, 퀴즈, 실습 | 30% |
| 출석, 수업 참여 | 10% |

- 성적은 절대평가로 부여하며, 시험은 감독하에 치른다.
- 기말고사는 15주차 텀 프로젝트 발표로 진행하며, 텀 프로젝트는 16주차에 제출한다.

### 수업 운영 규정

**출석**
- 총 수업 시간의 1/3 이상 결석하면 성적을 부여할 수 없다. 단, 담당 교수가 불가피한 결석으로 인정하는 경우에는 예외가 적용될 수 있다.
- 출석 인정은 특정 사유에 한해 신청할 수 있으며, 이 경우에도 출석 인정분을 제외하고 수업의 1/2 이상은 실제로 출석해야 한다.

### 주차별 계획 및 진도

| 주차 | 수업일 (화, 목) | 계획된 주제 | 진행한 강의자료 |
|:----:|:-----:|:--------------|:----------------------|
|1주차|09/01, 09/03|강의 소개||
|2주차|09/08, 09/10|기계학습과 데이터마이닝 개요|01. Classification: 0-R, 1-R, Naive Bayes<br>02. Information, Uncertainty, and Entropy|
|3주차|09/15, 09/17|입력과 출력|03. Decision Tree 의사결정나무<br>04. Linear Model에서 Neural Network까지|
|4주차|09/22, 09/24|기본 알고리즘 I||
|5주차|09/29, 10/01|기본 알고리즘 II||
|6주차|10/06, 10/08|평가 방법|05. Non-Linear 비선형 모델 (MLP)|
|7주차|10/13, 10/15|의사결정나무||
|8주차|10/20, 10/22|중간고사||
|9주차|10/27, 10/29|분류 규칙과 연관 규칙||
|10주차|11/03, 11/05|KNN과 SVM||
|11주차|11/10, 11/12|데이터 변환||
|12주차|11/17, 11/19|확률적 방법||
|13주차|11/24, 11/26|신경망과 딥러닝 소개||
|14주차|12/01, 12/03|앙상블 방법 외||
|15주차|12/08, 12/10|기말고사 (텀 프로젝트 발표)||
|16주차|12/15, 12/17|텀 프로젝트 제출||

- 계획된 주제는 강의 계획을 따르며, 실제 강의자료는 시간이 되는 만큼 이어서 진행한다.
- 강의자료 02는 2주차에 처음 배포된 뒤 3주차에 수정본으로 교체되었고, 강의자료 04는 5주차에 내용이 보강되어 다시 배포되었다. 정리 노트는 각 강의자료의 최신본을 따른다.

- **📖 참고 자료**

| 유형 | 내용 |
|:----:|:---------|
|교재|지정 교재 없음|
|추천 도서|Data Mining: Practical Machine Learning Tools and Techniques 3판 이상 (Ian Witten, Eibe Frank, Morgan Kaufmann)|
|강의자료|교수자 제공 강의노트 (PDF)|

<br><a name="prerequisites"></a>
## ✅ 사전 요구사항

- 지정된 선이수 과목은 없습니다.
- 수업 전반에 걸쳐 Python 프로그래밍 언어를 사용합니다.

- **💻 개발 환경**

| 도구 | 회사 |  운영체제  | 비고 |
|:-----|:-------:|:----:|:------|
|Visual Studio Code|Microsoft|macOS|    |

<br><a name="repository-structure"></a>
## 🗂 레포지토리 구조

```plaintext
26Fall_Machine-Learning
├── L01_0R-1R-and-Naive-Bayes
│   ├── Concepts_Lecture.ko.md
│   └── Concepts_Lecture.md
├── L02_Information-and-Entropy
│   ├── Concepts_Lecture.ko.md
│   └── Concepts_Lecture.md
├── L03_Decision-Tree
│   ├── Concepts_Lecture.ko.md
│   └── Concepts_Lecture.md
├── L04_Linear-Models
│   ├── Concepts_Lecture.ko.md
│   └── Concepts_Lecture.md
├── L05_Non-Linear-Models-and-MLP
│   ├── Concepts_Lecture.ko.md
│   └── Concepts_Lecture.md
├── images
│   └── (강의 도표 이미지)
├── LICENSE
├── README.ko.md
└── README.md
```

<br><a name="license"></a>
## 🤝 라이선스

이 레포지토리는 [MIT License](LICENSE) 하에 배포됩니다.

---
