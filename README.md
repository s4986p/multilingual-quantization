# multilingual quantization

## 배경

[the multilingual quantization tax](https://arxiv.org/abs/2608.09941)에서는 양자화가 저자원 언어와 일부 비라틴 문자 언어에서 특히 더 많이 잃는다고 보고한다. 그러나 nf4 한 가지 양자화 방법으로 thinking mode를 끄고 한 번 돌려 얻은 결과이다. 또 오차 범위 안의 작은 차이를 superior처럼 크게 표현하거나 bf16에서 이미 무너진 언어를 양자화 탓으로 해석하기도 한다. 그럼에도 문제의식 자체는 꽤나 흥미롭고 영어가 모국어가 아닌 한국인으로서 공감도 간다. 따라서 관련 논문들을 모아 같은 현상을 검증하고 비교해 정리하며 가능하다면 더 나은 방법까지 제시하고자 한다.

## 문서

- [docs/results.md](docs/results.md): 모든 실험의 설정과 결과
