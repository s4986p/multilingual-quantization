# multilingual quantization

## 배경

[the multilingual quantization tax](https://arxiv.org/abs/2608.09941)는 양자화가 영어가 아닌 언어에서 특히 더 많이 잃는다고 보고한다. 그러나 이 결과는 nf4 한 가지 양자화 방법으로 thinking mode를 끄고 얻은 것이다. superior 같은 과장된 표현도 쓴다. 그럼에도 언어마다 양자화의 영향이 다르다는 문제의식은 흥미롭다. 따라서 관련 논문들을 모아 같은 현상을 검증하고 비교해 정리한다. 가능하다면 더 나은 방법까지 제시한다.

## 문서

- [결과](docs/results.md): 모든 실험의 설정과 결과
