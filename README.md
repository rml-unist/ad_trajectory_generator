# 자율주행 차량 제어를 위한 참조 궤적 생성 프로그램

Reference Trajectory Generator for Autonomous Vehicle Control

이 프로그램은 nuPlan 데이터셋에서 자율주행 차량이 계획한 궤적과 초기 차량 상태를 받아 제어기가 추종할 참조 궤적을 생성하는 프로그램입니다.
계획 경로의 이동 거리와 시간 간격으로 목표 속도를 구하고 가감속 제약을 반영한 뒤, 운동학 모델 기반 예측 제어(KMPC)로 참조 위치, 헤딩, 속도, 가속도, 조향각을 계산합니다.

## 설치

Linux, Python 3.9.23에서 검증했습니다. Git과 uv를 설치한 뒤 저장소 루트에서 실행합니다.

```bash
uv sync --locked --python 3.9.23
```

`uv.lock`은 nuPlan devkit과 라이브러리 버전을 고정합니다. 라이브러리만 설치하려면 `--no-dev`를 추가합니다.

## 예제

nuPlan 데이터셋과 GPU 없이 실행할 수 있는 8초 가속 구간 주행 예제입니다. 실제 주행에서는 제어기의 속도가 매우 빠르기 때문에 생성된 8초의 궤적 중에서 초반부만 사용됩니다.

```bash
OUT="../trajectory_generator_outputs/example_$(date +%Y%m%d_%H%M%S)"
uv run --locked trajectory-generator demo --case accelerate --output "$OUT" --plot
uv run --locked trajectory-generator check --input "$OUT"
uv run --locked trajectory-generator info --input "$OUT"
```

`$OUT/accelerate/`에서 입력 경로(`raw.csv`), 속도 계획을 반영한 목표(`retimed.csv`), 최종 참조(`reference.csv`)와 비교 그림(`trajectory.png`)을 확인할 수 있습니다.
`check`는 저장된 파일 구성, 8초 시간축, 참조 갱신 기록과 요약값을 대조합니다.

## Python에서 사용

실제 데이터는 호출 프로그램에서 준비합니다. 입력은 계획 궤적(`AbstractTrajectory`)과 계획 시작 시점의 차량 상태(`EgoState`)입니다. 위치는 차량 후륜축 기준 전역 좌표(m), 헤딩은 rad, 시각은 마이크로초입니다. 전진 또는 정차 궤적을 대상으로 하며, 기본 호출에는 초기 차량 시각부터 0.1초 간격으로 구성한 차량 상태(`EgoState`) 목록을 담은 계획 궤적을 전달합니다.

```python
from trajectory_generation import TrajectoryGenerationPipeline
from trajectory_generation.synthetic import make_trajectory

plan, ego = make_trajectory("straight")
session = TrajectoryGenerationPipeline().start_session(plan, ego)
window = session.generate_reference(ego.time_point).trajectory
print(len(window.get_sampled_trajectory()))
```

기본 설정으로 최대 1.5초 구간의 QP를 5스텝(0.5초)마다 다시 풀고, 그 사이에는 직전 해를 사용해 0.1초마다 참조를 제공합니다.
같은 계획의 다음 호출에는 `session.reference_state.time_point`를 전달합니다. 새 계획을 받으면 해당 시점의 차량 상태로 새 세션을 시작합니다.

갱신 주기 등 설정을 바꾸려면 `TrajectoryGenerationConfig`를 파이프라인에 전달합니다. CLI에서는 저장된 `config.json`을 `demo --config 경로`로 지정할 수 있습니다.

## LICENSE

[MIT LICENSE](LICENSE)를 따릅니다.
