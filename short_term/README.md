# 단기 온도 예측 — 장기 실험과 동일한 조건

ST-GNN, GRU, LSTM을 10·30·60분 온도 예측에 각각 재학습했습니다. 기존 단기/장기 모델과 결과는 수정하지 않았습니다.

## 맞춘 조건

| 항목 | 적용값 |
|---|---|
| 센서 집합 | core 8개 / extended 12개, 장기 실험과 동일 |
| 날짜 | core: 학습 8/1~9/6, 검증 9/7~9/13, 시험 9/14~9/20; extended: 9/3~13, 9/14~16, 9/17~20 |
| 입력 | 과거 24시간(144개 10분 측정), 시간별 온도·습도 평균/마지막 값, 시각·요일 사인/코사인 |
| 입력 텐서 | 24개 시간 블록 × 센서 수 × 8개 특성 |
| 예측 기준 시각 간격 | 1시간, 장기 실험과 정확히 같은 시각 |
| 학습/검증/시험 창 수 | core 841/121/121, extended 217/25/49 |
| 공간 그래프 | 동일 좌표·가중치·정규화; N02 점선/최저 가중치, N15·N25 실선/중간 가중치 |
| 구조 | ST-GNN 시간·공간 블록 4개/채널24, GRU·LSTM 은닉48. 출력층만 5개 장기 시점에서 3개 단기 시점으로 변경 |
| 최적화 | AdamW, lr 0.001, weight_decay 0.0001, batch64, gradient clip1, 정규화 온도 MSE |
| 종료/선택 | 최대30 epoch, patience7, 검증 MAE로 체크포인트와 지도용 초기값 선택 |
| 초기값 | 17·29·43 |
| 평가 | 동일 시험 기준 시각의 센서·예측시차·요일·시간대별 MAE/RMSE, 초기값별 평균/표준편차 |

변경된 것은 **예측 목표(10/30/60분)와 그에 따른 출력층 크기**입니다. 단기 전용으로 더 많은 표본을 만들 수 있지만, 장기와의 통제된 비교를 위해 장기 실험에서 사용한 24시간 목표 여유 구간까지 그대로 유지합니다. `split_windows.csv`의 `target_end`는 실제 단기 최대 목표(60분), `comparison_guard_end`는 동일 기준 시각을 유지하기 위한 24시간 경계입니다.

학습 데이터의 시각과 입력은 동일해도 목표 손실함수가 바뀌므로 학습 가중치, 최적 epoch, 선택 초기값은 달라질 수 있습니다. 장기 모델의 1시간 출력과 단기 모델의 60분 출력도 동일할 필요가 없습니다. 60분 정답과 기준 시각은 완전히 같음을 검증합니다.

단기에 유리할 수 있는 10분 전체 시퀀스 입력이나 10분 간격의 추가 학습은 이번 비교에 넣지 않았습니다. 우선 장기 모델과 동일 조건으로 비교하기 위한 실험입니다.

## 실행

`short_term` 폴더에서 검증된 `C:\Python3.9\python.exe` 또는 동일 패키지가 설치된 Python으로 실행합니다.

```powershell
python -m unittest test_forecast -v
python forecast.py --project .. --output runs/new_run
python forecast.py --project .. --output runs/matched_long_term --stage report
python forecast.py --project .. --output runs/matched_long_term --stage predict --as-of "2026-09-21 13:00"
python verify_results.py --project .. --output runs/matched_long_term
python check_parity.py --project .. --output runs/matched_long_term
```

Python3.9, numpy2.0.2, pandas2.2.3, torch2.8.0 CPU, matplotlib3.9.4 및 Pillow를 사용했습니다. 기존 run 덮어쓰기는 금지되어 새 출력 폴더를 지정해야 합니다. `forecast.py`의 기본 출력은 `runs/initial`이므로 위처럼 출력 경로를 명시하는 것을 권장합니다.

## 결과

- `runs/matched_long_term/report.html`: 단기 전체 결과와 contour map
- `runs/matched_long_term/comparison.csv`: 모델·예측시점·실험별 비교
- 각 실험 폴더: 9개 학습 모델, 학습 이력, 세부 평가, 전체 시험 예측, 최신 예측, 공간관계 가중치 및 기간 경계
- `plots/STGNN_10min.png`, `STGNN_30min.png`, `STGNN_60min.png`: 시험 예시 지도
- `verification.json`: 원본 보존·지표 재계산·저장 모델 재실행 검증
- `protocol_comparison.json`: 장기 실험과 조건/기준 시각/60분 정답 일치 검증

Cont​our map은 센서 위치의 모델 예측을 RBF로 보간한 공간 추정입니다. 구획은 근사 경계이고, 단일 센서 구획은 균일 온도로 표시하며 센서 배치 범위 밖은 옅게 표시합니다. 바닥공조·천장형 FCU는 도면 표시이며 풍량/운전정보를 수치 입력으로 가정하지 않습니다. extended 검증25개/시험49개 기준 시각은 적고 연속 관측도 서로 의존하므로 결과의 일반화에는 한계가 있습니다.
