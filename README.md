# 경기도청 ST-GNN 모델

- 단기와 장기 미래로 분류하여 실내 온도를 예측하는 재현 가능한 연구용 파이프라인 
- 최종 결과와 해석은 `REPORT_KO.md`, 수치 원본은 `outputs/metrics.csv`에 있음

## 실행

Python 3.9, CPU PyTorch 2.8.0 환경에서 실행함

```powershell
python -m pip install -r requirements.txt
python pipeline.py --stage preprocess
python -m unittest test_pipeline -v
python pipeline.py --stage train
python pipeline.py --stage plot
python verify_run.py
python build_report.py
```

- 전처리부터 그림까지 한 번에 실행하려면 `python pipeline.py --stage all`을 사용하며, 이후 보고서는 `python build_report.py`로 갱신
- 설정은 `config.json`에 있음
- 다시 실행하면 해당 결과 폴더의 동명 산출물을 갱신
- 원자료는 읽기만 함

## 입력 자료

현재 원자료는 바탕 화면의 `경기도청 데이터` 폴더에서 읽습니다. 이전에 제공된 ZIP 경로는 현재 존재하지 않고 압축이 풀린 폴더가 확인되어 이를 사용했습니다. `source_path`에는 폴더 또는 ZIP을 지정할 수 있습니다. CSV 내용 해시와 파일 경로는 `outputs/audit/file_inventory.csv`에 기록합니다.

- 1차·2차 CSV 23개, 고유 장치 14개를 처리
- 실제 시각을 사용하며 파일명의 날짜는 처리 근거로 사용하지 않음
- 조사 시점이 두 번뿐인 재실인원과 99개 설문 응답은 온도 예측 입력에 사용하지 않음
- CO₂와 이슬점은 이번 예측의 입력·출력에서 제외
- 장비 시간대는 별도 검증하지 못했음
- 원자료의 현지 시각을 그대로 유지

## 전처리

1. 날짜·숫자 파싱 및 파일별 인코딩 확인.
2. 장치번호＋시각 중복 검출. 2차 수집분을 우선하고 1차에만 있는 자료는 보존. 동일 파일 내부 중복은 마지막 행 우선.
4. 온도 -20~60°C 및 RH 0~100% 범위를 벗어난 값은 결측화. 장비 사양이 아니라 보수적인 오류 점검 범위(정상 범위 안의 이상 신호는 자동 제거하지 않음)
5. 정확한 10분 격자에 맞지 않는 이벤트 기록은 별도 보존하고 주 모델에서 제외(미래 방향으로 시각을 옮기거나 평균내지 않았음)
6. 정답은 실제 관측만 사용. 입력만 최대 20분 과거값으로 채우며, 더 긴 결측에는 중립값과 마스크를 사용
7. 평균·표준편차는 9월 7일 이전 실제 관측만으로 계산(센서별이 아니라 변수별 공통 정규화를 사용하여 신규 장치의 짧은 이력에 대한 의존을 줄임)
8. 설치 전·측정 종료 후의 빈 구간은 실제 0값으로 해석하지 않음

## 데이터 분할

| 구간 | 기준 |
|---|---|
| 학습 | 7월 31일~9월 6일 |
| 검증 | 9월 7일~9월 13일 |
| 시험 | 9월 14일~9월 21일 13:00 |

입력은 과거 18시점(현재 포함 18개, 첫 값부터 마지막 값까지 170분), 출력은 1·3·6시점 뒤의 변수임 
최대 60분 뒤 정답이 각 분할을 넘는 윈도는 제외 
검증·시험 시작 시점에서 이전 구간의 실제 과거 입력을 보는 것은 실제 예측과 동일하게 허용하지만 미래 정답은 보지 않음

9월 21일 이후에는 소수 북측 장치만 남기 때문에 주 비교 실험에서 제외하고 `post_primary_period.csv.gz`에 보존
남측 7번은 시험기간에 관측이 없어 시험 성능을 보고하지 않음

## 그래프와 모델

공급된 도면에서 읽은 정규화 좌표로 대칭 3-최근접 그래프를 구성하고 거리 가중치를 적용
좌표는 근사값이며 실측 미터 단위가 아님 
벽체 투과, 실제 기류, 공조 제어구역은 확인되지 않았으므로 그래프에 확정적으로 반영하지 않음
S07과 S11은 도면상 같은 위치 근처지만 서로 다른 노드이며 시간별 관측 마스크로 구분

작은 사용자 정의 ST-GNN을 구현

- 입력 13채널: 정규화 온도·RH, 관측 마스크 2개, 입력 가용성 2개, 결측 경과시간 2개, 일·주기 사인/코사인 4개, 평일 표시 1개.
- 은닉 24채널, 시간 방향 causal dilated convolution(팽창률 1·2·4·8), 관측 가능한 이웃의 가중 메시지 전달, 잔차 연결.
- 최근 관측값에 모델의 변화량을 더해 세 예측시점을 직접 출력
- 학습손실은 실제 정답 마스크를 적용한 표준화 MSE, 조기 종료 및 시드 선택은 검증 표준화 MAE
- AdamW, 최대 60 epoch, 10 epoch 무개선 조기 종료, 시드 17·29·43.

그래프를 단위행렬로 바꾼 같은 구조의 TCN을 비교하여 공간 연결의 효과를 점검합니다. 
지속성, 전일 동시간(없으면 지속성), 개별 센서 과거값을 쓰는 ridge, 이웃 과거정보도 쓰는 ridge를 추가 비교 
모델 선택은 시험 결과 확인 전에 검증자료만으로 고정

## 새 기록으로 예측

```powershell
python predict.py --input data/processed/clean_observations_all_dates.csv.gz --origin "2026-09-21 13:00:00"
```

필수 열은 `timestamp,serial,temperature_c,relative_humidity_pct`임
예측시점 이후 행은 입력 구성 전에 제거
알려진 장치만 지원하며 새 장치는 좌표·그래프 검토와 재검증이 필요
기본 출력은 `outputs/custom_forecast.csv`
이는 사용자가 지정한 과거 또는 현재 기록으로 계산하는 추론 도구이며 센서 스트리밍 연결이나 자동 제어 시스템은 아님

## 주요 산출물

| 경로 | 내용 |
|---|---|
| `data/processed/source_observations.csv.gz` | 파싱된 원자료와 출처 |
| `data/processed/clean_observations_all_dates.csv.gz` | 전체 날짜의 중복·격자 처리 자료 |
| `data/processed/aligned_10min.csv.gz` | 주 실험 10분 정렬 관측·입력·마스크 |
| `data/processed/dataset.npz` | 실제 학습 텐서, 정규화, 그래프 |
| `data/processed/sensor_metadata.csv` | 장치, 도면 좌표, 기간, 관측 개수 |
| `outputs/audit/` | 중복, 시각, 값 범위 등 감사 자료 |
| `models/` | 6개 신경망 체크포인트와 ridge 계수 |
| `outputs/model_selection.json` | 검증점수 기반 선택 기록 |
| `outputs/metrics.csv` | 모델·변수·시간·센서·기간별 MAE/RMSE/편향 |
| `outputs/test_forecasts.csv.gz` | 시험 정답과 예측값 |
| `outputs/paired_day_bootstrap.csv` | 날짜 단위 paired bootstrap 오차차 구간 |
| `outputs/cutoff_forecast.csv` | 9월 21일 13:00을 기준으로 한 과거 시점 예측 예시 |
| `outputs/figures/` | 결과 그림 |
| `REPORT_KO.md` | 수치와 단계별 해석을 포함한 최종 보고서 |

## 해석 범위

하나의 사무공간, 여름~초가을, 한 번의 연속 시간 분할에 대한 결과입니다. 계절·건물 외부 일반화, 정확한 신규 센서 일반화, 인과관계, 제어 효과를 증명하지 않습니다. 공휴일을 제외하지 않은 평일 09~18시는 근무시간의 근사 지표입니다. 날짜 부트스트랩도 시험기간이 약 1주라 불확실성 추정이 제한됩니다. 공간 그래프가 성능을 개선하지 못하면 그 결과를 그대로 보고합니다.
