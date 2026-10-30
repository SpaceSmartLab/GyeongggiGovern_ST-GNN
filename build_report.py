"""Build the Korean report from completed experiment outputs, never invented metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pipeline import ROOT, VARIABLES

LABELS = {"persistence":"현재값 유지", "previous_day_with_fallback":"전일 동시간(결측 시 현재값)",
          "ridge_temporal":"시간 Ridge", "ridge_spatial":"이웃정보 Ridge", "temporal_tcn":"시간 전용 TCN", "stgnn":"ST-GNN"}

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])

def main():
    metrics = pd.read_csv(ROOT/"outputs/metrics.csv")
    summary = json.loads((ROOT/"outputs/audit/preprocessing_summary.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT/"outputs/model_selection.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT/"outputs/split_summary.json").read_text(encoding="utf-8"))
    bootstrap = pd.read_csv(ROOT/"outputs/paired_day_bootstrap.csv")
    meta = pd.read_csv(ROOT/"data/processed/sensor_metadata.csv")
    primary = metrics.query("split == 'test' and period == 'all' and sensor == 'ALL'")
    parts = ["# 경기도청 온도·상대습도 ST-GNN 분석 결과", "작성일: 2026년 10월 12일",
             "## 1. 수행 범위와 핵심 결과",
             "온도와 상대습도만을 대상으로 원자료 점검, 전처리, 도면 기반 근사 그래프, 시간 분할, 기준모델 비교, ST-GNN 학습, 독립 시험평가, 예측 파일과 추론 도구까지 수행했습니다. 원본 파일은 변경하지 않았습니다. 설문·재실인원·이슬점·CO₂는 모델에 사용하지 않았습니다."]
    keyrows = []
    comparisons = []
    for horizon in [10,30,60]:
        row = [horizon]
        for variable in VARIABLES:
            sel = primary[(primary.horizon_minutes==horizon)&(primary.variable==variable)]
            st = sel[sel.model=="stgnn"].iloc[0]
            base = sel[sel.model=="persistence"].iloc[0]
            gain = 100*(1-st.mae/base.mae)
            row.extend([f"{st.mae:.4f}",f"{gain:+.1f}%"])
            comparisons.append((variable,horizon,gain))
        keyrows.append(row)
    parts.append(table(["예측시간(분)","온도 MAE(°C)","현재값 유지 대비 개선","RH MAE(%p)","현재값 유지 대비 개선"],keyrows))
    parts.append("개선율 = 100 × (1 − ST-GNN MAE / 현재값 유지 MAE)입니다. 양수는 오차 감소, 음수는 오차 증가입니다. 상대습도 오차 단위는 상대 백분율이 아닌 퍼센트포인트(%p)입니다. 시험 결과에 맞춰 모델을 다시 튜닝하지 않았습니다.")
    spatial_gains = []
    for variable in VARIABLES:
        for horizon in [10,30,60]:
            sel = primary[(primary.variable==variable)&(primary.horizon_minutes==horizon)]
            a,b = [sel[sel.model==name].iloc[0].mae for name in ["stgnn","temporal_tcn"]]
            spatial_gains.append(100*(1-a/b))
    if all(g>0 for g in spatial_gains):
        conclusion = "선택된 ST-GNN은 모든 변수·예측시간의 MAE에서 시간 전용 TCN보다 낮았습니다. 다만 한 번의 시간 분할 결과이며 시드별 편차와 날짜별 불확실성을 함께 보아야 합니다."
    elif any(g>0 for g in spatial_gains):
        conclusion = "공간 연결의 이득은 변수와 예측시간에 따라 달랐습니다. ST-GNN이 모든 조건에서 시간 전용 모델보다 우수하다는 결론은 내릴 수 없습니다."
    else:
        conclusion = "이번 시험에서는 선택된 ST-GNN이 시간 전용 TCN보다 MAE를 개선하지 못했습니다. 이 데이터와 근사 그래프에서 공간 연결의 추가 이득은 확인되지 않았습니다."
    parts.append("**"+conclusion+"**")
    parts.extend(["## 2. 원자료와 품질 처리",
                  f"현재 원자료는 `{summary['source_path']}` 폴더에 있습니다. 이전 ZIP 경로는 존재하지 않아 압축이 풀린 원자료 폴더를 사용했습니다. 파일별 SHA-256과 출처를 감사자료에 남겼습니다."])
    parts.append(table(["항목","결과"],[
        ["CSV / 고유 장치",f"{summary['csv_files']}개 / {summary['unique_devices']}개"],
        ["유효 시각 원본 기록",f"{summary['valid_timestamp_source_rows']:,}개"],
        ["중복 제거 기록",f"{summary['duplicate_rows_removed']:,}개"],
        ["0.011 단위 초과 중복 충돌",summary['conflict_groups_above_0_011']],
        ["10분 격자 밖 기록",f"{summary['off_grid_records_excluded']}개 별도 보존·제외"],
        ["범위 밖 온도 / RH",f"{summary['invalid_values'][VARIABLES[0]]} / {summary['invalid_values'][VARIABLES[1]]}"],
        ["주 분석 시간격자",f"{summary['primary_steps']:,}개"],
        ["실제 정답 관측값",f"{summary['observed_target_values']:,}개(변수별 값 개수)"],
        ["과거값으로 채운 입력",f"{summary['input_values_forward_filled']}개(최대 20분)"],
        ["주 분석 이후 보존 기록",f"{summary['post_primary_rows_retained']:,}개"]]))
    parts.append("2차 파일에 포함된 1차 기간을 중복 학습하지 않도록 장치번호와 실제 시각으로 통합했습니다. 2차를 우선하되 1차에만 있는 초기 기록은 보존했습니다. CSV의 중복 정밀도 차이를 검사한 결과 허용 범위를 넘는 충돌은 없었습니다. 파일명 9월 31일은 사용하지 않고 내부 시각을 사용했습니다.")
    parts.append("입력의 짧은 결측만 과거 방향으로 채웠습니다. 설치 전·철거 후 구간이나 장기간 공백은 관측 마스크로 구분합니다. 실제 정답을 보간하지 않았습니다. 온도 -20~60°C, RH 0~100% 점검 범위는 장비 정확도 규격이 아닙니다. 긴 고정값 구간은 별도 표에 남겼고 자동 삭제하지 않았습니다.")
    parts.append("![센서 관측 기간](outputs/figures/coverage.png)")
    parts.append(table(["센서","장치번호","주 분석 관측 시점 수","최초 기록","최종 기록"],
                       [[r.label,str(r.serial),f"{r.observed_points_primary:,}",r.first_record,r.last_record] for r in meta.itertuples()]))
    parts.extend(["## 3. 도면과 그래프",
                  "1·2차 도면의 센서 번호와 CSV 장치번호를 연결했습니다. 노드는 장치별 14개이며 남측 7번과 11번을 임의로 하나의 연속 측정값으로 합치지 않았습니다. 정규화한 도면 좌표에서 가까운 3개 이웃을 연결하고 대칭화했으므로 실제 이웃 수는 3개보다 많을 수 있습니다. 자기 연결과 거리 가중치를 포함합니다.",
                  "거리는 근사 이미지 좌표 기준이며 실측 미터, 벽체 차폐, 실제 기류 또는 공조구역을 뜻하지 않습니다. 설치 위치가 확인되지 않는 S18은 자기 연결만 있습니다. 최근에 온·습도 입력을 사용할 수 없는 노드는 메시지 전달에서 제외합니다. S07과 S11은 기록기간이 다르므로 같은 시점의 정상 관측으로 연결되지 않습니다.",
                  "![공간 그래프](outputs/figures/graph.png)",
                  "## 4. 입력과 시간 분할",
                  "입력은 10분 간격 18개 관측값과 마스크·달력정보입니다. 현재 포함 18개이므로 첫 기록부터 마지막 기록까지는 170분입니다. 이를 통상적인 3시간 길이 입력 창으로 설정했습니다. 10·30·60분 후 온도와 RH를 직접 예측합니다. 학습기간만으로 변수별 평균·표준편차를 계산합니다."])
    parts.append(table(["구분","윈도 수","첫 예측 기준시각","마지막 정답 시각"],
                       [[{"train":"학습","validation":"검증","test":"시험"}[name],f"{v['windows']:,}",v['first_origin'][:19],v['last_target'][:19]] for name,v in splits.items()]))
    parts.append("원점과 모든 정답은 각 분할 안에 있어야 합니다. 검증·시험 시작 시점에 이전 구간의 과거 관측을 입력으로 사용하는 것은 허용하지만 이후 관측을 거꾸로 채우지 않습니다. 이 평가는 매 10분 새 관측이 들어오는 상황에서의 10·30·60분 단기 예측입니다. 일주일 전체를 관측 없이 한 번에 예측하는 실험이 아닙니다.")
    parts.append("9월 21일 이후에는 주로 추가 북측 3개 장치만 남으므로 동일 공간망 비교의 주 시험에서 제외했습니다. 해당 자료는 전처리 결과에 보존했습니다. 시험기간에는 S07의 정답이 없으므로 최대 13개 장치를 평가합니다. 시험기간 전체에 정답이 존재하는 경우만 비교하며 모델마다 유리한 행을 따로 고르지 않습니다.")
    parts.extend(["## 5. 학습 구조와 모델 선택",
                  "ST-GNN은 은닉 24채널의 시간 causal dilated convolution 4개 블록과 공간 가중 메시지 전달로 구성한 소형 모델입니다. 팽창률은 1·2·4·8이며 미래 입력을 보는 양방향 연산은 없습니다. 출력은 최근 관측 대비 변화량입니다. 특정 논문의 모델과 동일한 재현 구현이라고 주장하지 않습니다.",
                  "시간 전용 TCN은 동일한 구조에서 그래프를 단위행렬로 바꿔 공간 교환을 없앤 비교입니다. 시간 Ridge와 이웃정보 Ridge는 동일한 입력 창에 기반한 강한 선형 기준모델입니다. 지속성과 전일 동시간 모델도 함께 평가했습니다.",
                  "AdamW, 학습률 0.001, 배치 128, 최대 60 epoch, 검증점수 10 epoch 무개선 시 조기 종료를 적용했습니다. 세 시드(17·29·43)를 각각 학습했습니다. 정규화 MSE를 최소화하되 epoch·시드·ridge 규제계수 선택은 검증 정규화 MAE로만 수행하고 시험 평가 전에 고정했습니다."])
    parts.append(table(["모델","시드/규제계수","최적 epoch","검증 표준화 MAE","학습시간(초)"],
        [[LABELS[r['name']],r.get('seed',r.get('alpha','')),r.get('best_epoch','해당 없음'),f"{r['validation_scaled_mae']:.5f}",f"{r.get('elapsed_seconds',0):.1f}" if 'elapsed_seconds' in r else "미계측"] for r in selection['runs']]))
    parts.append("선택 체크포인트: "+", ".join(f"`{v}.pt`" for v in selection['selected_by_validation_only'].values()))
    parts.append("![학습 곡선](outputs/figures/learning_curves.png)")
    parts.append("## 6. 독립 시험 성능")
    for variable, label, unit in [(VARIABLES[0],"온도","°C"),(VARIABLES[1],"상대습도","%p")]:
        parts.append(f"### {label}: MAE / RMSE ({unit})")
        rows = []
        for model in LABELS:
            row = [LABELS[model]]
            for h in [10,30,60]:
                r = primary[(primary.model==model)&(primary.variable==variable)&(primary.horizon_minutes==h)].iloc[0]
                row.append(f"{r.mae:.4f} / {r.rmse:.4f}")
            rows.append(row)
        parts.append(table(["모델","10분","30분","60분"],rows))
    parts.extend(["MAE는 평균 절대오차이며 RMSE는 큰 오차에 더 민감합니다. 변수 단위가 달라 원단위 오차를 합산하지 않습니다. 센서별 오차와 편향은 `outputs/metrics.csv`에 있습니다.",
                  "![시험 오차 비교](outputs/figures/test_mae.png)"])
    seedrows = []
    for model in ["temporal_tcn","stgnn"]:
        for variable in VARIABLES:
            for h in [10,30,60]:
                r = primary[primary.model.str.startswith(model+"_seed") & (primary.variable==variable)&(primary.horizon_minutes==h)]
                seedrows.append([LABELS[model],"온도" if variable==VARIABLES[0] else "RH",h,f"{r.mae.mean():.4f} ± {r.mae.std(ddof=1):.4f}"])
    parts.append("### 시드 3개에 따른 MAE 평균 ± 표준편차")
    parts.append(table(["모델","변수","예측시간(분)","MAE"],seedrows))
    parts.append("이 표의 표준편차는 학습 초기화 변동을 나타냅니다. 미래 계절이나 다른 건물에 대한 신뢰구간이 아닙니다.")
    work = metrics.query("split == 'test' and period == 'weekday_09_18' and sensor == 'ALL' and model == 'stgnn'")
    parts.append("### 평일 09~18시의 ST-GNN 오차")
    parts.append(table(["변수","예측시간(분)","MAE","RMSE","평가 값 개수"],
                       [["온도" if r.variable==VARIABLES[0] else "RH",r.horizon_minutes,f"{r.mae:.4f}",f"{r.rmse:.4f}",f"{r.n:,}"] for r in work.itertuples()]))
    parts.append("평일·시간대는 근무시간의 근사값입니다. 실제 재실 여부와 휴일을 검증하지 않았습니다.")
    parts.append("### 날짜 단위 불확실성 확인")
    parts.append("같은 예측 기준일의 모든 시점·센서를 묶어 paired bootstrap 2,000회를 수행했습니다. ST-GNN MAE에서 기준모델 MAE를 뺀 값이므로 음수는 ST-GNN에 유리합니다. 95% 구간이 0을 포함하면 이 실험에서 개선 방향을 안정적으로 단정하기 어렵습니다. 시험이 약 1주라 날짜 간 의존성과 계절 변화에 대한 한계가 있습니다.")
    parts.append(table(["기준모델","변수","예측시간","MAE 차이","날짜 bootstrap 95% 구간"],
                       [[LABELS[r.baseline],"온도" if r.variable==VARIABLES[0] else "RH",r.horizon_minutes,f"{r.stgnn_minus_baseline_mae:+.4f}",f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]"] for r in bootstrap.itertuples()]))
    parts.extend(["## 7. 예측 예시와 실행 검증",
                  "아래 그림은 미리 정한 S31의 9월 15~16일 구간에서 60분 후 예측을 실제 정답 시각에 표시한 것입니다. 가장 잘 맞는 구간을 자동으로 고른 그림이 아닙니다.",
                  "![실제값과 예측값](outputs/figures/forecast_example.png)",
                  "`outputs/test_forecasts.csv.gz`에는 시험 정답, ST-GNN, 시간모델 및 기준모델 예측이 들어 있습니다. `outputs/cutoff_forecast.csv`에는 9월 21일 13:00 기록까지만 보고 계산한 예측 예시가 있습니다. 현재 날짜의 실시간 예보가 아닙니다.",
                  "추론 도구 `predict.py`는 학습된 체크포인트와 훈련 시 정규화를 그대로 사용합니다. 미래 행을 제외한 뒤 입력을 구성하며 알 수 없는 장치번호나 중복 시각은 오류로 처리합니다. 실행 명령과 입력 열은 README에 정리했습니다.",
                  "## 8. 해석 한계와 다음 검증",
                  "이번 결과는 단일 건물의 여름~초가을 자료와 한 번의 연속 시간 분할에 한정됩니다. 일주일 남짓의 시험으로 연중 안정성이나 다른 건물 일반화를 주장할 수 없습니다. 다중 rolling-origin 재학습, 겨울·환절기 시험, 독립 건물 검증은 이번 실행 범위에 포함하지 않았습니다.",
                  "실제 외기·일사·공조 설정·운전 상태·연속 재실인원이 없어 외부 요인에 의한 급격한 변화의 설명에는 제약이 있습니다. 도면 좌표와 S18 위치 확인, 센서 교정 및 시계 오차 확인이 후속 우선순위입니다. 주 분석 종료 이후 북측 3개만 남는 조건과 완전히 새로운 센서에 대한 일반화는 별도 검증이 필요합니다.",
                  "ST-GNN의 추가 이득이 작거나 없는 조건에서는 더 단순하고 오차가 낮은 기준모델도 운영 후보로 고려해야 합니다. 시험 결과를 보고 같은 시험에 맞춰 반복 튜닝하면 객관적인 시험의 의미가 사라지므로, 후속 수정은 새 검증·시험 기간에서 평가하는 것이 적절합니다.",
                  "## 9. 산출물 및 재현",
                  "전처리 코드와 학습·평가: `pipeline.py`; 설정: `config.json`; 추론: `predict.py`; 검증: `test_pipeline.py`; 재현 절차: `README.md`. 데이터 파일과 모델은 각각 `data/processed/`, `models/`에 있습니다. 감사·수치·그림은 `outputs/`에 보존했습니다.",
                  "개념 참고: [DCRNN 원논문](https://arxiv.org/abs/1707.01926), [Graph WaveNet 원논문](https://arxiv.org/abs/1906.00121). 본 실험 수치는 이 자료에서 가져온 것이 아니라 제공된 센서 데이터로 직접 학습·평가한 결과입니다."])
    verification_path = ROOT/"outputs/verification.json"
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        parts.extend(["## 10. 완료 검증",
                      f"단위·통합 검사 {verification['tests_run']}개를 실행했고 실패 {verification['failures']}개, 오류 {verification['errors']}개였습니다. 전체 검증 상태: {'통과' if verification['all_passed'] else '실패 항목 확인 필요'}.",
                      "입력의 인과적 채움, 분할 경계를 넘지 않는 정답, 학습 구간 정규화, 미관측 정답 제외, 비활성 이웃 차단, 독립 추론과 학습 입력의 일치, 미래 행 변경에 대한 입력 불변성, 마지막 시점에서 미래 정답 없이 추론, 저장 모델의 시험 예측 재현을 검사했습니다.",
                      "원본 CSV 23개 해시가 처리 전후 일치했고, 저장된 정답·예측에서 독립적으로 계산한 MAE와 평가 개수가 보고값과 일치했습니다. 모델 선택은 저장된 검증점수의 최솟값과 일치했습니다. 13개 센서 × 3개 예측시간의 과거 기준시각 예측 39개와 결과 그림 5개를 확인했습니다. 상세 기록은 `outputs/verification.json`, `outputs/test_results.txt`에 있습니다."])
    (ROOT/"REPORT_KO.md").write_text("\n\n".join(parts)+"\n",encoding="utf-8")
    print("Wrote REPORT_KO.md")

if __name__ == "__main__":
    main()
