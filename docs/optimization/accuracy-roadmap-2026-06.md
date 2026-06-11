# 정확도 최적화 로드맵 (2026-06 모색)

> **진행 현황 (2026-06-10)**: P0 완료(평가 하네스 + GT 앵커 4점), P1 **기각**(sdp 변형 효과 ≈ 0 — opt.md 기각 섹션), P2 적용(footprint 역전 수정 + bounds 교정 = opt.md A-5/A-6, fused 모드는 기본 off). 다음 레버: 실측 GT 20~50객체(P0 잔여), P3 GeoCalib pitch 분포, P5 depth 백엔드 A/B.

> **제안서**다. 적용·실측된 항목은 규칙대로 [opt.md](opt.md) 로그에만 기록한다.
> 근거: 코드 직접 검토(`ai/pipeline/*`, `boxer/` 원본) + 기존 실측(opt.md A-4 보정률 89%,
> 실험노트 §5 "병목은 검출이 아니라 기하").

## 진단 요약

| # | 사실 | 근거 |
|---|------|------|
| 1 | 치수 오차의 병목은 검출기가 아니라 **기하(pose/depth)** | 실험노트 §5 (OWLv2 vs SAM3 치수 동률) |
| 2 | sanitizer 보정률 **89%**(bounded) → 기하가 여전히 약하고 prior가 광범위 마스킹 중 | opt.md A-4 실측 |
| 3 | **GT(실측 치수)가 없다** — CSV(`ai/imgs/이삿찜 데이터`)는 개선 전/후 모델 출력 비교일 뿐 | CSV 컬럼 구조 |
| 4 | 래퍼 3D conf threshold **0.2** ≠ 공식 기본 **0.5** | `ai/pipeline/4_boxer.py:61` vs `boxer/run_boxer.py:101` |
| 5 | BoxerNet `prob = 1/(1+σ²)` (aleatoric 불확실성) — hard cut 외 미활용 | `boxer/boxernet/boxernet.py:199` |
| 6 | BoxerNet은 **yaw-only OBB**(중력 정렬 전제) → 카메라 pitch/roll 오차 = 높이↔footprint 누설 | `boxernet.py:189`, GeoCalib 기본 off |
| 7 | depth를 bilinear로 960²에 리사이즈 후 sdp 생성 → 경계 혼합값(flying points) 유입 | `4_boxer.py:294,359` |
| 8 | DA2 폴백 depth는 `max→10m` 정규화라 metric 의미 없음 | `3_depth_estimation.py:105` |
| 9 | estimate 단위 멀티뷰 중복 카운트 미처리, 비큐보이드(curtain/painting) OBB 부피가 견적 왜곡 | `furniture_pipeline.py` (이미지별 독립 처리) |

## 우선순위 (기대효과 ÷ 비용)

### P0. GT 평가셋 + 지표 고정 — 모든 후속 판단의 전제
- **무엇**: 실측 GT 20~50 객체 구축. ① 표준 치수 품목 우선(세탁기·냉장고·매트리스 — 모델명으로 spec 확정) ② 자체 줄자 실측 ③ 이사 실측 DB.
- **어떻게**: `scripts/validate_accuracy.py --manifest` 가 GT 비교·**scale-bias 진단**(geomean(pred/gt))을 이미 지원 — manifest에 `objects`만 채우면 즉시 가동.
- **지표**: per-axis MAE%, volume MAPE, 견적 총부피 오차(사업 지표), (proxy) **raw 보정률**(sanitizer off로 측정 — Goodhart 주의).
- **판정 규칙**: bias≈1 → 잔여 오차는 per-object 기하(P1/P2/P6 경로) / bias≠1 일관 → depth·focal 스케일 문제(P5 경로).

### P1. 저비용 즉효 후보 3건 (각 코드 1~5줄, GT로 전후 측정)
1. **sdp를 원본 해상도 depth에서 직접 샘플링** — 현재 resize(bilinear)→backproject 경로는 경계 혼합값을 만든다. 원본 depth + 원본 intrinsics로 backproject하면 리사이즈 아티팩트가 원천 제거되고 기하적으로 동치(목표 ~3만점, stride 자동 조정). 차선책: `4_boxer.py:294` INTER_LINEAR→INTER_NEAREST.
2. **sdp 밀도 상향** — stride 8(14.4k pts)→4(57.6k). boxer는 16×16 패치당 median이라 포인트가 많을수록 robust, NaN 패치 감소.
3. **3D threshold 정책** — 0.2는 공식(0.5)보다 느슨해 저신뢰 기하가 통과. 단, 컷 상향은 zero-dim 응답 증가(부피 누락) → **P2의 prior 융합과 묶어서**: prob<τ는 0이 아니라 클래스 표준값으로.

### P2. prob 가중 prior 융합 — sanitizer를 이분법에서 연속 융합으로
- **무엇**: 현재 sanitize(통과/clamp/대체) 대신 `dim_final = prob·dim_pred + (1−prob)·dim_prior` 류의 연속 shrinkage. prob는 모델 자신의 aleatoric 불확실성(진단 #5)이므로 가중치로 정당.
- **추가**: 축별 독립 clamp 대신 **aspect-보존 scale 보정** 분기 — 세 축이 비슷한 배율로 벗어났으면 형상은 맞고 스케일만 틀린 것(P0 bias 진단과 동일 논리) → 등비 보정이 per-axis clamp보다 정보 보존.
- **기대**: 보정률 89% 환경에서 prior와 모델 출력의 정보를 모두 살림. 위치: `dimension_bounds.py` + `4_boxer.py:394` (`_to_results`가 prob를 결과에 이미 실어줌).

### P3. GeoCalib 실증 → 기본 on 결정
- **무엇**: 실사용 사진의 pitch/roll 분포 측정(기존 `gravity_estimation.py` 재활용, 이미지당 수십 ms). |pitch|>5° 비율이 유의하면 기본 on.
- **왜**: 진단 #6 — 기울어진 사진에서 수직축 누설은 구조적 오차라 sanitizer로 못 잡는다. 기존 실증은 수평 사진 2장뿐(opt.md A-3 "미완").

### P4. 검출 업그레이드 A/B
- **OWLv2 base→large-patch14-ensemble**: bb2d는 BoxerNet의 쿼리 입력(`boxernet.py:706`) → 박스 품질이 3D 입력 품질. A/B는 `scripts/eval_ab.py` 스냅샷 교체로 수행(구 `compare_detectors.py`는 2026-06-11 제거, git 히스토리 참조).
- **class-aware NMS + 클래스별 threshold**: 최다 보정 클래스(cushion 9·lamp 7·pillow 4)가 과탐·중복 출처와 겹침. 현재 class-agnostic IoU 0.5 단일값(`2_owlv2_2d_detection.py:41`).

### P5. depth 백엔드 A/B — P0에서 bias≠1로 판정될 때
- 후보: **MoGe-2**(metric point map 직접 출력 → sdp 직결 + focal 제공), UniDepth v2, Metric3D v2. `DEPTH_BACKEND` 스위치 구조가 이미 있어 추가 비용 낮음.
- DA2 폴백(진단 #8)은 metric 무의미 → 폴백 시 치수 신뢰 불가 플래그(또는 prior-only 응답) 권장.

### P6. 추론 앙상블 실험 (분산 감소)
- **flip TTA**: 좌우 반전 + bb2d 변환 2-pass, prob 가중 평균. 비용 2×.
- **소형 객체 zoom 재추론**: 최다 보정 클래스가 모두 소형 — 960² 전역 뷰에서 몇 패치에 불과. bbox 1.5~2× 마진 크롭 + intrinsics 보정(cx,cy 이동) 후 재추론. 전역 컨텍스트 상실 영향 측정 필요.

### P7. 견적 단위 정확도 (사업 지표 직결)
- **중복 카운트**: 같은 가구가 여러 사진에 찍히면 총부피 과대. 이미지 간 상대 pose가 없어 boxer `view_fusion`(3D Hungarian)은 직접 사용 불가 — label+치수 유사도 휴리스틱 dedup이 현실적.
- **비큐보이드 정책**: curtain/painting/mirror/rug는 OBB 부피 대신 부피 0 또는 면적×고정두께.

### P8. 장기 — fine-tune (P0~P7로 천장 확인 후)
- 공개 체크포인트는 `hw960in4x6d768` 단일. 한국 주거 도메인 fine-tune은 3D GT 필요 → GT 축적 후 재검토.

## 비용·효과 요약

| 순위 | 항목 | 코드 비용 | 지연 비용 | 기대효과 근거 |
|------|------|-----------|-----------|---------------|
| P0 | GT+지표 | 데이터 작업만 | — | 방향 결정 자체 |
| P1 | sdp 원본 샘플링/밀도/threshold | 1~5줄×3 | ~0 | 진단 #4·#7 |
| P2 | prob 융합 sanitizer | ~50줄 | 0 | 진단 #2·#5 |
| P3 | GeoCalib 기본 on | 측정 후 토글 | +수십 ms | 진단 #6 |
| P4 | OWLv2 large / NMS | 모델명 교체 | +수백 ms | bb2d=쿼리 |
| P5 | depth 교체 | 백엔드 1개 | 모델별 | P0 bias 판정 |
| P6 | TTA/zoom | 실험 스크립트 | ×2 | 소형 객체 보정률 |
| P7 | dedup/비큐보이드 | 후처리 | ~0 | 총부피 지표 |
