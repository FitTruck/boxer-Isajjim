# 실험 노트: 2D 검출기(OWLv2 vs SAM3) & 단일 이미지 3D 기하

> 브랜치 `experiment/owlv2-vs-sam3`. 로컬 검증 환경: Apple M4 / MPS, transformers 5.9,
> torch 2.12. 테스트 이미지 2장(거실 사진 `ai/imgs/img.png` 5701×3801, `img_1.png` 720×480).
> 파이프라인: OWLv2(또는 SAM3) → Depth Pro → BoxerNet 3D OBB.

## 동기

callback 치수(WxDxH)가 비현실적으로 나오는 문제. "검출기(OWLv2)를 SAM3로 바꾸면 나아질까?"를
검증하려다, **오차의 진짜 출처가 검출이 아니라 3D 기하**임을 발견.

## 핵심 발견

### 1. (주범) 단일 이미지 lift에 identity pose를 넘기고 있었음 — 수정됨
- BoxerNet은 world<-cam pose에서 중력을 유도(`gravity_align_T_world_cam`, world gravity = -Z).
- 기존 `_identity_pose()` → 모델이 "카메라가 중력축을 따라 천장/바닥을 본다"고 가정 → 박스 방향 붕괴.
- 수정: Omni3D/SUN-RGBD 단일 이미지 기본값 `R_yz_swap = [[1,0,0],[0,0,1],[0,-1,0]]`
  (수평 정면 카메라 + 중력 아래)를 pose로 넘기고, depth 포인트도 같은 회전 적용.
- **실측 효과 (img.png 소파):** `3812×262×1010mm`(깊이 26cm, 불가능) → `2628×703×1094mm`(현실적).
  의자 높이 `1609 → 579mm`.
- commit `bb65c6e`.

### 2. focal 오추정 폴백 — 수정됨
- Depth Pro가 가끔 비정상 focal(매우 광각/협각) 반환 → metric 스케일 붕괴.
- `focal/width`가 [0.5, 2.5] 밖이면 기본 FOV로 폴백. commit `bb65c6e`.

### 3. GeoCalib per-image gravity + focal — opt-in 추가
- 단일 이미지에서 카메라 중력/focal 추정(논문이 권장). `--geocalib` 플래그.
- **두 테스트 이미지 모두 GeoCalib가 "거의 수평"으로 판정**(pitch 2.5° / -0.1°) → R_yz_swap 기본값이
  옳았음을 검증. 레벨 이미지에선 R_yz_swap과 결과 동일(회귀 없음).
- 가치: 기울어진 사진 + 신뢰할 focal. 단 본 테스트 2장이 다 수평이라 추가 이득 실증은 미완.
- commit `1318756`.

### 4. (기각) square-resize 왜곡 가설
- "960×960 정사각 스트레치가 비정사각 이미지를 왜곡"이라 의심했으나, **공식 `run_boxer.py`도
  동일하게 square resize + per-axis intrinsic 보정**(`loader.resize = boxernet.hw`). 레퍼런스와 일치
  → 버그 아님. 구현하지 않음.

### 5. OWLv2 vs SAM3 A/B (동일 depth + BoxerNet)

| 항목 | OWLv2 | SAM3 |
|------|-------|------|
| 검출 수 (img / img_1) | 14 / 18 (conf 0.25~0.78, 쿠션·커튼 과탐) | 4 / 2 (conf 0.96~0.98, 오탐 없음) |
| 같은 객체의 3D 치수 | — | **OWLv2와 거의 동일** |
| 속도 (일반 해상도) | 0.2~1.6s | ~20~30s (컨셉 19개 루프) |

- **치수: 동률.** sofa/chair/tv/shelf 모두 두 검출기에서 ±몇 % 이내 동일. → SAM3 tight mask가
  3D 치수를 개선하지 **않음**. 가설 기각.
- **img_1 소파: SAM3의 0.98 깨끗한 박스로도 높이 1.65m(틀림)** → 남은 오차는 검출이 아니라
  depth/BoxerNet 확정.
- **SAM3 장점**: precision(오탐 억제, 고신뢰). **단점**: recall 낮음, 30~100배 느림.

## 결론 / 권장

1. 치수 정확도의 병목은 **검출기가 아니라 기하(pose/depth)**. gravity-pose 수정이 핵심 레버였음.
2. 프로덕션: **OWLv2 유지 + 후처리**(confidence threshold 상향 + NMS 강화)로 과탐만 정리.
   SAM3는 정밀도가 critical하고 지연 여유가 있을 때(오프라인/배치)만.
3. 다음 정확도 레버: **depth 품질**(Depth Pro 검증/대안), 기울어진 사진용 **GeoCalib 실증**.

## 재현

```bash
# 검출기 단계별 진단 + 오차 출처 분리
python scripts/validate_accuracy.py --manifest scripts/local_test_manifest.json [--geocalib]
# OWLv2 vs SAM3 head-to-head (동일 depth+BoxerNet)
python scripts/compare_detectors.py --manifest scripts/local_test_manifest.json
```
