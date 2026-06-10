# 최적화 기록 (boxer-Isajjim)

> 원본 [facebookresearch/boxer](https://github.com/facebookresearch/boxer) 대비 **팀이 설계·추가한 최적화만** 기록한다.
> `boxer/` 서브디렉토리(Meta 원본)는 대상이 아니며, `ai/`·`api/`의 변경만 기록한다.
> 배경: 원본 boxer는 **posed 멀티뷰/egocentric(known pose·LiDAR)** 용 → 팀은 이를
> **"단일 업로드 가구 사진"** 에 맞게 적응시켰다. 그래서 팀 최적화의 무게중심은
> **속도가 아니라 3D 기하 정확도(현실적 mm 치수)** 에 있다.

## 유지 규칙 (IMPORTANT — 반드시 지킬 것)

1. **새 최적화를 적용/발견할 때마다 아래 "최적화 로그"에 항목을 추가**한다. (이 파일이 유일한 단일 기록처)
2. 각 항목은 맨 아래 **[새 항목 템플릿]** 을 따른다: 무엇 / 원본 대비 변경 / 실측 효과(before→after) / 근거(commit·file) / 날짜 / 분류.
3. **실측 수치를 가능한 한 포함**한다. 측정 전이면 `측정 전`이라 명시한다.
4. 시도했다가 **기각한 가설도 "제외/기각" 섹션**에 남겨 재시도를 막는다.
5. 분류는 A(기하/정확도) · B(검출) · C(서빙/처리량) 중 하나 이상.

## 분류

- **A. 3D 기하/정확도** — 단일 이미지 monocular lift 품질 (pose·focal·depth·치수)
- **B. 검출** — detector 선택 / 어휘 / 후처리
- **C. 서빙/처리량** — GPU 풀 · 동시성 · I/O

---

## 최적화 로그

### A. 3D 기하 / 정확도

#### A-1. 중력 정렬 pose (★최대 레버)
- **무엇**: BoxerNet에 넘기는 카메라 pose를 identity → 수평 카메라 기본값으로 교체.
- **원본 대비**: 기존엔 `_identity_pose()`를 넘겨 모델이 "카메라가 중력축을 따라 천장/바닥을 본다"고 오해 → 박스 방향 붕괴. Omni3D/SUN-RGBD 단일 이미지 기본값 `R_yz_swap=[[1,0,0],[0,0,1],[0,-1,0]]`(수평 정면 + 중력 -Z)으로 교체하고 depth 포인트(sdp)도 동일 회전 적용.
- **실측 효과**: 소파 `3812×262×1010mm`(깊이 26cm, 불가능) → `2628×703×1094mm`(현실적). 의자 높이 `1609 → 579mm`.
- **근거**: `ai/pipeline/4_boxer.py`, commit `bb65c6e`, tdd §12.1
- **날짜**: 2026-05 · **분류**: A

#### A-2. focal 오추정 폴백
- **무엇**: Depth Pro가 비정상 focal(초광각/협각) 반환 시 기본 FOV로 폴백.
- **원본 대비**: `focal/width`가 `[0.5, 2.5]` 밖이면 `focal_length_px=None` → `fx=fy=target×0.75` 기본 핀홀 사용. metric 스케일 붕괴 방지.
- **실측 효과**: 비정상 focal로 인한 스케일 붕괴 사례 차단(정성).
- **근거**: `ai/pipeline/4_boxer.py`, commit `bb65c6e`, tdd §12.2
- **날짜**: 2026-05 · **분류**: A

#### A-3. GeoCalib per-image 중력 + focal (opt-in)
- **무엇**: 기울어진 사진용으로 단일 이미지에서 카메라 중력/focal을 GeoCalib(Veicht 2024)로 추정해 `gravity_down_cam`으로 전달.
- **원본 대비**: 기본 **off**(`--geocalib`로 on) — 서버 기본 경로엔 의존성·지연 추가 없음. 레벨 이미지에선 A-1 기본값으로 정확히 환원됨을 수식 검증.
- **실측 효과**: 테스트 2장 모두 "거의 수평"(pitch 2.5°/-0.1°) 판정 → 기본값이 옳았음을 검증. 기울어진 사진 이득 실증은 **미완**.
- **근거**: `ai/pipeline/gravity_estimation.py`, commit `1318756`, tdd §12.3
- **날짜**: 2026-05 · **분류**: A

#### A-4. 클래스별 치수 sanity 보정
- **무엇**: BoxerNet이 물리적으로 불가능한 치수를 낼 때 클래스별 물리 범위로 clamp/대체.
- **원본 대비**: OWLv2 어휘 95종 전부 커버(91 bounded + 비큐보이드 8종 통과). 범위 안→유지, 살짝 벗어남→가까운 경계 clamp, 심함(`<0.5·min` 또는 `>2·max`)→클래스 표준값(범위 중앙) 대체. 수평축(width/depth)은 long/short 정렬 후 footprint 검사, 수직축(height)은 중력 기준 그대로. 한국 규격 반영(매트리스~2000mm, 장롱 자/303mm, KS 책상 ~700-750mm). 모든 보정은 `[sanitize]` 로그.
- **실측 효과**: 침대 `2062×1372×2375` → `2062×1372×450mm`, 책상 높이 `1381 → 760mm`.
- **보정률(실측, 2026-06-01, 로컬 5장/63객체, Mac MPS)**: **객체 78%**(49/63 전체) / **89%**(49/55 bounded)가 ≥1축 보정, **축 보정률 49.7%**(82/165). 유형: clamp(경미) 68 / severe→typical(심각) 14. 최다 보정 클래스: cushion 9·lamp 7·pillow 4·drawer 4(소형 연성 객체). ⚠ **89%는 매우 높은 보정률** → monocular depth/geometry 자체가 여전히 약하고 sanitizer가 광범위하게 마스킹 중이라는 신호(tdd §12.5의 경고가 실측으로 확인됨). 표본 작음(n=63, 실내 가구 밀집 이미지 편향) 주의.
- **재측정(2026-06-10, 동일 5장·옛 bounds·실모델 재추론)**: 객체 보정률 **75.8%**(47/62 bounded), 축 45.2%, severe 객체 6.5%. 89%→76 하락의 유력 원인은 6/9 OBB 수직축 라벨링 수정(`a323388`) — 단 분모 변화(55→62 bounded)·검출 변동이 섞여 있고 6/1 raw 기록이 없어 엄밀 분해는 불가. 최다 보정은 여전히 cushion 8·curtain 7·pillow 6(소형·연성) → **기하가 주 병목이라는 결론 유효**. `scripts/correction_rate.py`로 재현.
- **근거**: `ai/pipeline/dimension_bounds.py`, commit `ec9f6d3`/`7ed1f54`, tdd §12.5. `SANITIZE_DIMENSIONS` env(기본 on)로 토글.
- **한계**: prior 보정이라 부피 견적엔 실용적이나 개별 "측정" 정확도가 오른 건 아님. 보정률↑ = depth/geometry 손봐야 한다는 신호.
- **날짜**: 2026-05 · **분류**: A

#### A-5. sanitizer footprint 역전 버그 수정 + prob 가중 fused 모드(opt-in)
- **무엇**: ① long/short 독립 보정이 footprint 순서를 역전시키던 버그 수정(예: recliner short severe→typical 950mm > long 720mm — 적대적 설계검토에서 실행 재현). ② `SANITIZE_MODE=fused`: 범위 밖 축을 BoxerNet aleatoric prob(=1/(1+σ²))로 경계↔클래스 typical 사이 log-공간 연속 보간(`lam=α^(1+2(1−p))`, α=1−log(위반배율)/log2 — 경계 연속·2× 위반 시 typical 수렴), 3축 동방향·유사배율 위반은 aspect 보존 등비보정. **기본값 clamp(기존 동작 유지)**.
- **원본 대비**: 신규(A-4 확장). 초기 구현의 경계 불연속 결함(prob=0.5에서 −139mm 점프)은 코드리뷰에서 발견·수정, 연속성 테스트로 고정.
- **실측 효과**: 외부 표준 GT 앵커 3점(삼성 세탁기/건조기 spec, 65" TV)에서 fused가 clamp 대비 축 log-MAE 0.143→0.138, 부피 MRE 0.581→0.543. ⚠ n=3은 일화 수준 → 실측 GT 20+ 확보 후 기본 전환 재검토.
- **근거**: `ai/pipeline/dimension_bounds.py`, `tests/test_dimension_bounds_fused.py`, `scripts/eval_sanitizer.py`, commit `be2687c`
- **날짜**: 2026-06-10 · **분류**: A

#### A-6. 클래스 bounds 교정 — 대형 가전·TV가 정답을 깎던 구간 제거
- **무엇**: automatic washer/washing machine long 상한 700→830(삼성 그랑데 21kg 드럼 796mm), washer dryer 720→880(히트펌프 건조기 844mm), tv/television set height 800→950·long 1800→1950(65" 패널 1450×830, Danawa spec).
- **원본 대비**: A-4 bounds 테이블의 데이터 교정. GT 앵커 실측이 "모델이 맞혀도 sanitizer가 틀리게 만드는" 구간을 드러냄.
- **실측 효과**: A-4 동일조건 보정률 75.8%→74.2%(−1.6%p). 건조기 폭 예측 737mm가 720 clamp 대신 통과(|log err| 0.159→0.136).
- **근거**: `ai/pipeline/dimension_bounds.py`, `scripts/correction_rate.py`, commit `5b0d2b5`
- **날짜**: 2026-06-10 · **분류**: A

### B. 검출

#### B-1. OWLv2 채택 + curated LVIS+ 95종 어휘 (vs SAM3, A/B 검증)
- **무엇**: 검출기로 OWLv2 + 가구 특화 LVIS+ 95종 어휘 유지. `DETECTOR_BACKEND=owlv2|sam3` 스위치로 SAM3는 opt-in.
- **원본 대비**: 통합 단계 신규 설계. chunked 프롬프트(chunk 256) + chunk 간 NMS 병합.
- **실측 효과(A/B, 동일 depth+BoxerNet)**: SAM3 대비 **3D 치수 사실상 동률**인데 OWLv2가 **30~100× 빠름**(0.2~1.6s vs 20~30s). SAM3는 precision↑·고신뢰지만 recall↓·느림 → 프로덕션은 OWLv2 유지(+threshold/NMS 후처리).
- **핵심 발견**: 치수 정확도 병목은 **검출기가 아니라 기하(pose/depth)** → A-1이 진짜 레버.
- **근거**: `ai/pipeline/2_owlv2_2d_detection.py`, `sam3_detection.py`, commit `991a493`/`dd0a1ea`, 실험노트 §5
- **날짜**: 2026-05 · **분류**: B

### C. 서빙 / 처리량

#### C-1. GPU 풀 — no-polling 세마포어 스케줄링 + 멀티-GPU 병렬
- **무엇**: GPU 리소스 풀을 세마포어로 관리, 폴링 sleep 제거.
- **원본 대비**: 원본 boxer엔 서빙 계층 없음(신규). 대기 코루틴이 즉시 dispatch → 큐잉 latency 사실상 0. 라운드로빈으로 이미지 N장을 N개 디바이스에 진짜 병렬(`asyncio.gather`). startup 시 모델 사전 로드.
- **실측 효과**: acquire/release **~50ms** (기존 폴링 구현이면 ≥500ms로 반올림됐을 것 — `tests/test_gpu_pool.py`로 검증).
- **근거**: `ai/gpu/gpu_pool_manager.py:11`, commit `b48a48a`
- **날짜**: 2026-05 · **분류**: C

#### C-2. process_pil 워커 스레드 offload
- **무엇**: 블로킹 GPU 파이프라인(`process_pil`)을 워커 스레드로 분리.
- **원본 대비**: 이벤트 루프 비블로킹, Python GIL 이슈를 GPU 분산으로 우회.
- **실측 효과**: 측정 전(동시 요청 처리량 개선 목적).
- **근거**: commit `1ecc6d7`
- **날짜**: 2026-05 · **분류**: C

---

## 참고: 미적용 / 측정만 한 항목

- **Depth Pro 단계가 파이프라인 최대 병목** — MPS 직접 실측 `~30s/img`(median 30.1s, 952M 파라미터, 추론 피크 ~9GB). CUDA(V100)면 0.3s라 ~100× MPS 페널티. 속도 최적화 시 1순위 대상. (실측: 이 Mac MPS, 2026-06-01)
- **외부 표준 GT 앵커 4점 확보(2026-06-10)** — `scripts/eval_manifest.json`: 33.png 매트리스 K 1600×2000(사진 캡션 명시), 36.png 삼성 세탁기 796×686×984·건조기 844×686×984(제조사 spec), img_1.png 65" TV 1450×60×830. dimension_bounds prior와 독립인 측정 기준점. 진짜 정확도 검증엔 실측 GT 20~50객체가 필요(최우선 과제).
- **⚠ 데이터 이슈: `ai/imgs/31.png` ≡ `35.png`(md5 동일)** — '이삿찜 데이터' CSV의 1번/3번 사진이 같은 파일. CSV는 GT가 아니라 개선 전/후 모델 출력 비교표임. 평가셋 수집 과정 점검 필요.
- **평가 인프라(2026-06-10)** — `scripts/eval_ab.py`(검출·depth 캐시 paired A/B, A/A 노이즈 0.00mm 검증), `analyze_ablation.py`, `eval_sanitizer.py`, `correction_rate.py`. 실험 정밀도는 MPS fp32 — 프로덕션(T4 fp16/L4 bf16) 채택 전 GPU 서버 재검증 필요.

## 제외 / 기각 (최적화 아님 — 재시도 금지)

- **square-resize 960² 왜곡 가설 (기각)**: 비정사각 스트레치를 의심했으나 공식 `run_boxer.py`도 동일하게 square resize + per-axis intrinsic 보정 → 레퍼런스와 일치, 버그 아님. 구현 안 함. (실험노트 §4)
- **bf16 autocast / 단일 forward batching**: 원본 boxer 추론 방식과 동일하여 팀 고유 최적화로 보지 않음.
- **sdp 공급 개선 가설 (기각, 2026-06-10)**: "bilinear depth resize가 경계 혼합값(flying points)을 만들어 정확도를 해친다 → nearest/원본해상도 직접 백프로젝션/밀도 4배(57.6k pts)로 개선" 가설을 5변형 paired ablation(고정 입력, A/A 노이즈 0.00mm)으로 검증 → **전 변형 효과 ≈ 0**(per-image prior_dev 중앙값 Δ ≤ 0.002 log, GT 앵커 오차 변화 < 1%). 원인: BoxerNet `sdp_to_patches`의 16×16 패치 median 집계가 보간·밀도 차이를 흡수. 코드는 `SDP_SOURCE`/`SDP_INTERP`/`SDP_TARGET_POINTS` 스위치(기본 레거시 동일)로 유지 — GPU 정밀도(fp16/bf16) 재검증용. commit `996f1cd`, 재현: `scripts/eval_ab.py` + `analyze_ablation.py`.

---

## [새 항목 템플릿] — 복사해서 위 로그에 추가

```markdown
#### <분류-번호>. <한 줄 제목>
- **무엇**: <무엇을 했나>
- **원본 대비**: <facebookresearch/boxer 대비 무엇이 달라졌나>
- **실측 효과**: <before → after, 수치. 없으면 "측정 전">
- **근거**: <file:line>, commit `<hash>`, <문서 참조>
- **날짜**: YYYY-MM-DD · **분류**: A|B|C
```
