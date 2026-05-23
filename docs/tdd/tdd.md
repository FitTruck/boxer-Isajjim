# boxer-Isajjim 기술 설계 문서 (TDD)

> 본 문서는 `boxer-Isajjim` 저장소의 실제 코드 (`ai/`, `api/`, `tests/`)를 기반으로
> 작성된 기술 설계 문서다. 모든 파일/심볼 참조는 `path:line` 형식으로 표기되어
> 코드와 직접 대응시킬 수 있다.

---

## 1. 개요

`boxer-Isajjim`은 가구 이미지에서 **3D 회전 경계 박스(OBB)** 와 **절대 치수·부피**를
산출하는 FastAPI 기반 AI 추론 서버다.

- **차별점**: SAM-3D + Gaussian Splat 기반 파이프라인 (Isajjim-AI)과 달리,
  Meta `BoxerNet`이 직접 metric 단위의 3D OBB를 출력한다.
  → PLY 파일 생성·GCS 업로드·`ply_url` 응답 필드 모두 제거됨.
- **입력**: Firebase Storage URL (다중 이미지) 또는 base64 단일 이미지.
- **출력**: `{label, width(mm), depth(mm), height(mm), volume(m³), center_x, center_y}` 리스트.
- **호출 모델**: 비동기 callback(`/analyze-furniture`) + 동기(`*-single`, `*-base64`, `/detect-furniture`).

---

## 2. 시스템 아키텍처

```
                ┌─────────────────────────────────────────────────────────┐
                │                  FastAPI Application                     │
                │                       (api/app.py)                       │
                │                                                          │
   HTTP POST →  │  ┌────────────┐    ┌──────────────────────────────────┐ │
   /analyze-*   │  │  Routers   │ →  │  GPUPoolManager (singleton)      │ │
                │  │ (routes/)  │    │  ai/gpu/gpu_pool_manager.py      │ │
                │  └────────────┘    │  - asyncio.Semaphore             │ │
                │        │           │  - round-robin device cursor     │ │
                │        │           │  - 디바이스당 1× FurniturePipeline │ │
                │        │           └────────────┬─────────────────────┘ │
                │        ▼                        ▼                        │
                │  BackgroundTask          FurniturePipeline               │
                │  + send_callback         ai/pipeline/furniture_pipeline  │
                │                                                          │
                └──────────────────────────┬───────────────────────────────┘
                                           │
                ┌──────────────────────────┴─────────────────────────────┐
                │                Pipeline (per-device)                    │
                │                                                         │
                │  ① ImageFetcher  →  ② Owlv2Detector  →                  │
                │     (1_images_   →   (2_owlv2_2d_   →                   │
                │     fetch.py)        detection.py)                      │
                │                       └─ LVIS+ subset                   │
                │                          (lvisplus_classes.csv)         │
                │                                                         │
                │                  ③ DepthEstimator  →                    │
                │                     (3_depth_                           │
                │                     estimation.py)                      │
                │                                                         │
                │                  ④ BoxerLifter                          │
                │                     (4_boxer.py)                        │
                └─────────────────────────────────────────────────────────┘
```

레이어 책임 분리:

| 레이어 | 디렉토리 | 책임 |
|--------|----------|------|
| HTTP/Routing | `api/` | 요청/응답 모델 검증, 라우팅, 콜백 |
| 자원 관리 | `ai/gpu/` | 멀티 디바이스 풀, 세마포어 스케줄링 |
| 추론 파이프라인 | `ai/pipeline/` | 4단계 추론 오케스트레이션 + LVIS+ 클래스 CSV |

---

## 3. 추론 파이프라인

### 3.1 파이프라인 오케스트레이션 (`ai/pipeline/furniture_pipeline.py`)

핵심 데이터 클래스:

```python
# furniture_pipeline.py
@dataclass
class DetectedObject:
    image_id: Optional[int]
    label: str           # OWLv2 LVIS class name (lowercase, spaces)
    confidence: float
    bbox_xyxy: List[float]
    center_xy: Tuple[float, float]
    width_mm: float = 0.0
    depth_mm: float = 0.0
    height_mm: float = 0.0
    volume_m3: float = 0.0
```

`FurniturePipeline.__init__()`는 디바이스 문자열을 받아 `Owlv2Detector`,
`DepthEstimator`, `BoxerLifter`를 모두 동일 디바이스에 로드한다.
`enable_3d=False`일 경우 depth/boxer를 건너뛰고 detector 결과만 사용.

처리 진입점은 2개:
- `process_single_image(url, image_id)` — URL fetch + `process_pil` 위임.
- `process_pil(image, …, enable_3d=None)` — 디스크/네트워크 I/O 없는 순수 추론.
  `enable_3d` 인자로 호출별 3D 단계 on/off (풀에서 빌려온 파이프라인을 재로드 없이
  detect-only로도 사용 가능).

흐름 (3D 활성 시):

1. `Owlv2Detector.detect()` → 큐레이션된 LVIS+ 클래스로 chunked OWLv2 추론,
   chunk 간 class-agnostic NMS 병합.
2. `DepthEstimator.estimate()` → metric depth + (Depth Pro인 경우) `focal_length_px`.
3. `BoxerLifter.lift()` → `BoxerObb` 리스트 (각 OBB는 `input_index`로 원본
   detection을 가리킴).
4. `obb_by_idx = {obb.input_index: obb for obb in obbs}` → detection index 기반
   1:1 매칭. 동명 라벨이 여러 개여도 안정적.

### 3.2 Stage 1 — `ImageFetcher` (`ai/pipeline/1_images_fetch.py`)

- 비동기: `aiohttp.ClientSession` (총 timeout 30s).
- 동기 폴백: `requests` (aiohttp 미설치 시).
- 로컬 경로 지원: `file://` prefix 또는 `/`로 시작하는 절대 경로.
- 실패 시 `None` 반환 (호출자가 `PipelineResult.error`로 변환).

### 3.3 Stage 2 — `Owlv2Detector` (`ai/pipeline/2_owlv2_2d_detection.py`)

OWLv2 (`google/owlv2-base-patch16-ensemble`) 기반 텍스트-프롬프트 open-vocabulary 탐지기.
boxer 원본의 OWLv2 사용 노선을 따르되, 가구/실내 물품에 특화된 LVIS+ 서브셋을
프롬프트로 사용한다.

**클래스 어휘** (`ai/pipeline/lvisplus_classes.csv`):
- LVIS+에서 큐레이션된 95개 클래스 (sofa, chair, refrigerator, vase, …).
- CSV는 OWLv2가 직접 프롬프트로 쓰는 어휘이자, 동시에 응답 필터 역할 (CSV에 없는
  카테고리는 검출되지 않음 → `filter_excluded` 같은 별도 단계 불필요).

**프롬프트 생성**:
```python
self.display_labels = [c.replace("_", " ") for c in self.classes]
self.prompts = [f"a photo of a {lbl}" for lbl in self.display_labels]
```

**추론** (`detect()`):
1. 큐 95개를 `OWLV2_CHUNK_SIZE`(기본 256) 단위로 chunk.
2. 각 chunk에 대해 `Owlv2Processor` 전처리 + 모델 forward + `post_process_grounded_object_detection`.
3. 모든 chunk 결과를 합친 뒤 class-agnostic NMS (`IoU=0.5`)로 중복 제거.
4. `confidence` 임계값 `OWLV2_CONFIDENCE` (기본 0.25).

반환 dict 키:
- `boxes` (N,4) xyxy float32
- `scores` (N,) float32
- `classes` (N,) int (LVIS class id)
- `labels` (list[str]) 공백 정규화된 LVIS 라벨

### 3.4 Stage 3 — `DepthEstimator` (`ai/pipeline/3_depth_estimation.py`)

전략 패턴으로 두 backend 지원:

| Backend | 모델 | 특징 |
|---------|------|------|
| `depthpro` (default) | `apple/DepthPro-hf` | metric depth + **focal length** 동시 예측 → BoxerNet에 실제 intrinsics 전달 가능 |
| `da2` | `depth-anything/Depth-Anything-V2-Small-hf` | 더 가벼움, focal length 없음, depth는 up-to-scale → 10m로 normalize |

`DepthResult` 공통 인터페이스:

```python
# 3_depth_estimation.py:28-31
@dataclass
class DepthResult:
    depth: np.ndarray             # (H, W) float32, meters
    focal_length_px: Optional[float] = None   # Depth Pro only
```

**Backend 빌드 + 폴백** (`_build_backend`, `:130-139`): DepthPro 로드 실패 시
자동으로 Depth Anything V2로 폴백.

`AutoImageProcessor` / `AutoModelForDepthEstimation`을 직접 사용한다 — 일부
torch/transformers dev 조합에서 깨지는 `transformers.pipelines` import를 의도적으로 회피.

DA V2 출력은 up-to-scale이므로 `:103-105`에서 max로 나눠 0~10m 범위로 rescale.

> 카메라 좌표계 unprojection 로직은 BoxerLifter 내부의 `_depth_to_sdp`에서 직접
> 수행하며, DepthEstimator는 depth + focal length만 제공한다.

### 3.5 Stage 4 — `BoxerLifter` (`ai/pipeline/4_boxer.py`)

Meta `facebookresearch/boxer`의 `BoxerNet`을 wrapping.

**런타임 의존성**:
- `BOXER_REPO_PATH` (env) → `sys.path.insert(0, ...)`로 동적 import (`:92-99`).
- `BOXER_CHECKPOINT` (env) → `BoxerNet.load_from_checkpoint`.
- 둘 중 하나라도 없으면 `self.net=None`으로 두고 `lift()`가 빈 리스트 반환 (`:135-136`).

**디바이스 처리** (`_normalize_boxer_device`, `:68-75`): BoxerNet은 `cuda`/`mps`/`cpu`만
받으므로 `cuda:N`은 `cuda`로 변환 후 로드. 이후 `.to(self.device)`로 특정 GPU로 재배치.

**핵심 lift 로직** (`lift()`, `:113-181`):

1. **정사각 리사이즈**: BoxerNet은 `hw × hw` 정사각 입력만 받음 (default 960). 이미지·depth·bbox·intrinsics 모두 같은 비율로 스케일 (`sx, sy`).
2. **Intrinsics 스케일**: `focal_length_px`가 있으면 축별로 독립 스케일 (`fx *= sx, fy *= sy`). 없으면 `target * 0.75` (≈75° FOV pinhole) 가정.
3. **데이텀 구축**:
   - `img0`: `(1, 3, H, W)` float [0,1].
   - `cam0`: `CameraTW.from_surreal(type_str="Pinhole", params=[fx,fy,cx,cy])`.
   - `T_world_rig0`: identity pose (rig frame = world frame).
   - `sdp_w`: `(N, 3)` 카메라 프레임 3D points (identity pose이므로 world와 동일).
   - `bb2d`: `(1, M, 4)` boxer 순서 `(x1, x2, y1, y2)` — `_to_boxer_order`로 변환.
4. **추론**: CUDA + bf16 지원 시 `torch.autocast(bfloat16)`로, 그 외는 fp32.
5. **결과 파싱** (`_to_results`):
   - `obb_w.bb3_diagonal` (M,3) → `(w_m, h_m, d_m)`.
   - `obb_w.bb3_volumes` (M,) → `volume_m3`.
   - `obb_w.bb3_center_world` (M,3) → `center_world`.
   - `obb_w.prob < conf_threshold(0.2)` 항목은 드롭.
   - 각 `BoxerObb`에 `input_index`(원본 2D detection의 위치)를 같이 담아 반환 →
     상위 레이어에서 1:1 매칭 가능.

> ⚠️ Monocular depth 기반 BoxerNet은 멀티뷰 대비 보수적이므로 `conf_threshold`를
> 기본 0.2로 낮춰 잡았다.

### 3.6 라벨 매핑

별도 매퍼/사전 없음. OWLv2가 출력하는 LVIS 클래스명을 그대로 응답 `label`로
사용한다. CSV(`lvisplus_classes.csv`)에 등록된 어휘만 OWLv2가 검출하므로 응답
어휘는 결정론적이고, 백엔드/프론트가 LVIS 라벨 문자열을 직접 인식한다.

---

## 4. GPU 풀 매니저 (`ai/gpu/gpu_pool_manager.py`)

### 4.1 설계 의도

- 디바이스(`cuda:N` / `mps` / `cpu`)당 `FurniturePipeline`을 **미리 1개씩** 로드.
- 요청이 들어오면 round-robin으로 빈 슬롯을 골라 디스패치.
- 스케줄링은 **`asyncio.Semaphore`** 기반 — 폴링 sleep 없이 즉시 wake-up.
  → 테스트(`test_gpu_pool.py:40-58`)에서 0.05s 안에 dispatch되는 것을 검증.

### 4.2 핵심 자료구조

```python
# gpu_pool_manager.py:25-31
@dataclass
class _Slot:
    device: str             # "cuda:0" / "mps" / "cpu"
    available: bool = True
    task_id: Optional[str] = None
    pipeline: object = None
    last_used: float = field(default_factory=time.time)
```

`GPUPoolManager.__init__` (`:37-42`):
- `self._slots`: `{device: _Slot}`
- `self._lock = asyncio.Lock()` — 슬롯 마킹 동시성 보호
- `self._sem = asyncio.Semaphore(len(devices))` — 가용 슬롯 수 = 세마포어 토큰 수
- `self._idx = 0` — round-robin 커서

### 4.3 acquire / release

`acquire(task_id, wait_timeout=300)` (`:75-99`):

1. `await self._sem.acquire()` — 슬롯이 비기를 기다림.
2. `self._lock` 안에서 round-robin으로 사용 가능 슬롯 탐색 → `available=False` 마킹.
3. (방어적) 토큰을 받았는데 빈 슬롯이 없으면 `RuntimeError("inconsistency")`.

`release(device)` (`:101-111`):
- 이미 free인 슬롯에 대해 호출되어도 세마포어를 over-credit하지 않도록 가드 (`test_release_is_idempotent`로 검증).
- 실제로 한 자리가 비었을 때만 `self._sem.release()`.

`pipeline_context(task_id)` (`:121-127`):
- 비동기 컨텍스트 매니저로 `(device, pipeline)` 튜플을 yield → `finally`에서 release 보장.

### 4.4 라이프사이클

`initialize_pipelines(factory, skip_on_error=True)` (`:47-64`):
- 각 디바이스에 대해 `factory(device)`를 동기 호출하여 `_Slot.pipeline`을 채움.
- 실패한 디바이스는 (`skip_on_error=True`인 경우) 로그만 남기고 건너뜀.
- 정상 초기화된 슬롯 수를 반환.

서버 startup에서 호출 (`api/app.py:26-42`):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    devices = Config.get_available_devices() or ["cpu"]
    pool = initialize_gpu_pool(devices)
    await pool.initialize_pipelines(
        lambda device: FurniturePipeline(device=device),
        skip_on_error=True,
    )
    yield
    await shutdown_gpu_pool()
```

### 4.5 상태 조회 (`get_status`, `:132-163`)

`/gpu-status` 엔드포인트가 반환하는 dict:

```json
{
  "total_devices": 2,
  "available_devices": 1,
  "pipelines_initialized": 2,
  "devices": {
    "cuda:0": {"available": false, "task_id": "est123_img101", "memory_used_mb": 4321, "has_pipeline": true},
    "cuda:1": {"available": true,  "task_id": null,             "memory_used_mb": 1234, "has_pipeline": true}
  }
}
```

CUDA 메모리는 `torch.cuda.memory_allocated(idx)`로 디바이스별 조회.

---

## 5. API 레이어

### 5.1 앱 부트스트랩 (`api/app.py`)

- `sys.path`에 repo 루트 삽입 → `uvicorn api:app` 형태 실행 지원.
- `lifespan`에서 GPU 풀 + 파이프라인 사전 로드.
- 라우터 등록: `health_router`, `furniture_router`.

`api/config.py`에서 torch import **전에** OMP/BLAS/MKL 스레드 수를 4로 강제 → CPU 폴백 시 oversubscription 방지.

### 5.2 요청 모델 (`api/models.py`)

```python
# models.py:13-20
class AnalyzeFurnitureRequest(BaseModel):
    estimate_id: int
    image_urls: List[ImageUrlItem] = Field(..., min_length=1, max_length=20)
```

- `min_length=1` / `max_length=20` 검증 → 빈 리스트 또는 21개 이상은 Pydantic 422 응답.
- `AnalyzeFurnitureBase64Request.enable_3d: bool = True` — 3D 단계 스킵 옵션.

### 5.3 라우트 (`api/routes/furniture.py`)

| Endpoint | 모드 | 처리 |
|----------|------|------|
| `POST /analyze-furniture` | 비동기 + callback | `BackgroundTasks`에 등록 → 즉시 `{"status":"processing"}` 응답 |
| `POST /analyze-furniture-single` | 동기 | 단일 URL fetch + 추론, JSON 응답 |
| `POST /analyze-furniture-base64` | 동기 | base64 디코드 + (옵션: `enable_3d=False`) 추론 |
| `POST /detect-furniture` | 동기 | 탐지만 (`enable_3d=False`) + `processing_time_seconds` 동봉 |
| `GET /health` | sync | `{"status":"healthy","device":...}` |
| `GET /gpu-status` | sync | `pool.get_status()` |

**파이프라인 borrow 패턴** (`_borrow_pipeline`, `:36-53`):

```python
@asynccontextmanager
async def _borrow_pipeline(task_id):
    pool = get_gpu_pool()  # 또는 None
    if pool and any(pool.has_pipeline(d) for d in pool.devices):
        async with pool.pipeline_context(task_id=task_id) as (device, pipeline):
            yield pipeline, device
        return
    # Fallback: 풀 없음 → ad-hoc 1회용 파이프라인
    pipeline = FurniturePipeline()
    yield pipeline, pipeline.device or "cpu"
```

**비동기 분석 + 콜백** (`_analyze_and_callback`, `:56-90`):

이미지 N개를 각각 별도 디바이스에 분산하기 위해 `asyncio.gather`로 fan-out.

```python
async def _one(image_id, url):
    tid = f"est{estimate_id}_img{image_id}"
    async with pool.pipeline_context(task_id=tid) as (device, pipeline):
        return await pipeline.process_single_image(url, image_id=image_id)

results = await asyncio.gather(*(_one(iid, url) for iid, url in image_items))
await send_callback(estimate_id, result_data=FurniturePipeline.to_json_response(results))
```

- 디바이스가 N개면 최대 N개 이미지가 진짜 병렬로 추론된다 (Python GIL 이슈를 GPU 분산으로 우회).
- 풀이 없을 땐 `process_multiple_images`로 폴백 (단일 파이프라인, asyncio 동시 실행).

### 5.4 콜백 (`api/services/callback.py`)

```python
# callback.py:18-52
async def send_callback(estimate_id, result_data=None, error=None) -> bool:
    url = CALLBACK_URL_TEMPLATE.replace("{estimateId}", str(estimate_id))
    payload = {"error": error} if error else (result_data or {"error": "Unknown error"})
    headers = {"X-INTERNAL-TOKEN": INTERNAL_TOKEN}
    for attempt in range(CALLBACK_RETRY_COUNT + 1):
        async with aiohttp.ClientSession(timeout=...) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if 200 <= response.status < 300:
                    return True
                # log + retry
    return False
```

- URL 템플릿: `https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback` (default).
- 인증: `X-INTERNAL-TOKEN` 헤더 (env `AUTH_TOKEN`). 백엔드의 `auth.internal-token`과 일치해야 함.
- 재시도: `CALLBACK_RETRY_COUNT + 1`회 (default 2회 = 1회 재시도).
- 실패는 로그만 남기고 `False` 반환 (요청자에게 별도 알림 없음 — async 모드 한계).

---

## 6. 데이터 흐름 (시퀀스)

### 6.1 `POST /analyze-furniture` (비동기 멀티 이미지)

```
Client          FastAPI                BackgroundTask                Pool                  Pipeline                  Backend
  │                │                         │                         │                       │                          │
  │── POST ───────▶│                         │                         │                       │                          │
  │                │── add_task ────────────▶│                         │                       │                          │
  │◀── {processing}│                         │                         │                       │                          │
  │                │                         │── acquire(img1) ───────▶│                       │                          │
  │                │                         │                         │── pipeline_context ──▶│                          │
  │                │                         │                         │                       │── fetch URL              │
  │                │                         │                         │                       │── OWLv2 detect           │
  │                │                         │                         │                       │── Depth estimate         │
  │                │                         │                         │                       │── Boxer lift             │
  │                │                         │                         │── release ◀───────────│                          │
  │                │                         │   (병렬로 img2…N 동시 진행, asyncio.gather)                                    │
  │                │                         │── send_callback(results) ───────────────────────────────────────────────▶│
```

### 6.2 단일 이미지 시 (`/analyze-furniture-single`)

```
Client → FastAPI → _borrow_pipeline → pool.acquire → pipeline.process_single_image → JSON 응답
                                                  ↓
                                          (release in finally)
```

### 6.3 파이프라인 내부 (단일 이미지)

```
URL ─▶ ImageFetcher.fetch_async ─▶ PIL.Image (RGB)
                                       │
                                       ▼
                          Owlv2Detector.detect
                              (LVIS+ CSV 프롬프트, chunked OWLv2 + NMS)
                                       │
                                       ▼
                          DepthEstimator.estimate
                              → DepthResult(depth, focal_length_px?)
                                       │
                                       ▼
                          BoxerLifter.lift(
                              image, bboxes, labels,
                              depth, focal_length_px)
                              → List[BoxerObb]   (각 OBB에 input_index 부여)
                                       │
                                       ▼
                          obb_by_idx[i] 매칭 → DetectedObject(
                              label, confidence,
                              bbox_xyxy, center_xy,
                              width_mm, depth_mm, height_mm, volume_m3)
                                       │
                                       ▼
                          to_json_response → JSON
```

---

## 7. 설정 (`ai/config.py`, `api/config.py`)

### 7.1 환경변수 매트릭스

| Var | Default | 위치 | 설명 |
|-----|---------|------|------|
| `OWLV2_MODEL` | `google/owlv2-base-patch16-ensemble` | `ai/config.py` | OWLv2 HF repo id |
| `OWLV2_CONFIDENCE` | `0.25` | `ai/config.py` | 탐지 임계값 |
| `OWLV2_CHUNK_SIZE` | `256` | `ai/config.py` | 텍스트 프롬프트 chunk 크기 |
| `OWLV2_CLASSES_CSV` | (없음 → `ai/pipeline/lvisplus_classes.csv`) | `ai/config.py` | LVIS+ 클래스 CSV 경로 override |
| `DEPTH_BACKEND` | `depthpro` | `ai/config.py:17` | `depthpro` 또는 `da2` |
| `DEPTH_PRO_MODEL` | `apple/DepthPro-hf` | `ai/config.py:18` | HF repo id |
| `DEPTH_DA2_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | `ai/config.py:19` | HF repo id |
| `BOXER_REPO_PATH` | `<repo>/boxer` | `ai/config.py:24` | `facebookresearch/boxer` 클론 경로 |
| `BOXER_CHECKPOINT` | (없음) | `ai/config.py:28` | BoxerNet `.pt` 절대 경로 — **필수** |
| `DEVICES` | (없음) | `ai/config.py:34` | 예: `cuda:0,cuda:1` (최우선) |
| `GPU_IDS` | (없음) | `ai/config.py:35` | 예: `0,1,2,3` → `cuda:N` 확장 |
| `ENABLE_MULTI_GPU` | `true` | `ai/config.py:36` | false면 단일 GPU만 |
| `OMP/OPENBLAS/MKL_NUM_THREADS` | `4` | `api/config.py:6-8` | torch import 전 강제 |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | `api/config.py:9` | macOS MPS 미지원 op CPU 폴백 |
| `CALLBACK_URL_TEMPLATE` | `https://api.isajjim.kro.kr/.../{estimateId}/callback` | `api/config.py:28` | 백엔드 콜백 URL |
| `CALLBACK_TIMEOUT_SECONDS` | `30` | `api/config.py:32` | 콜백 HTTP timeout |
| `CALLBACK_RETRY_COUNT` | `1` | `api/config.py:33` | 콜백 재시도 횟수 |
| `AUTH_TOKEN` | `""` | `api/config.py:36` | 백엔드 `X-INTERNAL-TOKEN` |
| `LOG_LEVEL` | `INFO` | `api/app.py:22` | 로깅 레벨 |

### 7.2 디바이스 검출 로직 (`ai/config.py:38-50`)

우선순위 (높음 → 낮음):
1. `DEVICES` env (`"cuda:0,cuda:1"`)
2. `GPU_IDS` env (`"0,1"` → `cuda:0,cuda:1`)
3. `torch.cuda.is_available()` + `ENABLE_MULTI_GPU` → 모든 CUDA 장치
4. `torch.backends.mps.is_available()` → `["mps"]`
5. fallback → `["cpu"]`

---

## 8. 테스트 전략

`pytest.ini`:

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
addopts = -v --tb=short
markers =
    slow: marks tests as slow
    gpu: requires CUDA GPU
```

| 테스트 파일 | 대상 | 모델 가중치 요구 |
|-------------|------|------------------|
| `test_gpu_pool.py` | round-robin, idempotent release, **즉시 wake-up (no polling)**, status reporting | ❌ |
| `test_image_fetcher.py` | 로컬 파일 로드, 404 graceful failure | ❌ |
| `test_api.py` | `/health`, `/gpu-status`, base64 validation 400, Pydantic 422 | ❌ |
| `e2e_pipeline.py` (스크립트) | OWLv2 + Depth Pro + BoxerNet 풀 스택 스모크 | ✅ |

**모델 가중치가 필요한 경로** (실제 OWLv2 / DepthPro / BoxerNet 추론)는 단위 테스트
범위 밖이며, boxer 체크포인트가 있는 환경에서 `tests/e2e_pipeline.py`로 수동 검증한다.

특이 사항:
- `test_gpu_pool.py:40-58`은 세마포어가 폴링이 아님을 보장한다 — `release()` 후
  대기 중인 `acquire()`가 0.3s 안에 깨어나야 통과. 구 `asyncio.sleep` 폴링 구현은
  0.5s 이상 걸려 실패한다.
- `test_image_fetcher.py:25`는 `file://` URL이 `fetch_sync`로 fall through하지 않는
  현재 구현의 한계를 우회하기 위해 `_load_local`을 직접 호출한다 (스모크 의도).

---

## 9. 응답 스펙

### 9.1 `/analyze-furniture` 콜백 페이로드 (`FurniturePipeline.to_json_response`)

```json
{
  "results": [
    {
      "image_id": 101,
      "objects": [
        {
          "label": "sofa",
          "width": 1850.0,
          "depth": 920.0,
          "height": 870.0,
          "volume": 1.482,
          "center_x": 540.5,
          "center_y": 360.0
        }
      ]
    }
  ]
}
```

- 모든 치수: **mm** (소수점 1자리).
- 부피: **m³** (소수점 6자리).
- 중심 좌표: 이미지 픽셀 (소수점 1자리).
- `ply_url` 필드 **없음** — Isajjim-AI TDD와의 핵심 차이.

### 9.2 `/detect-furniture` 응답 (`api/routes/furniture.py`)

탐지만 수행하는 경량 응답 — 3D 필드 없이 raw bbox + confidence를 그대로 노출.

```json
{
  "success": true,
  "total_objects": 3,
  "processing_time_seconds": 0.123,
  "objects": [
    {"label": "bed", "bbox": [...], "center_point": [...], "confidence": 0.87}
  ]
}
```

---

## 10. 운영 고려사항

### 10.1 워커 수

- **GPU 환경**: `uvicorn --workers 1`. 풀이 GPU 전체를 점유하므로 워커를 늘리면
  같은 GPU를 경쟁한다. README §실행 참고.
- **CPU 환경**: 워커 N 가능하지만 추론 throughput은 낮음.

### 10.2 동시성 한계

- 동시 처리 가능한 이미지 수 = `len(devices)` (풀 슬롯 수).
- 초과 요청은 세마포어에서 대기. `wait_timeout=300s` 후 `RuntimeError`.
- 요청당 이미지 수 ≤ 20 (`models.py:20` `max_length=20`).

### 10.3 GPU 메모리

`/gpu-status`의 `memory_used_mb`로 모니터링. 비정상 누적 시 풀이 자동 회수하지 않으므로
별도 모니터링/재시작 정책이 필요.

### 10.4 콜백 실패 시 동작

`send_callback`이 모든 재시도 실패 → ERROR 로그만 남고 클라이언트는 결과를 받지 못함.
백엔드가 timeout 기반으로 재요청하는 패턴을 가정한다.

### 10.5 BoxerNet 미설치 환경

`BOXER_CHECKPOINT` / `BOXER_REPO_PATH` 누락 시:
- `BoxerLifter._load`가 경고 로그 후 `self.net=None` (`4_boxer.py:78-90`).
- `lift()`는 빈 리스트 반환 → 모든 객체의 3D 필드가 0.0으로 채워짐.
- detect-only 엔드포인트 및 `/health`는 정상 동작 — 점진적 배포 가능.

---

## 11. 향후 개선 후보 (관찰된 코드 기반)

> 코드에서 직접 드러나는 단순화/개선 여지. 우선순위 없음, 의사결정 자료로만.

- **콜백 실패 영속화**: 현재는 로그만 남는다. 재시도 가능한 큐(Redis/DB)에 적재 시
  복구 가능성 ↑.
- **DA V2의 metric scale 가정** (`3_depth_estimation.py:104-105`)은 max를 10m로
  고정 스케일하는 단순 휴리스틱 — DepthPro 폴백 시 부피가 크게 어긋날 수 있다.
- **단일 시점 monocular의 OBB depth 한계**: BoxerNet은 멀티뷰/LiDAR 학습 분포라 단일
  사진 입력에서 SOFA depth가 underestimate되는 경향. 후처리에서 카테고리별 표준
  비율로 보정하거나, 멀티뷰 입력 경로(`view_fusion`) 추가 검토.
- **번호 접두 파일명** (`1_*.py` ~ `4_*.py`)은 importlib 우회를 강제한다
  (`ai/pipeline/__init__.py:14-17`). 가독성을 위해 일반 이름(`stage1_fetch.py` 등)
  으로 바꾸는 것을 고려할 수 있다.

---

## 부록 A. 파일 인덱스

| 파일 | 핵심 심볼 | 역할 |
|------|-----------|------|
| `api/app.py` | `app`, `lifespan` | FastAPI 인스턴스, 풀 부트스트랩 |
| `api/config.py` | `device`, `CALLBACK_*`, `INTERNAL_TOKEN` | 환경/콜백 설정 |
| `api/models.py` | `AnalyzeFurnitureRequest`, `…Single`, `…Base64` | Pydantic 요청 |
| `api/routes/furniture.py` | `analyze_furniture`, `_analyze_and_callback`, `_borrow_pipeline` | 비즈니스 라우트 |
| `api/routes/health.py` | `health_check`, `gpu_status` | 운영용 |
| `api/services/callback.py` | `send_callback` | 백엔드 콜백 |
| `ai/config.py` | `Config`, `get_available_devices` | 추론 환경 설정 |
| `ai/gpu/gpu_pool_manager.py` | `GPUPoolManager`, `pipeline_context` | 멀티 디바이스 풀 |
| `ai/pipeline/furniture_pipeline.py` | `FurniturePipeline`, `DetectedObject`, `PipelineResult` | 오케스트레이션 |
| `ai/pipeline/1_images_fetch.py` | `ImageFetcher` | URL→PIL |
| `ai/pipeline/2_owlv2_2d_detection.py` | `Owlv2Detector`, `_load_lvis_classes` | OWLv2 텍스트-프롬프트 2D 탐지 |
| `ai/pipeline/3_depth_estimation.py` | `DepthEstimator`, `_DepthProBackend`, `_DepthAnythingBackend` | metric depth |
| `ai/pipeline/4_boxer.py` | `BoxerLifter`, `BoxerObb` (`input_index` 포함) | 3D OBB lifting |
| `ai/pipeline/lvisplus_classes.csv` | (데이터) | OWLv2 프롬프트 어휘 (95개) |
| `tests/conftest.py` | (sys.path 설정) | repo root import 가능하게 |
| `tests/test_*.py` | — | 단위 테스트 (모델 가중치 불요) |
| `tests/e2e_pipeline.py` | — | OWLv2 + Depth Pro + BoxerNet 전체 스모크 |

## 부록 B. 외부 의존 모델

- **OWLv2** — `google/owlv2-base-patch16-ensemble` (transformers `Owlv2ForObjectDetection`).
- **Apple Depth Pro** — `apple/DepthPro-hf` (transformers AutoModel).
- **Depth Anything V2 Small** — `depth-anything/Depth-Anything-V2-Small-hf` (fallback).
- **Meta BoxerNet** — `facebookresearch/boxer` 별도 클론 + `boxernet_*.ckpt`.
