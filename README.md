# boxer-Isajjim

가구 분석을 위한 FastAPI 기반 AI 서버.
**OWLv2 (LVIS+ 큐레이션 어휘) → Depth Pro → Meta Boxer (3D OBB lifting)** 파이프라인으로
2D 이미지에서 가구를 탐지하고 절대 치수/부피를 산출한다.

> Isajjim-AI의 API 명세를 따르되, **PLY 파일을 생성/저장하지 않으므로**
> GCS 업로드와 `ply_url` 응답 필드는 모두 제거되었다.

---

## 핵심 차이점: SAM-3D vs Boxer

| 항목 | Isajjim-AI (SAM-3D) | boxer-Isajjim (Boxer) |
|------|---------------------|----------------------|
| 3D 생성 | Gaussian Splat PLY | 3D OBB 직접 (no PLY) |
| 절대 부피 | DB 표준 치수와 비교 후 변환 | **모델이 metric 단위로 직접 출력** |
| GCS 업로드 | PLY → `ply_url` | **없음** |
| 응답 `ply_url` | 있음 | **제거됨** |
| 응답 `type` (서브타입) | DB 비교로 매칭 | **없음** (비교 로직 제거) |

---

## 디렉토리 구조

```
boxer-Isajjim/
├── ai/
│   ├── config.py
│   ├── gpu/                        # Multi-GPU round-robin pool
│   │   └── gpu_pool_manager.py
│   └── pipeline/
│       ├── 1_images_fetch.py       # Firebase URL → PIL
│       ├── 2_owlv2_2d_detection.py # OWLv2 텍스트-프롬프트 탐지
│       ├── 3_depth_estimation.py   # Depth Pro (default) / Depth Anything V2 fallback
│       ├── 4_boxer.py              # BoxerNet 3D OBB lifting
│       ├── furniture_pipeline.py   # 오케스트레이터
│       └── lvisplus_classes.csv    # OWLv2 프롬프트 어휘 (95개 가구/물품)
├── api/
│   ├── app.py                      # FastAPI app + startup
│   ├── config.py                   # callback URL, device
│   ├── models.py                   # Pydantic 요청/응답
│   ├── routes/{health,furniture}.py
│   └── services/callback.py        # 백엔드 callback POST
├── tests/                          # pytest 단위 + e2e_pipeline.py 스모크
├── requirements.txt
└── pytest.ini
```

---

## 설치

```bash
# 1) Python 의존성
pip install -r requirements.txt

# 2) Boxer 레포 클론 + 체크포인트 다운로드
git clone https://github.com/facebookresearch/boxer.git
cd boxer && bash scripts/download_ckpts.sh && cd ..

# 3) 환경변수
export BOXER_REPO_PATH=$PWD/boxer
export BOXER_CHECKPOINT=$PWD/boxer/ckpts/boxernet_default.pt  # 실제 파일명 확인
```

추가 환경변수:

| Var | 기본값 | 설명 |
|-----|--------|------|
| `OWLV2_MODEL` | `google/owlv2-base-patch16-ensemble` | OWLv2 HF repo id |
| `OWLV2_CONFIDENCE` | `0.25` | 탐지 confidence threshold |
| `OWLV2_CHUNK_SIZE` | `256` | 텍스트 프롬프트 chunk 크기 |
| `OWLV2_CLASSES_CSV` | (없음 → `ai/pipeline/lvisplus_classes.csv`) | 어휘 CSV 경로 override |
| `DEPTH_BACKEND` | `depthpro` | `depthpro` (Apple, metric + 초점거리 출력) 또는 `da2` (Depth Anything V2, 가볍지만 초점거리 없음) |
| `DEPTH_PRO_MODEL` | `apple/DepthPro-hf` | Depth Pro HF 가중치 |
| `DEPTH_DA2_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | DA V2 HF 가중치 |
| `ENABLE_MULTI_GPU` | `true` | 모든 GPU 자동 검출 |
| `GPU_IDS` | (auto) | 예: `0,1,2,3` |
| `CALLBACK_URL_TEMPLATE` | `https://api.isajjim.kro.kr/api/v1/estimates/{estimateId}/callback` | 비동기 결과 callback URL |
| `AUTH_TOKEN` | (필수) | 백엔드 `X-INTERNAL-TOKEN` 헤더에 실리는 공유 시크릿. Isajjim-Backend의 `auth.internal-token`과 동일해야 함 |

---

## 실행

```bash
# 개발 (auto-reload)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# 프로덕션
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

> Multi-GPU 환경에서는 워커 1개 + GPU 풀이 모든 GPU를 점유한다.
> 워커를 늘리면 같은 GPU를 경쟁하므로 권장하지 않는다.

---

## API

### `POST /analyze-furniture` (비동기 callback)

요청:
```json
{
  "estimate_id": 123,
  "image_urls": [
    {"id": 101, "url": "https://firebase.../1.jpg"},
    {"id": 102, "url": "https://firebase.../2.jpg"}
  ]
}
```

즉시 응답:
```json
{"success": true, "estimate_id": 123, "status": "processing"}
```

분석이 끝나면 `CALLBACK_URL_TEMPLATE`로 POST:
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

> 모든 치수는 **mm**, 부피는 **m³**. boxer가 metric 단위로 직접 출력하므로
> 백엔드의 fallback 변환이 거의 트리거되지 않는다.

### `POST /analyze-furniture-single` — 단일 URL 동기 분석
### `POST /analyze-furniture-base64` — base64 이미지 동기 분석
### `POST /detect-furniture` — 탐지만 (3D 생략)
### `GET /health`, `GET /gpu-status`

---

## 테스트

```bash
pytest -v                          # 전체 단위 테스트
pytest --cov=ai --cov=api          # 커버리지
python tests/e2e_pipeline.py ai/imgs/img.png   # OWLv2 + Depth Pro + BoxerNet 풀 스택 스모크
```

모델 weight 없이 통과하는 테스트: `test_api.py`, `test_gpu_pool.py`, `test_image_fetcher.py`.
실제 추론 검증은 boxer 체크포인트 + Depth Pro 가중치가 있는 환경에서 `e2e_pipeline.py`로 수행.

---

## 데이터 흐름 요약

```
URL → PIL Image
   │
   ▼
OWLv2 (LVIS+ CSV 프롬프트, chunked + NMS)  ───→  bbox (xyxy), label, score
   │
   ▼
Depth Pro (or DA V2)  ───→  (H, W) metric depth  + focal_length_px (Pro only)
   │
   ▼
BoxerNet ( img + predicted intrinsics + identity pose + sparse 3D pts + bb2d )
   │
   ▼
ObbTW  ───→  bb3_diagonal (m), bb3_volumes (m³), bb3_center_world
                  + input_index (원본 detection 매칭용)
   │
   ▼
JSON ( label, width_mm, depth_mm, height_mm, volume_m3, center_x, center_y )
   │
   ▼
Backend callback POST
```
