#!/bin/sh
set -eu

export BOXER_REPO_PATH="${BOXER_REPO_PATH:-/app/boxer}"
export REQUIRE_BOXER_CHECKPOINT="${REQUIRE_BOXER_CHECKPOINT:-true}"

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ -z "${BOXER_CHECKPOINT:-}" ] && [ -d "$BOXER_REPO_PATH/ckpts" ]; then
  checkpoint="$(find "$BOXER_REPO_PATH/ckpts" -maxdepth 1 -type f -name '*.ckpt' | sort | head -n 1 || true)"
  if [ -n "$checkpoint" ]; then
    export BOXER_CHECKPOINT="$checkpoint"
  fi
fi

if [ "$REQUIRE_BOXER_CHECKPOINT" = "true" ]; then
  if [ ! -d "$BOXER_REPO_PATH" ]; then
    echo "BOXER_REPO_PATH does not exist: $BOXER_REPO_PATH" >&2
    echo "Mount the Boxer repo, for example: -v /opt/boxer/boxer:/app/boxer:ro" >&2
    exit 1
  fi

  if [ -z "${BOXER_CHECKPOINT:-}" ] || [ ! -f "$BOXER_CHECKPOINT" ]; then
    echo "BOXER_CHECKPOINT not found." >&2
    echo "Mount Boxer checkpoints under $BOXER_REPO_PATH/ckpts or set BOXER_CHECKPOINT explicitly." >&2
    exit 1
  fi
fi

if [ "$#" -eq 0 ]; then
  set -- uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${UVICORN_WORKERS:-1}" --log-level "${UVICORN_LOG_LEVEL:-info}"
fi

exec "$@"
