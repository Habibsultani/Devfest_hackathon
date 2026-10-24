#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_root"
setup=false
check_only=false
port=8501
while (($#)); do
  case "$1" in
    --setup) setup=true; shift ;;
    --check) check_only=true; shift ;;
    --port) port="${2:?Provide a port after --port}"; shift 2 ;;
    *) echo "Usage: bash scripts/start.sh [--setup] [--check] [--port NUMBER]" >&2; exit 2 ;;
  esac
done
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1024 || port > 65535)); then
  echo "Port must be between 1024 and 65535." >&2; exit 2
fi
project_python="$project_root/.venv/bin/python"
if [[ ! -x "$project_python" ]]; then
  "${PYTHON:-python3}" -m venv .venv
  setup=true
fi
if "$setup"; then "$project_python" -m pip install -r requirements.txt; fi
"$project_python" -m scripts.doctor
if ! "$check_only"; then
  exec "$project_python" -m streamlit run app.py --server.address 127.0.0.1 --server.port "$port"
fi
