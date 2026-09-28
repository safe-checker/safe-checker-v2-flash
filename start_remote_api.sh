#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "未找到项目虚拟环境，请先执行：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "未找到 cloudflared。macOS 可先执行：brew install cloudflared" >&2
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo "未找到 .env，请先复制 .env.example 并填写模型 API Key。" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source ".env"
set +a

if [[ -z "${API_ACCESS_TOKEN:-}" || "${API_ACCESS_TOKEN}" == "change-this-to-a-long-random-token" ]]; then
  echo "对外开放前必须在 .env 设置随机 API_ACCESS_TOKEN。" >&2
  exit 1
fi

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
TUNNEL_MODE="${CLOUDFLARE_TUNNEL_MODE:-quick}"

if [[ "$API_HOST" != "127.0.0.1" && "$API_HOST" != "localhost" ]]; then
  echo "提示：远程隧道只需要 API_HOST=127.0.0.1，已继续使用当前配置 $API_HOST。" >&2
fi

cleanup() {
  trap - INT TERM EXIT
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

echo "启动本机检测 API：http://${API_HOST}:${API_PORT}"
API_HOST="$API_HOST" API_PORT="$API_PORT" .venv/bin/python api_server.py &
API_PID=$!

for _ in {1..50}; do
  if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
  echo "检测 API 未能在预期时间内启动。" >&2
  exit 1
fi

case "$TUNNEL_MODE" in
  quick)
    echo "启动临时公网 HTTPS 隧道。请把输出的 trycloudflare.com 地址和 X-API-Key 一起发给学长。"
    echo "该地址只在本脚本运行期间有效，适合联调，不适合长期部署。"
    cloudflared tunnel --url "http://127.0.0.1:${API_PORT}" &
    TUNNEL_PID=$!
    ;;
  named)
    if [[ -z "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
      echo "named 模式需要在 .env 设置 CLOUDFLARE_TUNNEL_TOKEN。" >&2
      exit 1
    fi
    echo "启动已配置的 Cloudflare 命名隧道。请使用 Cloudflare 中绑定的 HTTPS 域名。"
    cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN" &
    TUNNEL_PID=$!
    ;;
  *)
    echo "CLOUDFLARE_TUNNEL_MODE 只能是 quick 或 named。" >&2
    exit 1
    ;;
esac

wait "$TUNNEL_PID"
