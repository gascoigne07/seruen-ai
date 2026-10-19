#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Создаю окружение Python 3.13…"
  make venv
fi
exec .venv/bin/python -m streamlit run seruen/app/streamlit_app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
