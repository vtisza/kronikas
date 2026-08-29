#!/usr/bin/env bash
# Put a working kronikas installation in a local virtual environment.
#
#   ./setup_kronikas.sh                 # install the released package
#   ./setup_kronikas.sh --source        # clone the repo and install from it
#   ./setup_kronikas.sh --dir ~/forecast
#
# Prints the interpreter to use for every later command. Safe to re-run.
set -euo pipefail

REPO_URL="https://github.com/vtisza/kronikas.git"
WORKDIR="$(pwd)"
FROM_SOURCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --source) FROM_SOURCE=1; shift ;;
    --dir) WORKDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$WORKDIR"
cd "$WORKDIR"
VENV="$WORKDIR/.kronikas-venv"

# kronikas supports Python 3.10-3.12. Pick the newest interpreter in range
# rather than whatever "python3" happens to be.
PYBIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)'; then
      PYBIN="$candidate"
      break
    fi
  fi
done
if [ -z "$PYBIN" ]; then
  echo "No Python between 3.10 and 3.12 was found. Install one, then re-run." >&2
  echo "macOS:  brew install python@3.12    Ubuntu: sudo apt install python3.12-venv" >&2
  exit 1
fi
echo "Using $($PYBIN --version) from $(command -v "$PYBIN")"

if [ ! -d "$VENV" ]; then
  "$PYBIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
[ -x "$PY" ] || PY="$VENV/Scripts/python.exe"   # Git Bash on Windows

"$PY" -m pip install --quiet --upgrade pip

if [ "$FROM_SOURCE" -eq 1 ]; then
  if [ ! -d "$WORKDIR/kronikas/.git" ]; then
    echo "Cloning $REPO_URL ..."
    git clone --depth 1 "$REPO_URL" "$WORKDIR/kronikas"
  else
    echo "Repository already present; pulling the latest commit."
    git -C "$WORKDIR/kronikas" pull --ff-only
  fi
  echo "Installing kronikas from source (a few minutes: PyMC is large) ..."
  "$PY" -m pip install --quiet -e "$WORKDIR/kronikas"
else
  echo "Installing kronikas from PyPI (a few minutes: PyMC is large) ..."
  "$PY" -m pip install --quiet --upgrade kronikas
fi

"$PY" - <<'CHECK'
import kronikas

print(f"kronikas {kronikas.__version__} is ready.")
CHECK

KRONIKAS="$(dirname "$PY")/kronikas"

cat <<EOF

Done. Use this command for everything that follows:

  $KRONIKAS

For example:

  $KRONIKAS skill install                  # hand the workflow to an assistant
  $KRONIKAS guided forecast.yaml --check   # validate your settings
  $KRONIKAS guided forecast.yaml           # run the forecast

The interpreter, if you need it directly:

  $PY
EOF
