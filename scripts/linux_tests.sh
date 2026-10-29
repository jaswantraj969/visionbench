#!/bin/sh
# Run lint and the full backend test suite on Linux (CPU torch) inside the pinned Python image,
# as a non-root user (permission tests are meaningless as root). Exercises the POSIX code paths
# (fcntl worker lock, psutil process handling) that the Windows runs do not.
#
#   docker run --rm -v "<clean export of the repo>:/src:ro" -v vb-pip-cache:/pip-cache \
#     python:3.11-slim@sha256:<digest from the Dockerfile> sh /src/scripts/linux_tests.sh
#
# The optional named volume at /pip-cache keeps downloaded wheels between runs (pip still checks
# every wheel against the index's hash).
set -eu
INDEX="https://download.pytorch.org/whl/cpu"
useradd --create-home tester
mkdir -p /pip-cache && chown tester:tester /pip-cache
cp -r /src /home/tester/src
chown -R tester:tester /home/tester/src
su tester -s /bin/sh -c "
  set -eu
  cd /home/tester/src
  python -m venv /home/tester/venv
  . /home/tester/venv/bin/activate
  export PIP_CACHE_DIR=/pip-cache
  python -m pip install -q -r requirements/lock-dev-cpu-linux.txt --extra-index-url $INDEX
  python -m pip install -q -e . --no-deps
  python -c 'import platform, torch; print(platform.platform(), \"python\", platform.python_version(), \"torch\", torch.__version__)'
  ruff check .
  ruff format --check .
  python -m pytest -q -rsw -p no:cacheprovider
"
