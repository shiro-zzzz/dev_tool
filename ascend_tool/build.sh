#!/bin/bash
# SPDX-License-Identifier: MIT
# Description: ascend_tool build entry

set -e

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)"
export ROOT_PATH="${SCRIPT_PATH}"

export SRC_PATH="${ROOT_PATH}/src"
export SCRIPTS_PATH="${ROOT_PATH}/scripts"
export BUILD_OUT_PATH="${ROOT_PATH}/output"
export TEST_PATH="${ROOT_PATH}/test"
export BUILD_PATH="${ROOT_PATH}/build_tmp"

export BUILD_TYPE="Release"

print_help() {
  echo "
Usage:
  $0 <opt>...

Options:
  -x  Extract the run package
  -c  Target SOC VERSION (ascend910_93, ascend910b4, all)
  -d  Enable debug
  -t  Enable UT build (disabled by default)
  -p  Enable pybind build only
  -r  Enable code coverage
  -h  Show help
"
}

bash "${SCRIPTS_PATH}/build.sh" "$@"

