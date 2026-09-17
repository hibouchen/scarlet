#!/usr/bin/env bash
set -euo pipefail

work_dir="${SCARLET_WORK_DIR:-/home/jovyan/work}"
tutorial_source="/opt/scarlet/tutorial.ipynb"
tutorial_target="${work_dir}/tutorial.ipynb"

if [[ "$(id -u)" == "0" ]]; then
    mkdir -p "${work_dir}"
    chown jovyan:jovyan "${work_dir}"

    if [[ ! -e "${tutorial_target}" ]]; then
        install -o jovyan -g jovyan -m 644 "${tutorial_source}" "${tutorial_target}"
    fi

    exec gosu jovyan jupyterhub-singleuser --ServerApp.root_dir="${work_dir}" "$@"
fi

mkdir -p "${work_dir}"
if [[ ! -e "${tutorial_target}" ]]; then
    cp "${tutorial_source}" "${tutorial_target}"
fi

exec jupyterhub-singleuser --ServerApp.root_dir="${work_dir}" "$@"
