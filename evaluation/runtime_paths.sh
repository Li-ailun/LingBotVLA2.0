#!/usr/bin/env bash
# Resolve deployment-specific Conda paths without binding the repository to one host.

resolve_conda_sh() {
  if [[ -n "${LINGBOT_CONDA_SH:-}" ]]; then
    [[ -f "$LINGBOT_CONDA_SH" ]] || {
      echo "LINGBOT_CONDA_SH does not exist: $LINGBOT_CONDA_SH" >&2
      return 2
    }
    return
  fi

  local conda_exe="${CONDA_EXE:-}"
  if [[ -z "$conda_exe" ]]; then
    conda_exe="$(command -v conda 2>/dev/null || true)"
  fi
  [[ -n "$conda_exe" ]] || {
    echo "Cannot locate Conda; set LINGBOT_CONDA_SH explicitly" >&2
    return 2
  }

  local conda_base
  conda_base="$("$conda_exe" info --base)"
  LINGBOT_CONDA_SH="$conda_base/etc/profile.d/conda.sh"
  [[ -f "$LINGBOT_CONDA_SH" ]] || {
    echo "Conda activation script not found: $LINGBOT_CONDA_SH" >&2
    return 2
  }
  export LINGBOT_CONDA_SH
}

resolve_conda_env_python() {
  local env_name="$1"
  local override_name="$2"
  local current_value="${!override_name:-}"
  if [[ -n "$current_value" ]]; then
    [[ -x "$current_value" ]] || {
      echo "$override_name is not executable: $current_value" >&2
      return 2
    }
    return
  fi

  resolve_conda_sh
  # shellcheck disable=SC1090
  source "$LINGBOT_CONDA_SH"
  local resolved
  resolved="$(conda run -n "$env_name" python -c 'import sys; print(sys.executable)' | tail -n 1)"
  [[ -x "$resolved" ]] || {
    echo "Cannot resolve Python for Conda environment '$env_name'" >&2
    return 2
  }
  printf -v "$override_name" '%s' "$resolved"
  export "$override_name"
}

resolve_lingbot_runtime_paths() {
  resolve_conda_sh
  resolve_conda_env_python RoboTwin ROBOTWIN_PYTHON
  resolve_conda_env_python lingbotvla LINGBOTVLA_PYTHON
}
