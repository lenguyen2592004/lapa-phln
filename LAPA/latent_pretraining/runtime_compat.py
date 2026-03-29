import os
import sys
import traceback

import jax
from tux import StreamingCheckpointer


_JAX_BACKEND_ERROR_HINTS = (
    "cudnn_status",
    "dnn library initialization failed",
    "xlaruntimeerror",
    "failed_precondition",
    "cuda_error",
    "could not create cudnn handle",
    "unable to load cupti",
    "no gpu/visible devices",
    "unable to initialize backend",
)


def _format_exception(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).lower()


def is_probable_jax_backend_error(exc: Exception) -> bool:
    formatted = _format_exception(exc)
    return any(hint in formatted for hint in _JAX_BACKEND_ERROR_HINTS)


def maybe_restart_with_cpu(exc: Exception, context: str) -> bool:
    if not is_probable_jax_backend_error(exc):
        return False

    if os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        return False

    if os.environ.get("LAPA_CPU_FALLBACK_DONE") == "1":
        return False

    print(f"[{context}] Detected JAX/CUDA backend error. Retrying once with JAX_PLATFORMS=cpu.")
    new_env = os.environ.copy()
    new_env["JAX_PLATFORMS"] = "cpu"
    new_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    new_env["LAPA_CPU_FALLBACK_DONE"] = "1"

    try:
        os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)
    except OSError:
        return False
    return True


def get_checkpoint_buffer_size_bytes() -> int:
    env_value = os.environ.get("LAPA_CHECKPOINT_BUFFER_GB", "").strip()
    if env_value:
        try:
            # Very small buffers can trigger msgpack BufferFull while reading checkpoints.
            return max(4, int(float(env_value))) * (2 ** 30)
        except ValueError:
            pass

    # CPU fallback should use a smaller streaming buffer to avoid OOM.
    platform = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if platform == "cpu":
        return 8 * (2 ** 30)

    try:
        if any(device.platform == "gpu" for device in jax.devices()):
            return 8 * (2 ** 30)
    except Exception:
        pass
    return 8 * (2 ** 30)


def _is_buffer_full_error(exc: Exception) -> bool:
    formatted = _format_exception(exc)
    return (
        "bufferfull" in formatted
        or "msgpack.exceptions.bufferfull" in formatted
        or "max_buffer_size" in formatted
    )


def load_checkpoint_with_adaptive_buffer(load_checkpoint: str):
    base = get_checkpoint_buffer_size_bytes()
    sizes = [base]
    for gb in (8, 12, 16, 24, 32):
        size = gb * (2 ** 30)
        if size not in sizes:
            sizes.append(size)

    last_exc = None
    for max_buffer_size in sizes:
        try:
            return StreamingCheckpointer.load_trainstate_checkpoint(
                load_checkpoint,
                disallow_trainstate=True,
                max_buffer_size=max_buffer_size,
            )
        except Exception as exc:
            if _is_buffer_full_error(exc):
                print(
                    f"[checkpoint] Buffer too small ({max_buffer_size // (2 ** 30)}GB). "
                    "Retrying with larger buffer..."
                )
                last_exc = exc
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load checkpoint for an unknown reason.")


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return -1


def ensure_enough_host_memory_for_7b() -> None:
    if os.environ.get("LAPA_SKIP_MEMORY_CHECK") == "1":
        return

    available = _available_memory_bytes()
    if available < 0:
        return

    min_gb = float(os.environ.get("LAPA_MIN_AVAILABLE_GB", "12"))
    min_bytes = int(min_gb * (2 ** 30))
    if available >= min_bytes:
        return

    avail_gb = available / (2 ** 30)
    raise RuntimeError(
        "Insufficient host memory to load LAPA-7B checkpoint. "
        f"Available RAM: {avail_gb:.2f} GB, required at least: {min_gb:.1f} GB. "
        "Increase WSL memory/swap (or run on a higher-memory machine) and retry. "
        "You may override this check by setting LAPA_SKIP_MEMORY_CHECK=1."
    )
