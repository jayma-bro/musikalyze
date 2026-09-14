"""Runtime information used by the public processing entry points."""

from __future__ import annotations


def compute_device() -> str:
    """Return the device visible to the TensorFlow runtime.

    This reports visibility, not a per-operation placement guarantee. The
    latter depends on the concrete Essentia graph and TensorFlow device
    placement.
    """
    try:
        import tensorflow as tf

        physical = tf.config.list_physical_devices("GPU")
        logical = tf.config.list_logical_devices("GPU")
        if physical or logical:
            return "GPU-visible"
    except Exception:  # noqa: BLE001
        return "CPU-only/undetected"
    return "CPU-only/undetected"


def configure_tensorflow_memory() -> str:
    """Enable progressive GPU allocation before the first TensorFlow model.

    TensorFlow otherwise may reserve most or all VRAM immediately. Memory
    growth must be configured before the runtime initializes; if a notebook
    already initialized TensorFlow, the setting cannot be changed until its
    kernel is restarted.
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # TensorFlow has already initialized in this process.
                continue
        return "GPU-visible" if gpus else "CPU-only/undetected"
    except Exception:  # noqa: BLE001
        return "CPU-only/undetected"


def report_compute_device() -> str:
    """Print and return the inference device, once per Python process."""
    global _reported_device
    if _reported_device is None:
        _reported_device = configure_tensorflow_memory()
        print(
            "musikalyze: TensorFlow device visibility: "
            f"{_reported_device} (operation placement is handled by TensorFlow)"
        )
    return _reported_device


_reported_device: str | None = None
