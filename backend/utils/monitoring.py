"""
VaultRAG – System Monitoring

Returns CPU, RAM, disk, and GPU metrics.
"""

import logging
import psutil

from config.settings import APP_NAME

logger = logging.getLogger(APP_NAME)


def get_system_metrics() -> dict:
    """Return a dict with current system resource usage."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    metrics = {
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count(logical=True),
        },
        "ram": {
            "percent": mem.percent,
            "used_gb": round(mem.used / 1024 ** 3, 2),
            "total_gb": round(mem.total / 1024 ** 3, 2),
        },
        "disk": {
            "percent": disk.percent,
            "used_gb": round(disk.used / 1024 ** 3, 2),
            "total_gb": round(disk.total / 1024 ** 3, 2),
            "free_gb": round(disk.free / 1024 ** 3, 2),
        },
        "gpu": _get_gpu_metrics(),
    }
    return metrics


def _get_gpu_metrics() -> dict:
    """Try to read GPU metrics via pynvml (NVIDIA). Gracefully degrade."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "available": True,
            "name": name,
            "percent": util.gpu,
            "mem_used_gb": round(mem_info.used / 1024 ** 3, 2),
            "mem_total_gb": round(mem_info.total / 1024 ** 3, 2),
        }
    except Exception:
        return {"available": False, "percent": 0}
