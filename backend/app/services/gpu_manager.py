"""
GPU Manager - ROCm GPU monitoring, HIP↔ROCm device mapping, and model→GPU assignment.

Uses rocm-smi for accurate VRAM data (torch.cuda reports only its own process).
Persists model→GPU assignments to /app/data/gpuSelector/.

HIP_VISIBLE_DEVICES vs ROCm device IDs:
  - ROCm device IDs (GPU[0]-GPU[7] in rocm-smi) are physical device indices.
  - HIP_VISIBLE_DEVICES maps those to process-local cuda:0, cuda:1, etc.
  - When HIP_VISIBLE_DEVICES=4, that GPU becomes cuda:0 inside the process.
  - The mapping can change between reboots depending on PCIe enumeration.

Usage:
    from app.services.gpu_manager import (
        get_gpu_status,        # Full status for admin panel
        get_model_assignment,  # Which GPU a model should use
        set_model_assignment,  # Assign a model to a GPU
        get_hip_env,           # Get HIP_VISIBLE_DEVICES value for a model
        apply_gpu_assignment,  # Set env before model load
    )
"""
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Persistence directory
GPU_SELECTOR_DIR = Path(os.getenv("GPU_SELECTOR_DIR", "/app/data/gpuSelector"))
ASSIGNMENTS_FILE = GPU_SELECTOR_DIR / "assignments.json"

# Known model keys that can be assigned to GPUs
MODEL_KEYS = [
    "vllm",         # vLLM inference server
    "image_gen",    # Z-Image-Turbo / diffusion model
    "tts",          # Text-to-Speech (Kokoro/etc)
    "stt",          # Speech-to-Text (Whisper)
    "rag",          # RAG embeddings (sentence-transformers)
    "faiss",        # FAISS GPU index
]

MODEL_LABELS = {
    "vllm": "LLM (vLLM)",
    "image_gen": "Image Generation",
    "tts": "Text-to-Speech",
    "stt": "Speech-to-Text (Whisper)",
    "rag": "RAG Embeddings",
    "faiss": "FAISS Index",
}


# ─── rocm-smi parsing ───────────────────────────────────────────────────────

def _run_rocm_smi(*args: str) -> Optional[str]:
    """Run rocm-smi with args, return stdout or None."""
    try:
        result = subprocess.run(
            ["rocm-smi", *args],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"rocm-smi failed: {e}")
        return None


def _parse_rocm_smi_memuse(output: str) -> Dict[int, Dict[str, Any]]:
    """Parse rocm-smi --showmemuse output into per-GPU dicts."""
    gpus: Dict[int, Dict[str, Any]] = {}
    for line in output.splitlines():
        line = line.strip()
        m = re.match(r'GPU\[(\d+)\]\s*:\s*(.+?):\s*(.+)', line)
        if not m:
            continue
        gpu_id = int(m.group(1))
        key = m.group(2).strip()
        val = m.group(3).strip()
        if gpu_id not in gpus:
            gpus[gpu_id] = {"rocm_id": gpu_id}
        if "VRAM%" in key:
            try:
                gpus[gpu_id]["vram_pct"] = int(val)
            except ValueError:
                gpus[gpu_id]["vram_pct"] = 0
        elif "Read/Write Activity" in key:
            try:
                gpus[gpu_id]["mem_activity_pct"] = int(val)
            except ValueError:
                gpus[gpu_id]["mem_activity_pct"] = 0
        elif "Avg. Memory Bandwidth" in key:
            try:
                gpus[gpu_id]["avg_bandwidth"] = int(val)
            except ValueError:
                gpus[gpu_id]["avg_bandwidth"] = 0
    return gpus


def _parse_rocm_smi_full() -> List[Dict[str, Any]]:
    """Parse rocm-smi for device names, VRAM usage, and totals."""
    gpus: Dict[int, Dict[str, Any]] = {}

    # Get memory use percentages
    mem_output = _run_rocm_smi("--showmemuse")
    if mem_output:
        gpus = _parse_rocm_smi_memuse(mem_output)

    # Get total VRAM per GPU
    vram_output = _run_rocm_smi("--showmeminfo", "vram")
    if vram_output:
        for line in vram_output.splitlines():
            line = line.strip()
            m = re.match(r'GPU\[(\d+)\]\s*:\s*(.+?):\s*(.+)', line)
            if not m:
                continue
            gpu_id = int(m.group(1))
            key = m.group(2).strip()
            val = m.group(3).strip()
            if gpu_id not in gpus:
                gpus[gpu_id] = {"rocm_id": gpu_id}
            if "Total Memory" in key and "Used" not in key:
                try:
                    gpus[gpu_id]["total_bytes"] = int(val)
                except ValueError:
                    pass
            elif "Total Used" in key:
                try:
                    gpus[gpu_id]["used_bytes"] = int(val)
                except ValueError:
                    pass

    # Get device names
    id_output = _run_rocm_smi("--showproductname")
    if id_output:
        for line in id_output.splitlines():
            m = re.match(r'GPU\[(\d+)\]\s*:\s*Card [Ss]eries:\s*(.+)', line)
            if not m:
                m = re.match(r'GPU\[(\d+)\]\s*:\s*Card [Mm]odel:\s*(.+)', line)
            if m:
                gpu_id = int(m.group(1))
                if gpu_id not in gpus:
                    gpus[gpu_id] = {"rocm_id": gpu_id}
                gpus[gpu_id]["name"] = m.group(2).strip()

    # Get PCIe bus IDs for device mapping
    bus_output = _run_rocm_smi("--showbus")
    if bus_output:
        for line in bus_output.splitlines():
            m = re.match(r'GPU\[(\d+)\]\s*:\s*PCI Bus:\s*(\S+)', line)
            if m:
                gpu_id = int(m.group(1))
                if gpu_id in gpus:
                    gpus[gpu_id]["pci_bus"] = m.group(2).strip()

    # Build sorted list
    result = []
    for gpu_id in sorted(gpus.keys()):
        g = gpus[gpu_id]
        total_b = g.get("total_bytes", 0)
        used_b = g.get("used_bytes", 0)
        total_gb = total_b / (1024**3) if total_b else 0
        used_gb = used_b / (1024**3) if used_b else 0
        vram_pct = g.get("vram_pct", 0)

        if total_b and not vram_pct:
            vram_pct = round((used_b / total_b) * 100) if total_b > 0 else 0

        result.append({
            "rocm_id": gpu_id,
            "name": g.get("name", f"AMD GPU {gpu_id}"),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(total_gb - used_gb, 2),
            "vram_pct": vram_pct,
            "mem_activity_pct": g.get("mem_activity_pct", 0),
            "avg_bandwidth": g.get("avg_bandwidth", 0),
            "pci_bus": g.get("pci_bus", ""),
        })

    return result


# ─── HIP ↔ ROCm device mapping ──────────────────────────────────────────────

def _run_cmd(*args: str) -> Optional[str]:
    """Run a command, return stdout or None."""
    try:
        result = subprocess.run(
            list(args),
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Command {args[0]} failed: {e}")
        return None


def _parse_amd_smi_list() -> Dict[str, Any]:
    """Parse `amd-smi list -e` for authoritative HIP_ID ↔ GPU (ROCm) mapping.
    
    Example output:
        GPU: 0
            BDF: 0000:11:00.0
            HIP_ID: 2
            ...
        GPU: 1
            BDF: 0000:26:00.0
            HIP_ID: 5
            ...
    
    GPU: N is the ROCm device index (matches rocm-smi GPU[N]).
    HIP_ID: M is what HIP_VISIBLE_DEVICES uses.
    """
    output = _run_cmd("amd-smi", "list", "-e")
    if not output:
        return {"hip_to_rocm": {}, "rocm_to_hip": {}, "gpu_details": {}}
    
    hip_to_rocm: Dict[int, int] = {}
    rocm_to_hip: Dict[int, int] = {}
    gpu_details: Dict[int, Dict[str, Any]] = {}  # rocm_id → details
    
    current_gpu: Optional[int] = None
    current_detail: Dict[str, Any] = {}
    
    for line in output.splitlines():
        line_stripped = line.strip()
        
        # "GPU: 0" — start of new GPU block
        m = re.match(r'^GPU:\s*(\d+)', line_stripped)
        if m:
            # Save previous GPU
            if current_gpu is not None:
                gpu_details[current_gpu] = current_detail
            current_gpu = int(m.group(1))
            current_detail = {"rocm_id": current_gpu}
            continue
        
        if current_gpu is None:
            continue
        
        # Parse key: value pairs
        if ':' in line_stripped:
            key, _, val = line_stripped.partition(':')
            key = key.strip()
            val = val.strip()
            
            if key == "HIP_ID":
                try:
                    hip_id = int(val)
                    current_detail["hip_id"] = hip_id
                    hip_to_rocm[hip_id] = current_gpu
                    rocm_to_hip[current_gpu] = hip_id
                except ValueError:
                    pass
            elif key == "BDF":
                current_detail["bdf"] = val
            elif key == "UUID":
                current_detail["uuid"] = val
            elif key == "HIP_UUID":
                current_detail["hip_uuid"] = val
            elif key == "CARD":
                current_detail["card"] = val
            elif key == "RENDER":
                current_detail["render"] = val
    
    # Save last GPU
    if current_gpu is not None:
        gpu_details[current_gpu] = current_detail
    
    logger.info(f"[GPU] amd-smi device map: hip_to_rocm={hip_to_rocm}, rocm_to_hip={rocm_to_hip}")
    
    return {
        "hip_to_rocm": hip_to_rocm,
        "rocm_to_hip": rocm_to_hip,
        "gpu_details": gpu_details,
    }


def get_device_map() -> Dict[str, Any]:
    """Build HIP device index ↔ ROCm device ID mapping.

    Primary: `amd-smi list -e` (authoritative HIP_ID ↔ GPU mapping).
    Fallback: PCIe bus correlation, HIP_VISIBLE_DEVICES, or 1:1.
    
    HIP_VISIBLE_DEVICES takes a HIP_ID (not ROCm GPU index).
    rocm-smi shows GPU[N] where N is the ROCm index.
    These are NOT the same — `amd-smi list -e` shows the mapping.
    """
    # Try amd-smi first — this is the authoritative source
    amd_smi = _parse_amd_smi_list()
    if amd_smi["hip_to_rocm"]:
        return amd_smi

    # Fallback: try rocm-smi --showbus + torch PCI correlation
    hip_to_rocm: Dict[int, int] = {}
    rocm_to_hip: Dict[int, int] = {}

    rocm_buses: Dict[str, int] = {}
    bus_output = _run_rocm_smi("--showbus")
    if bus_output:
        for line in bus_output.splitlines():
            m = re.match(r'GPU\[(\d+)\]\s*:\s*PCI Bus:\s*(\S+)', line)
            if m:
                rocm_id = int(m.group(1))
                bus = m.group(2).strip().lower()
                rocm_buses[bus] = rocm_id

    if rocm_buses:
        try:
            import torch
            if torch.cuda.is_available():
                for hip_id in range(torch.cuda.device_count()):
                    try:
                        pci = getattr(torch.cuda, 'get_device_pci_bus_id', None)
                        if pci:
                            hip_bus = pci(hip_id).lower()
                            if hip_bus in rocm_buses:
                                rocm_id = rocm_buses[hip_bus]
                                hip_to_rocm[hip_id] = rocm_id
                                rocm_to_hip[rocm_id] = hip_id
                    except Exception:
                        pass
        except ImportError:
            pass

    if hip_to_rocm:
        return {"hip_to_rocm": hip_to_rocm, "rocm_to_hip": rocm_to_hip, "gpu_details": {}}

    # Last fallback: assume 1:1
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                hip_to_rocm[i] = i
                rocm_to_hip[i] = i
    except ImportError:
        rocm_gpus = _parse_rocm_smi_full()
        for g in rocm_gpus:
            rid = g["rocm_id"]
            hip_to_rocm[rid] = rid
            rocm_to_hip[rid] = rid

    return {"hip_to_rocm": hip_to_rocm, "rocm_to_hip": rocm_to_hip, "gpu_details": {}}


# ─── Model → GPU assignments ────────────────────────────────────────────────

def _ensure_dir():
    GPU_SELECTOR_DIR.mkdir(parents=True, exist_ok=True)


def load_assignments() -> Dict[str, Any]:
    """Load model→GPU assignments from disk."""
    _ensure_dir()
    if ASSIGNMENTS_FILE.exists():
        try:
            with open(ASSIGNMENTS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load GPU assignments: {e}")
    return {}


def save_assignments(assignments: Dict[str, Any]):
    """Save model→GPU assignments to disk."""
    _ensure_dir()
    try:
        with open(ASSIGNMENTS_FILE, 'w') as f:
            json.dump(assignments, f, indent=2)
        logger.info(f"Saved GPU assignments: {assignments}")
    except OSError as e:
        logger.error(f"Failed to save GPU assignments: {e}")


def get_model_assignment(model_key: str) -> Optional[int]:
    """Get the ROCm GPU ID assigned to a model, or None if unassigned."""
    assignments = load_assignments()
    entry = assignments.get(model_key)
    if entry and isinstance(entry, dict):
        return entry.get("rocm_id")
    elif entry is not None:
        try:
            return int(entry)
        except (ValueError, TypeError):
            pass
    return None


def set_model_assignment(model_key: str, rocm_id: Optional[int]):
    """Assign a model to a specific ROCm GPU (or None to unassign)."""
    assignments = load_assignments()
    if rocm_id is not None:
        assignments[model_key] = {"rocm_id": rocm_id, "label": f"GPU {rocm_id}"}
    else:
        assignments.pop(model_key, None)
    save_assignments(assignments)


def get_hip_env(model_key: str) -> Optional[str]:
    """Get the HIP_VISIBLE_DEVICES value for a model.

    CRITICAL: HIP_VISIBLE_DEVICES takes HIP_IDs, NOT ROCm GPU indices.
    GPU:0 in rocm-smi might be HIP_ID:2. We must translate.
    
    Returns the HIP_ID as a string, or None if no assignment.
    """
    rocm_id = get_model_assignment(model_key)
    if rocm_id is None:
        return None
    
    # Get the device map to translate rocm_id → hip_id
    device_map = get_device_map()
    rocm_to_hip = device_map.get("rocm_to_hip", {})
    
    # Look up the HIP_ID for this ROCm GPU
    hip_id = rocm_to_hip.get(rocm_id)
    if hip_id is None:
        # Also try string key (JSON serialization may stringify keys)
        hip_id = rocm_to_hip.get(str(rocm_id))
    
    if hip_id is not None:
        logger.info(f"[GPU] {model_key}: rocm_id={rocm_id} → HIP_ID={hip_id}")
        return str(hip_id)
    else:
        # No mapping found — warn and fall back to rocm_id (may be wrong!)
        logger.warning(f"[GPU] {model_key}: No HIP_ID mapping for rocm_id={rocm_id}, "
                      f"falling back to rocm_id (may be incorrect!)")
        return str(rocm_id)


# ─── Composite status for admin panel ────────────────────────────────────────

def get_gpu_status() -> Dict[str, Any]:
    """Get full GPU status for admin panel."""
    gpus = _parse_rocm_smi_full()

    # Fallback to torch if rocm-smi not available
    if not gpus:
        gpus = _get_torch_gpu_info()

    device_map = get_device_map()
    assignments = load_assignments()

    is_rocm = False
    rocm_version = None
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip:
            is_rocm = True
            rocm_version = torch.version.hip
    except ImportError:
        pass

    if not is_rocm:
        ver_output = _run_rocm_smi("--version")
        if ver_output:
            is_rocm = True

    return {
        "available": len(gpus) > 0,
        "count": len(gpus),
        "rocm": is_rocm,
        "rocm_version": rocm_version,
        "gpus": gpus,
        "device_map": device_map,
        "assignments": assignments,
        "model_keys": MODEL_KEYS,
        "model_labels": MODEL_LABELS,
    }


def _get_torch_gpu_info() -> List[Dict[str, Any]]:
    """Fallback: get GPU info from torch when rocm-smi isn't available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)
            used = max(reserved, allocated)
            gpus.append({
                "rocm_id": i,
                "name": torch.cuda.get_device_name(i),
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round((total - used) / (1024**3), 2),
                "vram_pct": round((used / total) * 100) if total > 0 else 0,
                "mem_activity_pct": 0,
                "avg_bandwidth": 0,
                "pci_bus": "",
            })
        return gpus
    except ImportError:
        return []


# ─── Helper for services to set their GPU before model load ──────────────────

def apply_gpu_assignment(model_key: str) -> Optional[str]:
    """Set HIP_VISIBLE_DEVICES for a model before loading.

    Call this BEFORE importing torch or loading any model.
    When HIP_VISIBLE_DEVICES=N, that ROCm GPU becomes cuda:0.

    Returns "cuda:0" if assignment exists, None otherwise.
    """
    hip_val = get_hip_env(model_key)
    if hip_val is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = hip_val
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_val
        logger.info(f"[GPU] {model_key}: HIP_VISIBLE_DEVICES={hip_val}")
        return "cuda:0"
    return None
