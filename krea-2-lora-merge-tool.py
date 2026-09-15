#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Krea 2 LoRA Merge Tool (ComfyUI)
--------------------------------
Merges one or more LoRA / LoKr ("LyCORIS") files INTO a base diffusion model,
tensor by tensor (low RAM), with optional GPU (CUDA) and CPU fallback.

Supports both common LoRA layouts:
  - LoKr (ai-toolkit / LyCORIS): ``<module>.lokr_w1`` + ``.lokr_w2`` (+ optional
    ``.lokr_w1_a/.lokr_w1_b``, ``.lokr_w2_a/.lokr_w2_b``, ``.lokr_t2``) + ``.alpha``
  - Classic LoRA: ``<module>.lora_A/.lora_B`` or ``.lora_down/.lora_up``
    (optionally with a ``.default`` segment), including kohya ``lora_unet_`` /
    ``lora_te_`` prefixes.

The base weight is dequantized to fp32 (handles pure fp8, fp8+scale and
int8+scale ComfyUI layouts), the LoRA delta is added and the result is
re-quantized to the requested output format.

Output formats:
  - fp8  : float8_e4m3fn (default - matches typical Krea 2 checkpoints)
  - fp16 : float16
  - bf16 : bfloat16
  - int8 : int8 + per-row ``_scale`` + ``.comfy_quant`` descriptor
  - auto : preserves each tensor's original dtype (requantizes int8)

Usage:
  python krea-2-lora-merge-tool-v1.py                       # GUI
  python krea-2-lora-merge-tool-v1.py --inspect base.safetensors lora.safetensors
  python krea-2-lora-merge-tool-v1.py --model base.safetensors \\
      --lora lora.safetensors:1.0 --out merged.safetensors --format fp8
"""

import gc
import json
import math
import os
import queue
import re
import struct
import threading
import time
import traceback
import argparse

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, font as tkfont
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

import torch

try:
    from safetensors import safe_open
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DTYPES_ST = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
DTYPES_ST_REV = {v: k for k, v in DTYPES_ST.items()}
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
# dtype used in merge calculations
MATH_DTYPE = torch.float32
# byte limit for processing a tensor on GPU (avoids VRAM OOM)
GPU_MAX_TENSOR_BYTES = 512 * 1024 * 1024

# prefixes stripped from base model keys so LoRA module names can be matched
BASE_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# kohya prefixes found on LoRA module names (underscore-joined form)
KOHYA_PREFIXES = ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")
# recognized trailing tokens on a LoRA key -> logical role
LORA_ROLE_TOKENS = {
    "lora_A": "A", "lora_B": "B",
    "lora_down": "A", "lora_up": "B",
    "lokr_w1": "W1", "lokr_w1_a": "W1A", "lokr_w1_b": "W1B",
    "lokr_w2": "W2", "lokr_w2_a": "W2A", "lokr_w2_b": "W2B",
    "lokr_t2": "T2",
    "alpha": "ALPHA",
}
# segments ignored when scanning for the role token
LORA_NOISE_TOKENS = {"weight", "default", "bias"}
# sane alpha range; anything outside is treated as "alpha == rank" (scale 1)
ALPHA_MAX = 4096.0



# ----------------------------------------------------------------------------
# Safetensors header reading (without loading weights)
# ----------------------------------------------------------------------------
def read_header(path):
    """Reads the JSON header of a .safetensors file. Returns (header, header_len)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def tensor_infos(header):
    """Returns {name: info} without the __metadata__."""
    return {k: v for k, v in header.items() if k != "__metadata__"}


def human_size(nbytes):
    """Formats a byte count as a short human readable string."""
    try:
        nbytes = float(nbytes)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


# ----------------------------------------------------------------------------
# Individual tensor reading (streaming, without loading the whole file)
# ----------------------------------------------------------------------------
class TensorReader:
    """Reads tensors individually from a .safetensors via mmap."""

    def __init__(self, path):
        self.path = path
        header, self.header_len = read_header(path)
        self.header = header
        self.infos = tensor_infos(header)
        self._file = None
        self._mmap = None

    def _ensure_open(self):
        if self._file is None:
            import mmap
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, name):
        """Reads a tensor as torch.Tensor (original dtype)."""
        self._ensure_open()
        info = self.infos[name]
        dt = DTYPES_ST[info["dtype"]]
        start, end = info["data_offsets"]
        nbytes = end - start
        buf = self._mmap[self.header_len + start: self.header_len + end]
        t = torch.frombuffer(bytearray(buf), dtype=dt).reshape(info["shape"])
        return t

    def read_raw(self, name):
        """Reads raw bytes of a tensor (to copy without converting)."""
        self._ensure_open()
        info = self.infos[name]
        start, end = info["data_offsets"]
        return self._mmap[self.header_len + start: self.header_len + end]

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None


# ----------------------------------------------------------------------------
# Dequantization / Quantization (ComfyUI int8 format)
# ----------------------------------------------------------------------------
def dequantize_int8(weight_i8, scale, math_dtype=MATH_DTYPE):
    """weight (I8, [out, in]) * scale (F32, [out, 1]) -> math_dtype."""
    return weight_i8.to(math_dtype) * scale.to(math_dtype)


def quantize_int8(weight_f32, math_dtype=MATH_DTYPE):
    """Quantizes to int8 with per-row scale. Returns (i8, scale_f32)."""
    w = weight_f32.to(math_dtype)
    if w.dim() == 1:
        w = w.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    if squeeze:
        q = q.squeeze(1)
        scale = scale.squeeze(1).unsqueeze(1)  # scale [n,1] like the originals
    else:
        scale = scale.reshape(-1, 1)
    return q, scale.to(torch.float32)


def cast_float(t, dtype_key):
    """Converts a float tensor to the target dtype."""
    if dtype_key == "F8_E4M3":
        return t.to(torch.float8_e4m3fn)
    if dtype_key == "F8_E5M2":
        return t.to(torch.float8_e5m2)
    return t.to(DTYPES_ST[dtype_key])


def tensor_to_bytes(t):
    """Serializes a tensor to bytes in safetensors format (little-endian)."""
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


def _comfy_quant_bytes():
    """Generates the comfy_quant descriptor (JSON bytes) for int8_tensorwise."""
    desc = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    return json.dumps(desc, separators=(",", ":")).encode("utf-8")


# ----------------------------------------------------------------------------
# LoRA parsing (LoKr / LoRA / LoCon, ai-toolkit & kohya conventions)
# ----------------------------------------------------------------------------
def normalize_base_key(key):
    """Strips the common diffusion-model prefix used by different exporters."""
    for p in BASE_KEY_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def parse_lora_role(key):
    """Splits a LoRA key into (module, role).

    ``diffusion_model.blocks.0.attn.wq.lokr_w1``      -> (..wq, "W1")
    ``lora_unet_blocks_0_attn_wq.lora_down.weight``   -> (..wq, "A")
    ``...wq.lora_A.default``                          -> (..wq, "A")
    """
    parts = key.split(".")
    for i in range(len(parts) - 1, max(len(parts) - 4, -1), -1):
        token = parts[i]
        if token in LORA_NOISE_TOKENS:
            continue
        role = LORA_ROLE_TOKENS.get(token)
        if role is not None:
            return ".".join(parts[:i]), role
    return None, None


def _lokr_factor(roles, wrole, arole, brole, reader, math_dtype):
    """Returns w1/w2, reconstructing it from its low-rank factors if split."""
    if wrole in roles:
        return reader.read(roles[wrole]).to(math_dtype)
    if arole in roles and brole in roles:
        a = reader.read(roles[arole]).to(math_dtype)
        b = reader.read(roles[brole]).to(math_dtype)
        return a @ b
    return None


def make_weight_cp(t, wa, wb):
    """Reconstructs the CP-decomposed w2 factor (LyCORIS / LoKr)."""
    return torch.einsum("i j k l, i p, j r -> p r k l", t, wa, wb)


def _lora_matmul(a, b):
    """Computes B @ A for a standard LoRA, supporting 1x1 Conv layouts."""
    if a.dim() == 4 and tuple(a.shape[2:]) == (1, 1):
        a = a[:, :, 0, 0]
        b = b[:, :, 0, 0]
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"unsupported LoRA tensor ranks: A={list(a.shape)} B={list(b.shape)} "
            f"(only Linear and 1x1 Conv are supported)")
    return b @ a


def _lora_delta(info, reader, math_dtype=MATH_DTYPE):
    """Builds the (unweighted) LoRA delta in math dtype for one module.

    The alpha/rank scale is folded in; the per-LoRA strength is applied by
    the caller. Returns None if the module has no usable tensors.
    """
    roles = info["roles"]
    if info["kind"] == "kron":
        w1 = _lokr_factor(roles, "W1", "W1A", "W1B", reader, math_dtype)
        w2 = _lokr_factor(roles, "W2", "W2A", "W2B", reader, math_dtype)
        if w1 is None or w2 is None:
            return None
        if "T2" in roles:
            t2 = reader.read(roles["T2"]).to(math_dtype)
            wa = reader.read(roles["W2A"]).to(math_dtype)
            wb = reader.read(roles["W2B"]).to(math_dtype)
            w2 = make_weight_cp(t2, wa, wb)
        if w2.dim() == 4:  # convolutional kron
            w1 = w1.unsqueeze(2).unsqueeze(3)
        delta = torch.kron(w1, w2)
    else:
        a = reader.read(roles["A"]).to(math_dtype)
        b = reader.read(roles["B"]).to(math_dtype)
        delta = _lora_matmul(a, b)
    return delta * float(info["alpha"])


class LoraReader:
    """Parses a LoRA/LoKr safetensors file into per-module descriptors."""

    def __init__(self, path):
        self.path = path
        self.reader = TensorReader(path)
        self.metadata = self.reader.header.get("__metadata__", {})
        self.modules = {}        # normalized module name -> info dict
        self._parse()

    # -------------------------------------------------------------- parsing
    def _parse(self):
        grouped = {}
        for key in self.reader.infos:
            module, role = parse_lora_role(key)
            if role is None:
                continue
            grouped.setdefault(module, {})[role] = key

        for module, roles in grouped.items():
            if "W1" in roles or "W2" in roles or "W2A" in roles:
                kind = "kron"
            elif "A" in roles and "B" in roles:
                kind = "lora"
            else:
                continue
            rank = self._rank_of(roles)
            self.modules[normalize_base_key(module)] = {
                "name": module,
                "roles": roles,
                "kind": kind,
                "rank": rank,
                "alpha": self._module_scale(roles, rank),
            }

    def _rank_of(self, roles):
        for role in ("A", "W1", "W1A", "W2A"):
            if role in roles:
                t = self.reader.read(roles[role])
                if t.dim() >= 1:
                    return int(t.shape[0])
        return None

    def _module_scale(self, roles, rank):
        """Returns effective alpha/rank (scale 1.0 when alpha is absent/bogus)."""
        alpha = None
        if "ALPHA" in roles:
            t = self.reader.read(roles["ALPHA"])
            if t.numel() >= 1:
                alpha = float(t.reshape(-1)[0].float().item())
        if alpha is None:
            for mk in ("ss_network_alpha", "network_alpha", "alpha"):
                if mk in self.metadata:
                    try:
                        alpha = float(self.metadata[mk])
                    except (TypeError, ValueError):
                        alpha = None
                    break
        if alpha is None:
            return 1.0
        # Guard against corrupted / overflowed alpha values seen in the wild
        # (e.g. a bf16 scalar decoding to ~1e10): treat those as alpha == rank.
        if not math.isfinite(alpha) or alpha <= 0 or alpha > ALPHA_MAX:
            return 1.0
        return (alpha / rank) if rank else float(alpha)

    # ----------------------------------------------------------- resolving
    @staticmethod
    def _candidates(module):
        """Possible base tensor keys for a LoRA module name (best first)."""
        cands = []

        def add(name):
            for t in (name + ".weight", name):
                if t not in cands:
                    cands.append(t)

        add(normalize_base_key(module))
        stripped = module
        for pfx in BASE_KEY_PREFIXES + KOHYA_PREFIXES:
            if stripped.startswith(pfx):
                stripped = stripped[len(pfx):]
        if "_" in stripped:
            add(stripped.replace("_", "."))
        return cands

    def _match(self, module, base_keys):
        for cand in self._candidates(module):
            if cand in base_keys:
                return cand
        return None

    def resolve(self, base_keys):
        """Maps modules to base keys. Returns (mapping, unmatched names)."""
        mapping, unmatched = {}, []
        for norm, info in self.modules.items():
            key = self._match(info["name"], base_keys)
            if key is None:
                unmatched.append(info["name"])
            else:
                mapping[key] = info
        return mapping, unmatched

    def kind_counts(self):
        counts = {}
        for info in self.modules.values():
            counts[info["kind"]] = counts.get(info["kind"], 0) + 1
        return counts

    def close(self):
        self.reader.close()


class LoraMergeJob:
    """Merges N LoRAs into a base model, streaming tensor by tensor."""

    def __init__(self, base_path, loras, out_path, out_format="fp8",
                 use_gpu=True, math_dtype=MATH_DTYPE,
                 gpu_max_tensor_bytes=GPU_MAX_TENSOR_BYTES,
                 keep_metadata=True, custom_meta_tag=None,
                 log_fn=None, progress_fn=None, cancel_flag=None,
                 inspect_only=False):
        self.base_path = base_path
        self.loras = list(loras)          # [(path, strength), ...]
        self.out_path = out_path
        self.out_format = out_format
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.math_dtype = math_dtype
        self.gpu_max_tensor_bytes = gpu_max_tensor_bytes
        self.keep_metadata = keep_metadata
        self.custom_meta_tag = custom_meta_tag
        self.log = log_fn or (lambda msg: None)
        self.progress = progress_fn or (lambda cur, total, name: None)
        self.cancel_flag = cancel_flag or (lambda: False)
        self.inspect_only = inspect_only

        self.base_reader = None
        self.plan = {}          # base_key -> [(LoraReader, info, strength), ...]
        self.targets = set()    # base keys that actually receive a delta
        self.applied = []       # per-LoRA report rows
        self._open_loras = []
        self.report = {}

    # ------------------------------------------------------------------ utils
    def _pick_device(self, t):
        if not self.use_gpu:
            return torch.device("cpu")
        nbytes = t.numel() * t.element_size()
        return self.device if nbytes <= self.gpu_max_tensor_bytes \
            else torch.device("cpu")

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _has(self, name):
        return name in self.base_reader.infos

    def _target_dtype(self):
        return {"fp8": "F8_E4M3", "fp16": "F16", "bf16": "BF16"}.get(
            self.out_format, "F32")

    def _weight_is_i8(self, name):
        """True if `name` is (or becomes) int8 in the output."""
        if self.out_format not in ("auto", "int8"):
            return False
        info = self.base_reader.infos.get(name)
        return bool(info and info["dtype"] == "I8")

    def _read_base_value(self, key):
        """Reads a base tensor and dequantizes it to math dtype.
        Handles plain floats, fp8 (pure or fp8+scale) and int8+scale."""
        info = self.base_reader.infos[key]
        dtype = info["dtype"]
        scale_key = key + "_scale"
        has_scale = scale_key in self.base_reader.infos
        t = self.base_reader.read(key)
        if dtype == "I8" and has_scale:
            return dequantize_int8(t, self.base_reader.read(scale_key),
                                   self.math_dtype)
        if dtype in FLOAT_DTYPES:
            t = t.to(self.math_dtype)
            if has_scale:
                t = t * self.base_reader.read(scale_key).to(self.math_dtype)
            return t
        return t

    # -------------------------------------------------------------- planning
    def _load_loras(self):
        """Opens every LoRA, resolves its modules against the base key set."""
        base_keys = set(self.base_reader.infos)
        plan = {}
        for path, strength in self.loras:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"LoRA not found: {path}")
            if os.path.getsize(path) == 0:
                raise ValueError(f"LoRA ({path}) is empty (0 bytes).")
            try:
                lora = LoraReader(path)
            except (json.JSONDecodeError, struct.error) as e:
                raise ValueError(
                    f"Corrupted safetensors header in {path}.\nDetails: {e}"
                ) from e
            self._open_loras.append(lora)

            mapping, unmatched = lora.resolve(base_keys)
            if unmatched:
                self.log(f"Warning: {len(unmatched)} module(s) of "
                         f"{os.path.basename(path)} could not be matched to base "
                         f"keys (e.g. {', '.join(unmatched[:3])}).")
            for base_key, info in mapping.items():
                plan.setdefault(base_key, []).append((lora, info, strength))
            self.applied.append({
                "file": os.path.basename(path),
                "strength": strength,
                "modules": len(mapping),
                "unmatched": len(unmatched),
                "kinds": lora.kind_counts(),
            })
        self.plan = plan
        self.targets = set(plan)
        self.report = {
            "base": os.path.basename(self.base_path),
            "base_tensors": len(self.base_reader.infos),
            "base_dtypes": self._dtype_hist(self.base_reader),
            "loras": self.applied,
            "targets": len(self.targets),
        }
        return plan

    @staticmethod
    def _dtype_hist(reader):
        hist = {}
        for info in reader.infos.values():
            hist[info["dtype"]] = hist.get(info["dtype"], 0) + 1
        return hist

    # ------------------------------------------------------------------ merge
    def run(self):
        if not os.path.isfile(self.base_path):
            raise FileNotFoundError(f"Base model not found: {self.base_path}")
        if os.path.getsize(self.base_path) == 0:
            raise ValueError(f"Base model ({self.base_path}) is empty (0 bytes).")
        if not self.loras:
            raise ValueError("No LoRA files were provided.")
        try:
            self.base_reader = TensorReader(self.base_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not open base model: {getattr(e, 'filename', e)}") from e
        except (json.JSONDecodeError, struct.error) as e:
            raise ValueError(
                "Corrupted safetensors header in the base model.\n"
                f"Details: {e}") from e

        try:
            self._load_loras()
            if self.inspect_only:
                return self.report
            tmp_data_path = self.out_path + ".dat"
            try:
                return self._run_merge(tmp_data_path)
            finally:
                if os.path.exists(tmp_data_path):
                    try:
                        os.remove(tmp_data_path)
                    except OSError:
                        pass
        finally:
            for lora in self._open_loras:
                lora.close()
            self.base_reader.close()
            self.base_reader = None

    def _run_merge(self, tmp_data_path):
        names = list(self.base_reader.infos.keys())
        total = len(names)
        out_header = {}
        offset = 0

        merged_meta = {}
        if self.keep_metadata:
            merged_meta.update(self.base_reader.header.get("__metadata__", {}))
        merged_meta["merge_tool"] = "krea2_lora_merge_tool"
        merged_meta["merge_base"] = os.path.basename(self.base_path)
        merged_meta["merge_loras"] = json.dumps(
            [{"file": a["file"], "strength": a["strength"]}
             for a in self.applied])
        merged_meta["merge_format"] = self.out_format
        merged_meta["merge_targets"] = str(len(self.targets))
        if self.custom_meta_tag:
            merged_meta["merge_tag"] = self.custom_meta_tag

        self.log(f"Device: {self.device} | Base tensors: {total} | "
                 f"LoRAs: {len(self.loras)} | Targeted weights: "
                 f"{len(self.targets)} | Format: {self.out_format}")

        with open(tmp_data_path, "wb") as data_f:
            for idx, name in enumerate(names):
                if self.cancel_flag():
                    raise InterruptedError("Cancelled by the user.")
                self.progress(idx, total, name)
                try:
                    offset = self._process_one_tensor(
                        name, data_f, out_header, offset)
                except Exception as e:
                    raise type(e)(f"[{name}] {e}") from e
                self._cleanup()
            self.progress(total, total, "writing final file...")

        self._write_final_file(tmp_data_path, out_header, merged_meta)
        self.log(f"Done! File saved to: {self.out_path}")
        return self.out_path

    def _write_final_file(self, tmp_data_path, out_header, merged_meta):
        """Assembles [len][json header][data] and atomically replaces the output."""
        final_header = dict(out_header)
        final_header["__metadata__"] = merged_meta
        header_bytes = json.dumps(final_header, separators=(",", ":")).encode("utf-8")
        # safetensors reads the header as UTF-8 JSON; keep it aligned to 8 bytes.
        pad = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * pad

        final_path = self.out_path + ".tmp"
        with open(tmp_data_path, "rb") as src, open(final_path, "wb") as dst:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        os.replace(final_path, self.out_path)

    # ----------------------------------------------------------- tensor merge
    def _process_one_tensor(self, name, data_f, out_header, offset):
        """Processes a single tensor (returns the new offset)."""
        # ---- companion scale: emitted together with its weight; skip here.
        if name.endswith("_scale"):
            base = name[:-len("_scale")]
            if self._has(base):
                return offset

        # ---- comfy_quant descriptor: only kept when the blob is int8.
        if name.endswith(".comfy_quant"):
            base_norm = name[:-len(".comfy_quant")] + ".weight"
            if self.out_format in ("auto", "int8") and self._weight_is_i8(base_norm):
                raw = _comfy_quant_bytes()
                data_f.write(raw)
                out_header[name] = {"dtype": "U8", "shape": [len(raw)],
                                    "data_offsets": [offset, offset + len(raw)]}
                offset += len(raw)
            return offset

        if name in self.targets:
            return self._merge_weight(name, data_f, out_header, offset)
        return self._pass_through(name, data_f, out_header, offset)

    def _merge_weight(self, name, data_f, out_header, offset):
        """Dequantizes, adds every LoRA delta, then (re)quantizes the weight."""
        base = self._read_base_value(name)

        # non-float base (e.g. an embedding stored as int): write through unchanged
        if not base.is_floating_point():
            raw = tensor_to_bytes(base)
            data_f.write(raw)
            out_header[name] = {
                "dtype": DTYPES_ST_REV.get(base.dtype, "F32"),
                "shape": list(base.shape),
                "data_offsets": [offset, offset + len(raw)]}
            return offset + len(raw)

        device = self._pick_device(base)
        if device.type == "cuda":
            base = base.to(device)
        for lora, info, strength in self.plan[name]:
            delta = _lora_delta(info, lora.reader, self.math_dtype)
            if delta is None:
                continue
            if tuple(delta.shape) != tuple(base.shape):
                delta = delta.reshape(base.shape)
            if device.type == "cuda":
                delta = delta.to(device)
            base = base + delta * float(strength)
            del delta
        if base.device.type != "cpu":
            base = base.to("cpu")
        base = base.to(torch.float32)

        return self._write_merged(name, base, data_f, out_header, offset)

    def _pass_through(self, name, data_f, out_header, offset):
        """Writes a non-target tensor, recasting float tensors to the output."""
        info = self.base_reader.infos[name]
        dtype = info["dtype"]

        # auto keeps the original bytes; non-float tensors are never recast.
        if self.out_format == "auto" or dtype not in FLOAT_DTYPES:
            raw = self.base_reader.read_raw(name)
            data_f.write(raw)
            out_header[name] = {"dtype": dtype, "shape": info["shape"],
                                "data_offsets": [offset, offset + len(raw)]}
            return offset + len(raw)

        t = cast_float(self.base_reader.read(name).to(self.math_dtype),
                       self._target_dtype())
        raw = tensor_to_bytes(t)
        data_f.write(raw)
        out_header[name] = {"dtype": self._target_dtype(),
                            "shape": list(t.shape),
                            "data_offsets": [offset, offset + len(raw)]}
        return offset + len(raw)

    def _write_merged(self, name, w, data_f, out_header, offset):
        """Serializes a merged fp32 weight in the requested output format."""
        source_i8 = self.base_reader.infos[name]["dtype"] == "I8"
        if self.out_format == "int8" or (self.out_format == "auto" and source_i8):
            q, scale = quantize_int8(w, self.math_dtype)
            raw = tensor_to_bytes(q)
            data_f.write(raw)
            out_header[name] = {"dtype": "I8", "shape": list(q.shape),
                                "data_offsets": [offset, offset + len(raw)]}
            sraw = tensor_to_bytes(scale)
            data_f.write(sraw)
            out_header[name + "_scale"] = {
                "dtype": "F32", "shape": list(scale.shape),
                "data_offsets": [offset + len(raw), offset + len(raw) + len(sraw)]}
            return offset + len(raw) + len(sraw)

        key = self.base_reader.infos[name]["dtype"] if self.out_format == "auto" \
            else self._target_dtype()
        t = cast_float(w, key)
        raw = tensor_to_bytes(t)
        data_f.write(raw)
        out_header[name] = {"dtype": key, "shape": list(t.shape),
                            "data_offsets": [offset, offset + len(raw)]}
        return offset + len(raw)


# ----------------------------------------------------------------------------
# Theme palettes
# ----------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg":        "#12141a",   # window background
        "surface":   "#1a1d26",   # cards
        "surface_2": "#232734",   # inputs / hover
        "border":    "#2e3342",
        "fg":        "#e6e9f0",
        "fg_muted":  "#8b93a7",
        "accent":    "#6c8cff",
        "accent_2":  "#8aa2ff",
        "accent_fg": "#0d0f14",
        "danger":    "#ff6b6b",
        "warn":      "#ffb454",
        "ok":        "#4ade80",
        "log_bg":    "#0e1015",
        "trough":    "#232734",
    },
    "light": {
        "bg":        "#f2f4f8",
        "surface":   "#ffffff",
        "surface_2": "#eef1f6",
        "border":    "#d3d9e3",
        "fg":        "#1b1f27",
        "fg_muted":  "#697086",
        "accent":    "#3b62e8",
        "accent_2":  "#5b7cf0",
        "accent_fg": "#ffffff",
        "danger":    "#d23b3b",
        "warn":      "#b46a00",
        "ok":        "#1a8a4a",
        "log_bg":    "#fbfcfe",
        "trough":    "#e2e7ef",
    },
}


# ----------------------------------------------------------------------------
# Tkinter GUI
# ----------------------------------------------------------------------------
if _HAS_TK:
    class MergeApp(tk.Tk):
        def __init__(self, theme="dark"):
            super().__init__()
            self.title("Safetensors Model Merge  ·  Krea 2")
            self.minsize(1000, 640)

            self.theme_name = theme
            self.C = THEMES[self.theme_name]

            self.msg_queue = queue.Queue()
            self.cancel_event = threading.Event()
            self.worker = None
            self._start_time = None

            self._init_fonts()
            self.style = ttk.Style(self)
            try:
                self.style.theme_use("clam")
            except tk.TclError:
                pass

            self._build_ui()
            self._apply_theme()
            self.update_idletasks()
            self.geometry(f"{max(1000, self.winfo_reqwidth())}x"
                          f"{max(640, self.winfo_reqheight())}")
            self.after(80, self._poll_queue)
            self.after(400, self._tick)

        # --------------------------------------------------------- fonts
        def _init_fonts(self):
            base = tkfont.nametofont("TkDefaultFont").actual()["family"]
            for cand in ("Segoe UI", "Inter", "Ubuntu", "Helvetica Neue", "DejaVu Sans"):
                try:
                    if cand.lower() in (f.lower() for f in tkfont.families()):
                        base = cand
                        break
                except tk.TclError:
                    break
            mono = "Consolas"
            for cand in ("JetBrains Mono", "Cascadia Mono", "Consolas",
                         "DejaVu Sans Mono", "Menlo", "Courier New"):
                try:
                    if cand.lower() in (f.lower() for f in tkfont.families()):
                        mono = cand
                        break
                except tk.TclError:
                    break
            self.f_base = tkfont.Font(family=base, size=10)
            self.f_small = tkfont.Font(family=base, size=9)
            self.f_bold = tkfont.Font(family=base, size=10, weight="bold")
            self.f_title = tkfont.Font(family=base, size=17, weight="bold")
            self.f_sub = tkfont.Font(family=base, size=9)
            self.f_card = tkfont.Font(family=base, size=9, weight="bold")
            self.f_mono = tkfont.Font(family=mono, size=9)
            self.f_pct = tkfont.Font(family=base, size=12, weight="bold")

        # --------------------------------------------------------- theming
        def _apply_theme(self):
            C = self.C = THEMES[self.theme_name]
            st = self.style
            self.configure(bg=C["bg"])

            st.configure(".", background=C["bg"], foreground=C["fg"],
                         font=self.f_base, borderwidth=0, focuscolor=C["accent"])
            st.configure("TFrame", background=C["bg"])
            st.configure("Surface.TFrame", background=C["surface"])
            st.configure("Header.TFrame", background=C["surface"])
            st.configure("Sep.TFrame", background=C["border"])

            st.configure("TLabel", background=C["bg"], foreground=C["fg"])
            st.configure("Surface.TLabel", background=C["surface"], foreground=C["fg"])
            st.configure("Muted.TLabel", background=C["surface"],
                         foreground=C["fg_muted"], font=self.f_small)
            st.configure("MutedBg.TLabel", background=C["bg"],
                         foreground=C["fg_muted"], font=self.f_small)
            st.configure("Title.TLabel", background=C["surface"],
                         foreground=C["fg"], font=self.f_title)
            st.configure("Sub.TLabel", background=C["surface"],
                         foreground=C["fg_muted"], font=self.f_sub)
            st.configure("CardTitle.TLabel", background=C["surface"],
                         foreground=C["accent"], font=self.f_card)
            st.configure("Pct.TLabel", background=C["surface"],
                         foreground=C["accent"], font=self.f_pct)
            st.configure("Ok.TLabel", background=C["surface"], foreground=C["ok"],
                         font=self.f_small)
            st.configure("Warn.TLabel", background=C["surface"], foreground=C["warn"],
                         font=self.f_small)

            # --- buttons
            st.configure("TButton", background=C["surface_2"], foreground=C["fg"],
                         bordercolor=C["border"], relief="flat", padding=(12, 6))
            st.map("TButton",
                   background=[("active", C["border"]), ("disabled", C["surface"])],
                   foreground=[("disabled", C["fg_muted"])])
            st.configure("Accent.TButton", background=C["accent"],
                         foreground=C["accent_fg"], font=self.f_bold, padding=(18, 9))
            st.map("Accent.TButton",
                   background=[("active", C["accent_2"]), ("disabled", C["surface_2"])],
                   foreground=[("disabled", C["fg_muted"])])
            st.configure("Ghost.TButton", background=C["surface"],
                         foreground=C["fg_muted"], padding=(8, 4), font=self.f_small)
            st.map("Ghost.TButton",
                   background=[("active", C["surface_2"])],
                   foreground=[("active", C["fg"])])
            st.configure("Danger.TButton", background=C["surface_2"],
                         foreground=C["danger"], padding=(18, 9))
            st.map("Danger.TButton",
                   background=[("active", C["danger"]), ("disabled", C["surface"])],
                   foreground=[("active", "#ffffff"), ("disabled", C["fg_muted"])])

            # --- entries / combos
            st.configure("TEntry", fieldbackground=C["surface_2"],
                         foreground=C["fg"], insertcolor=C["fg"],
                         selectbackground=C["accent"],
                         selectforeground=C["accent_fg"],
                         bordercolor=C["border"], lightcolor=C["border"],
                         darkcolor=C["border"], padding=6)
            st.map("TEntry", bordercolor=[("focus", C["accent"])],
                   lightcolor=[("focus", C["accent"])])
            st.configure("TCombobox", fieldbackground=C["surface_2"],
                         background=C["surface_2"], foreground=C["fg"],
                         selectbackground=C["accent"],
                         selectforeground=C["accent_fg"],
                         arrowcolor=C["fg_muted"], bordercolor=C["border"],
                         lightcolor=C["border"], darkcolor=C["border"], padding=5)
            st.map("TCombobox",
                   fieldbackground=[("readonly", C["surface_2"])],
                   foreground=[("readonly", C["fg"])],
                   selectbackground=[("readonly", C["accent"])],
                   selectforeground=[("readonly", C["accent_fg"])],
                   bordercolor=[("focus", C["accent"])],
                   arrowcolor=[("active", C["accent"])])
            st.configure("TSpinbox", fieldbackground=C["surface_2"],
                         background=C["surface_2"], foreground=C["fg"],
                         insertcolor=C["fg"],
                         selectbackground=C["accent"],
                         selectforeground=C["accent_fg"],
                         arrowcolor=C["fg_muted"], bordercolor=C["border"],
                         lightcolor=C["border"], darkcolor=C["border"], padding=4)
            st.map("TSpinbox",
                   bordercolor=[("focus", C["accent"])],
                   arrowcolor=[("active", C["accent"])])
            self.option_add("*TCombobox*Listbox.background", C["surface_2"])
            self.option_add("*TCombobox*Listbox.foreground", C["fg"])
            self.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
            self.option_add("*TCombobox*Listbox.selectForeground", C["accent_fg"])

            # --- checkbutton
            st.configure("TCheckbutton", background=C["surface"], foreground=C["fg"],
                         indicatorcolor=C["surface_2"], focuscolor=C["surface"])
            st.map("TCheckbutton",
                   background=[("active", C["surface"])],
                   indicatorcolor=[("selected", C["accent"])],
                   foreground=[("disabled", C["fg_muted"])])

            # --- scale
            st.configure("Horizontal.TScale", background=C["surface"],
                         troughcolor=C["trough"], bordercolor=C["border"],
                         lightcolor=C["accent"], darkcolor=C["accent"])
            st.configure("B.Horizontal.TScale", background=C["surface"],
                         troughcolor=C["trough"], bordercolor=C["border"],
                         lightcolor=C["accent"], darkcolor=C["accent"])

            # --- progressbar
            st.configure("Thin.Horizontal.TProgressbar", background=C["accent"],
                         troughcolor=C["trough"], bordercolor=C["surface"],
                         lightcolor=C["accent"], darkcolor=C["accent"],
                         thickness=10)

            # --- notebook
            st.configure("TNotebook", background=C["surface"], borderwidth=0,
                         tabmargins=(0, 4, 0, 0))
            st.configure("TNotebook.Tab", background=C["surface"],
                         foreground=C["fg_muted"], padding=(14, 7),
                         font=self.f_small, borderwidth=0)
            st.map("TNotebook.Tab",
                   background=[("selected", C["surface_2"])],
                   foreground=[("selected", C["accent"]), ("active", C["fg"])])

            # --- raw tk widgets
            self.txt_log.configure(bg=C["log_bg"], fg=C["fg"],
                                   insertbackground=C["fg"],
                                   selectbackground=C["accent"],
                                   selectforeground=C["accent_fg"],
                                   highlightbackground=C["border"])
            self.txt_log.tag_configure("muted", foreground=C["fg_muted"])
            self.txt_log.tag_configure("warn", foreground=C["warn"])
            self.txt_log.tag_configure("err", foreground=C["danger"])
            self.txt_log.tag_configure("ok", foreground=C["ok"])
            self.txt_log.tag_configure("accent", foreground=C["accent"])
            for canvas in getattr(self, "_cards", []):
                canvas.configure(bg=C["bg"])
                self._redraw_card(canvas)
            self.btn_theme.configure(
                text="☀  Light" if self.theme_name == "dark" else "☾  Dark")

        def _toggle_theme(self):
            self.theme_name = "light" if self.theme_name == "dark" else "dark"
            self._apply_theme()

        # --------------------------------------------------------- card helper
        def _card(self, parent, title, subtitle=None):
            """Rounded 'card' container drawn on a canvas + inner ttk frame."""
            C = self.C
            wrap = tk.Canvas(parent, bg=C["bg"], highlightthickness=0, bd=0)
            inner = ttk.Frame(wrap, style="Surface.TFrame", padding=(16, 12, 16, 14))
            wrap.inner = inner
            win = wrap.create_window(0, 0, anchor="nw", window=inner)
            wrap._win = win

            head = ttk.Frame(inner, style="Surface.TFrame")
            head.pack(fill="x", pady=(0, 8))
            ttk.Label(head, text=title.upper(), style="CardTitle.TLabel").pack(anchor="w")
            if subtitle:
                ttk.Label(head, text=subtitle, style="Sub.TLabel").pack(anchor="w")

            wrap.stretch = False  # when True the card fills its cell instead

            def _resize(_e=None):
                w = wrap.winfo_width()
                wrap.itemconfigure(win, width=max(w - 4, 1))
                if not wrap.stretch:
                    h = max(inner.winfo_reqheight(), 1)
                    wrap.configure(height=h + 4)
                self._redraw_card(wrap)

            wrap.bind("<Configure>", _resize)
            inner.bind("<Configure>", _resize)
            if not hasattr(self, "_cards"):
                self._cards = []
            self._cards.append(wrap)
            return wrap, inner

        def _redraw_card(self, canvas):
            """Draws the rounded rectangle background behind the card content."""
            C = self.C
            canvas.delete("cardbg")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 4 or h < 4:
                return
            r = 12
            x0, y0, x1, y1 = 1, 1, w - 2, h - 2
            fill, out = C["surface"], C["border"]
            canvas.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90,
                              fill=fill, outline=out, tags="cardbg")
            canvas.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90,
                              fill=fill, outline=out, tags="cardbg")
            canvas.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90,
                              fill=fill, outline=out, tags="cardbg")
            canvas.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90,
                              fill=fill, outline=out, tags="cardbg")
            canvas.create_rectangle(x0 + r, y0, x1 - r, y1, fill=fill,
                                    outline=fill, tags="cardbg")
            canvas.create_rectangle(x0, y0 + r, x1, y1 - r, fill=fill,
                                    outline=fill, tags="cardbg")
            canvas.create_line(x0 + r, y0, x1 - r, y0, fill=out, tags="cardbg")
            canvas.create_line(x0 + r, y1, x1 - r, y1, fill=out, tags="cardbg")
            canvas.create_line(x0, y0 + r, x0, y1 - r, fill=out, tags="cardbg")
            canvas.create_line(x1, y0 + r, x1, y1 - r, fill=out, tags="cardbg")
            canvas.tag_lower("cardbg")

        # ------------------------------------------------------------- UI
        def _build_ui(self):
            # ---------------- header
            header = ttk.Frame(self, style="Header.TFrame", padding=(20, 14))
            header.pack(fill="x")
            hl = ttk.Frame(header, style="Header.TFrame")
            hl.pack(side="left")
            ttk.Label(hl, text="Krea 2 · LoRA Merge",
                      style="Title.TLabel").pack(anchor="w")
            ttk.Label(hl, text="Base model + LoRA / LoKr  ·  fp8 / fp16 / bf16 / int8",
                      style="Sub.TLabel").pack(anchor="w")

            hr = ttk.Frame(header, style="Header.TFrame")
            hr.pack(side="right")
            self.lbl_dev = ttk.Label(hr, text=self._device_text(), style="Muted.TLabel")
            self.lbl_dev.pack(side="left", padx=(0, 12))
            self.btn_theme = ttk.Button(hr, text="☀  Light", style="Ghost.TButton",
                                        command=self._toggle_theme)
            self.btn_theme.pack(side="left")

            ttk.Frame(self, style="Sep.TFrame", height=1).pack(fill="x")

            body = ttk.Frame(self, padding=(16, 14))
            body.pack(fill="both", expand=True)
            body.columnconfigure(0, weight=3, uniform="col")
            body.columnconfigure(1, weight=2, uniform="col")
            body.rowconfigure(0, weight=1)

            left = ttk.Frame(body)
            left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            left.columnconfigure(0, weight=1)
            right = ttk.Frame(body)
            right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
            right.columnconfigure(0, weight=1)
            right.rowconfigure(0, weight=1)

            self._build_base_card(left)
            self._build_lora_card(left)
            self._build_output_card(left)
            self._build_options_card(left)
            self._build_log_card(right)
            self._build_footer()

            self._populate_models()

        # ------------------------------------------------------- cards
        def _build_base_card(self, parent):
            card, f = self._card(parent, "Base model",
                                 "The diffusion model the LoRAs are merged into")
            card.grid(row=0, column=0, sticky="ew", pady=(0, 10))

            grid = ttk.Frame(f, style="Surface.TFrame")
            grid.pack(fill="x")
            grid.columnconfigure(1, weight=1)

            self.var_base = tk.StringVar()
            ttk.Label(grid, text="Base", style="Surface.TLabel",
                      width=6).grid(row=0, column=0, sticky="w")
            self.cmb_base = ttk.Combobox(grid, textvariable=self.var_base,
                                         state="readonly")
            self.cmb_base.grid(row=0, column=1, sticky="ew", padx=(6, 6))
            ttk.Button(grid, text="Browse…",
                       command=lambda: self._browse(self.var_base)).grid(
                row=0, column=2)
            self.lbl_info_base = ttk.Label(grid, text="—", style="Muted.TLabel")
            self.lbl_info_base.grid(row=1, column=1, sticky="w", padx=(6, 0),
                                    pady=(1, 4))

            ttk.Button(f, text="⟳  Rescan folder", style="Ghost.TButton",
                       command=self._populate_models).pack(anchor="w", pady=(2, 0))
            self.var_base.trace_add("write", lambda *_: self._refresh_file_info())

        def _build_lora_card(self, parent):
            card, f = self._card(parent, "LoRAs",
                                 "One or more LoRA / LoKr files with per-LoRA strength")
            card.grid(row=1, column=0, sticky="ew", pady=(0, 10))

            self.lora_rows_frame = ttk.Frame(f, style="Surface.TFrame")
            self.lora_rows_frame.pack(fill="x")
            self.lora_rows_frame.columnconfigure(0, weight=1)
            self.lora_rows = []

            btns = ttk.Frame(f, style="Surface.TFrame")
            btns.pack(fill="x", pady=(6, 0))
            ttk.Button(btns, text="＋  Add LoRA", style="Ghost.TButton",
                       command=self._add_lora_row).pack(side="left")
            ttk.Button(btns, text="⇅  Sort", style="Ghost.TButton",
                       command=self._populate_models).pack(side="left", padx=6)

            self._add_lora_row()

        def _build_output_card(self, parent):
            card, f = self._card(parent, "Output")
            card.grid(row=2, column=0, sticky="ew", pady=(0, 10))

            out = ttk.Frame(f, style="Surface.TFrame")
            out.pack(fill="x")
            out.columnconfigure(1, weight=1)
            ttk.Label(out, text="File", style="Surface.TLabel", width=6).grid(
                row=0, column=0, sticky="w")
            self.var_out = tk.StringVar()
            ttk.Entry(out, textvariable=self.var_out).grid(
                row=0, column=1, sticky="ew", padx=(6, 6))
            ttk.Button(out, text="Save as…", command=self._browse_out).grid(
                row=0, column=2)

        # --------------------------------------------------- LoRA list rows
        def _add_lora_row(self, path=""):
            row = ttk.Frame(self.lora_rows_frame, style="Surface.TFrame")
            row.pack(fill="x", pady=2)

            var = tk.StringVar(value=path)
            cmb = ttk.Combobox(row, textvariable=var, state="readonly")
            cmb.pack(side="left", fill="x", expand=True)
            cmb["values"] = self._model_files()

            ttk.Label(row, text="str", style="Muted.TLabel").pack(side="left",
                                                                  padx=(8, 2))
            sv = tk.StringVar(value="0.7")
            ttk.Spinbox(row, from_=0.0, to=4.0, increment=0.05, width=6,
                        textvariable=sv).pack(side="left")
            ttk.Button(row, text="…",
                       command=lambda v=var: self._browse(v)).pack(side="left",
                                                                  padx=(4, 2))
            entry = {"var": var, "strength": sv, "frame": row, "cmb": cmb}
            ttk.Button(row, text="✕", style="Ghost.TButton",
                       command=lambda e=entry: self._remove_lora_row(e)).pack(
                side="left")
            self.lora_rows.append(entry)
            var.trace_add("write", lambda *_: self._refresh_file_info())

        def _remove_lora_row(self, entry):
            if len(self.lora_rows) <= 1:
                entry["var"].set("")
                return
            entry["frame"].destroy()
            self.lora_rows.remove(entry)
            self._refresh_file_info()

        def _lora_entries(self):
            """Returns the configured (path, strength) pairs, ignoring blanks."""
            out = []
            for row in self.lora_rows:
                path = row["var"].get()
                if not path:
                    continue
                try:
                    strength = float(row["strength"].get())
                except (ValueError, tk.TclError):
                    strength = 0.7
                out.append((path, strength))
            return out

        def _build_options_card(self, parent):
            card, f = self._card(parent, "Options")
            card.grid(row=3, column=0, sticky="ew")

            nb = ttk.Notebook(f)
            nb.pack(fill="x")

            tab_basic = ttk.Frame(nb, style="Surface.TFrame", padding=(12, 12))
            nb.add(tab_basic, text="Basic")
            tab_basic.columnconfigure(1, weight=1)

            ttk.Label(tab_basic, text="Output format", style="Surface.TLabel").grid(
                row=0, column=0, sticky="w", pady=3)
            self.var_fmt = tk.StringVar(value="fp8")
            cmb_fmt = ttk.Combobox(tab_basic, textvariable=self.var_fmt,
                                   state="readonly", width=20,
                                   values=["fp8", "bf16", "fp16", "int8", "auto"])
            cmb_fmt.grid(row=0, column=1, sticky="w", padx=8, pady=3)
            self.lbl_fmt_hint = ttk.Label(tab_basic, text="", style="Muted.TLabel",
                                          wraplength=380, justify="left")
            self.lbl_fmt_hint.grid(row=1, column=0, columnspan=2, sticky="w",
                                   pady=(4, 0))
            self.var_fmt.trace_add("write", lambda *_: self._update_fmt_hint())
            self._update_fmt_hint()

            tab_adv = ttk.Frame(nb, style="Surface.TFrame", padding=(12, 12))
            nb.add(tab_adv, text="Advanced")
            tab_adv.columnconfigure(1, weight=1)

            ttk.Label(tab_adv, text="Merge math dtype", style="Surface.TLabel").grid(
                row=0, column=0, sticky="w", pady=3)
            self.var_math = tk.StringVar(value="F32")
            ttk.Combobox(tab_adv, textvariable=self.var_math, state="readonly",
                         width=20, values=["F32", "F16", "BF16"]).grid(
                row=0, column=1, sticky="w", padx=8, pady=3)

            ttk.Label(tab_adv, text="GPU max tensor (MB)", style="Surface.TLabel").grid(
                row=1, column=0, sticky="w", pady=3)
            self.var_gpu_mb = tk.StringVar(
                value=str(GPU_MAX_TENSOR_BYTES // (1024 * 1024)))
            ttk.Entry(tab_adv, textvariable=self.var_gpu_mb, width=22).grid(
                row=1, column=1, sticky="w", padx=8, pady=3)

            self.var_gpu = tk.BooleanVar(value=torch.cuda.is_available())
            self.chk_gpu = ttk.Checkbutton(
                tab_adv,
                text="Use GPU (CUDA)" + ("" if torch.cuda.is_available()
                                         else "  —  not available"),
                variable=self.var_gpu,
                state="normal" if torch.cuda.is_available() else "disabled")
            self.chk_gpu.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 2))

            self.var_keep_meta = tk.BooleanVar(value=True)
            ttk.Checkbutton(tab_adv, text="Keep base metadata",
                            variable=self.var_keep_meta).grid(
                row=3, column=0, columnspan=2, sticky="w", pady=2)

            ttk.Label(tab_adv, text="Custom metadata tag", style="Surface.TLabel").grid(
                row=4, column=0, sticky="w", pady=3)
            self.var_meta_tag = tk.StringVar()
            ttk.Entry(tab_adv, textvariable=self.var_meta_tag, width=22).grid(
                row=4, column=1, sticky="w", padx=8, pady=3)

        def _build_log_card(self, parent):
            card, f = self._card(parent, "Log")
            card.grid(row=0, column=0, sticky="nsew")
            card.stretch = True

            wrap = ttk.Frame(f, style="Surface.TFrame")
            wrap.pack(fill="both", expand=True)
            self.txt_log = tk.Text(wrap, height=22, state="disabled", wrap="none",
                                   font=self.f_mono, bd=0, relief="flat",
                                   padx=10, pady=8, highlightthickness=1)
            sb = ttk.Scrollbar(wrap, command=self.txt_log.yview)
            self.txt_log.configure(yscrollcommand=sb.set)
            self.txt_log.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")

            tools = ttk.Frame(f, style="Surface.TFrame")
            tools.pack(fill="x", pady=(8, 0))
            ttk.Button(tools, text="Clear", style="Ghost.TButton",
                       command=self._clear_log).pack(side="left")
            ttk.Button(tools, text="Copy", style="Ghost.TButton",
                       command=self._copy_log).pack(side="left", padx=6)

            # the log card fills its cell instead of hugging its content
            def _grow(_e=None):
                h = parent.winfo_height()
                if h > 40:
                    card.itemconfigure(card._win, height=h - 4)
                    self._redraw_card(card)
            parent.bind("<Configure>", _grow)

        def _build_footer(self):
            C = self.C
            ttk.Frame(self, style="Sep.TFrame", height=1).pack(fill="x")
            foot = ttk.Frame(self, style="Header.TFrame", padding=(20, 12))
            foot.pack(fill="x")

            bar = ttk.Frame(foot, style="Header.TFrame")
            bar.pack(fill="x")
            bar.columnconfigure(0, weight=1)

            self.progress = ttk.Progressbar(bar, mode="determinate",
                                            style="Thin.Horizontal.TProgressbar")
            self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 16))

            btns = ttk.Frame(bar, style="Header.TFrame")
            btns.grid(row=0, column=1, sticky="e")
            self.btn_start = ttk.Button(btns, text="▶  Start Merge",
                                        style="Accent.TButton", command=self._start)
            self.btn_start.pack(side="left")
            self.btn_cancel = ttk.Button(btns, text="Cancel", style="Danger.TButton",
                                         command=self._cancel, state="disabled")
            self.btn_cancel.pack(side="left", padx=(8, 0))

            info = ttk.Frame(foot, style="Header.TFrame")
            info.pack(fill="x", pady=(6, 0))
            self.lbl_prog = ttk.Label(info, text="Ready.", style="Muted.TLabel")
            self.lbl_prog.pack(side="left")
            self.lbl_eta = ttk.Label(info, text="", style="Muted.TLabel")
            self.lbl_eta.pack(side="right")

        # ------------------------------------------------------------ helpers
        def _device_text(self):
            if torch.cuda.is_available():
                try:
                    return f"● CUDA · {torch.cuda.get_device_name(0)}"
                except Exception:
                    return "● CUDA available"
            return "● CPU only"

        def _update_fmt_hint(self):
            hints = {
                "fp8":  "float8_e4m3fn — half the size of bf16/fp16. Default, "
                        "matches typical Krea 2 checkpoints. fp8's 4-bit mantissa "
                        "can swallow small LoRA deltas; switch to bf16 if the "
                        "effect looks weak.",
                "bf16": "bfloat16 — same size as fp16, keeps the full LoRA effect. "
                        "Best choice when merging small deltas.",
                "fp16": "float16 — high fidelity, safe on most GPUs.",
                "int8": "int8 + per-row weight_scale (ComfyUI layout). Smallest, "
                        "lossy.",
                "auto": "Keeps each tensor's original dtype (requantizes int8).",
            }
            self.lbl_fmt_hint.config(text=hints.get(self.var_fmt.get(), ""))

        def _model_files(self):
            """Lists the .safetensors from the script directory and the cwd."""
            dirs = {
                os.path.normcase(os.path.realpath(
                    os.path.dirname(os.path.abspath(__file__)))),
                os.path.normcase(os.path.realpath(os.getcwd())),
            }
            files = []
            for d in dirs:
                if os.path.isdir(d):
                    files += [os.path.join(d, f) for f in os.listdir(d)
                              if f.lower().endswith(".safetensors")]
            return sorted(set(files))

        def _populate_models(self):
            files = self._model_files()
            self.cmb_base["values"] = files
            for row in self.lora_rows:
                row["cmb"]["values"] = files
            if files and not self.var_base.get():
                self.cmb_base.current(0)
            self._refresh_file_info()

        def _refresh_file_info(self):
            self.lbl_info_base.config(text=self._file_info(self.var_base.get()))
            self._suggest_output()

        @staticmethod
        def _file_info(path):
            if not (path and os.path.isfile(path)):
                return "—"
            try:
                h, _ = read_header(path)
                n = len(tensor_infos(h))
                dts = {v["dtype"] for v in tensor_infos(h).values()}
                return (f"{human_size(os.path.getsize(path))} · {n} tensors · "
                        f"{'/'.join(sorted(dts))}")
            except Exception:
                return f"{human_size(os.path.getsize(path))} · header unreadable"

        def _suggest_output(self):
            if self.var_out.get():
                return
            base = self.var_base.get()
            if not base:
                return
            nbase = os.path.splitext(os.path.basename(base))[0][:26]
            suffix = ""
            loras = self._lora_entries()
            if loras:
                suffix = "__" + os.path.splitext(
                    os.path.basename(loras[0][0]))[0][:20]
                if len(loras) > 1:
                    suffix += f"_x{len(loras)}"
            self.var_out.set(os.path.join(
                os.path.dirname(base),
                f"{nbase}{suffix}_merged.safetensors"))

        def _browse(self, var):
            p = filedialog.askopenfilename(
                title="Select model",
                filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")])
            if p:
                var.set(p)

        def _browse_out(self):
            p = filedialog.asksaveasfilename(
                title="Save merged model", defaultextension=".safetensors",
                filetypes=[("Safetensors", "*.safetensors")])
            if p:
                self.var_out.set(p)

        def _log(self, msg):
            self.msg_queue.put(("log", msg))

        def _log_tag(self, msg):
            low = msg.lower()
            if low.startswith("warning") or "warn" in low[:12]:
                return "warn"
            if "error" in low or "traceback" in low or "cancel" in low:
                return "err"
            if low.startswith("done") or "completed" in low:
                return "ok"
            if low.startswith("starting") or low.startswith("device"):
                return "accent"
            return "muted"

        def _clear_log(self):
            self.txt_log.configure(state="normal")
            self.txt_log.delete("1.0", "end")
            self.txt_log.configure(state="disabled")

        def _copy_log(self):
            self.clipboard_clear()
            self.clipboard_append(self.txt_log.get("1.0", "end-1c"))

        def _tick(self):
            """Updates the elapsed/ETA readout while a job is running."""
            if self._start_time is not None:
                el = time.time() - self._start_time
                cur = float(self.progress["value"])
                mx = float(self.progress["maximum"]) or 1.0
                eta = ""
                if cur > 2:
                    remain = (el / cur) * (mx - cur)
                    eta = f"  ·  ETA {int(remain // 60)}m {int(remain % 60)}s"
                self.lbl_eta.config(
                    text=f"elapsed {int(el // 60)}m {int(el % 60)}s{eta}")
            self.after(400, self._tick)

        def _poll_queue(self):
            try:
                while True:
                    kind, payload = self.msg_queue.get_nowait()
                    if kind == "log":
                        self.txt_log.configure(state="normal")
                        for line in str(payload).rstrip("\n").split("\n"):
                            self.txt_log.insert("end", line + "\n",
                                                self._log_tag(line))
                        self.txt_log.see("end")
                        self.txt_log.configure(state="disabled")
                    elif kind == "progress":
                        cur, total, name = payload
                        self.progress.configure(maximum=total, value=cur)
                        pct = (cur / total * 100) if total else 0
                        short = name if len(name) <= 58 else "…" + name[-57:]
                        self.lbl_prog.config(text=f"{pct:5.1f}%  ·  {cur}/{total}  ·  {short}")
                    elif kind == "done":
                        self._start_time = None
                        self.progress.configure(value=self.progress["maximum"])
                        self.lbl_prog.config(text="Done.")
                        self.btn_start.config(state="normal")
                        self.btn_cancel.config(state="disabled")
                        messagebox.showinfo("Merge", "Merge completed successfully!")
                    elif kind == "error":
                        self._start_time = None
                        self.lbl_prog.config(text="Error.")
                        self.btn_start.config(state="normal")
                        self.btn_cancel.config(state="disabled")
                        messagebox.showerror("Merge error", payload)
            except queue.Empty:
                pass
            self.after(80, self._poll_queue)

        # ------------------------------------------------------------ actions
        def _start(self):
            base_path = self.var_base.get()
            out_path = self.var_out.get()

            if not base_path or not os.path.isfile(base_path):
                messagebox.showerror("Error", "Select the base model.")
                return
            loras = self._lora_entries()
            if not loras:
                messagebox.showerror("Error", "Add at least one LoRA file.")
                return
            for path, _ in loras:
                if not os.path.isfile(path):
                    messagebox.showerror("Error", f"LoRA not found:\n{path}")
                    return
                if os.path.realpath(path) == os.path.realpath(base_path):
                    messagebox.showerror("Error",
                                         "A LoRA cannot be the base model.")
                    return
            if not out_path:
                messagebox.showerror("Error", "Set the output file.")
                return
            if os.path.exists(out_path) and not messagebox.askyesno(
                    "Overwrite", "The output file already exists. Overwrite?"):
                return

            self.cancel_event.clear()
            self.btn_start.config(state="disabled")
            self.btn_cancel.config(state="normal")
            self.progress.configure(value=0)
            self._start_time = time.time()
            desc = ", ".join(f"{os.path.basename(p)}@{s:g}" for p, s in loras)
            self._log(f"Starting merge into {os.path.basename(base_path)}: {desc}")

            job = LoraMergeJob(
                base_path, loras, out_path,
                out_format=self.var_fmt.get(),
                use_gpu=self.var_gpu.get(),
                math_dtype=DTYPES_ST.get(self.var_math.get(), MATH_DTYPE),
                gpu_max_tensor_bytes=self._gpu_max_bytes(),
                keep_metadata=self.var_keep_meta.get(),
                custom_meta_tag=self.var_meta_tag.get().strip() or None,
                log_fn=self._log,
                progress_fn=lambda c, t, n: self.msg_queue.put(("progress", (c, t, n))),
                cancel_flag=self.cancel_event.is_set,
            )
            self.worker = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            self.worker.start()

        def _run_job(self, job):
            try:
                job.run()
                self.msg_queue.put(("done", None))
            except InterruptedError:
                self.msg_queue.put(("log", "Merge cancelled."))
                self.msg_queue.put(("error", "Merge cancelled by the user."))
            except Exception as e:
                tb = traceback.format_exc()
                self.msg_queue.put(("log", tb))
                self.msg_queue.put(("error", f"{type(e).__name__}: {e}"))

        def _cancel(self):
            self.cancel_event.set()
            self._log("Cancelling... (may take a few seconds)")

        def _gpu_max_bytes(self):
            """Parses the GPU max tensor size (MB) field, falling back to the default."""
            try:
                mb = float(self.var_gpu_mb.get())
                if mb > 0:
                    return int(mb * 1024 * 1024)
            except (ValueError, tk.TclError):
                pass
            return GPU_MAX_TENSOR_BYTES


def parse_lora_arg(value):
    """Parses a ``--lora FILE[:STRENGTH]`` argument (drive letters are safe)."""
    head, sep, tail = value.rpartition(":")
    if sep and head:
        try:
            return os.path.abspath(head), float(tail)
        except ValueError:
            pass
    return os.path.abspath(value), 0.7


def _dtype_hist(reader):
    hist = {}
    for info in reader.infos.values():
        hist[info["dtype"]] = hist.get(info["dtype"], 0) + 1
    return hist


def _print_file_summary(path, is_lora=False):
    if not os.path.isfile(path):
        print("  (file not found)")
        return
    try:
        reader = TensorReader(path)
    except Exception as e:
        print(f"  header unreadable: {e}")
        return
    print(f"  size     : {human_size(os.path.getsize(path))}")
    print(f"  tensors  : {len(reader.infos)}")
    print(f"  dtypes   : {_dtype_hist(reader)}")
    meta = reader.header.get("__metadata__", {})
    print(f"  metadata : {list(meta.keys())}")
    print("  sample keys:")
    for k in list(reader.infos)[:6]:
        print(f"    {k}  {reader.infos[k]['dtype']} {reader.infos[k]['shape']}")
    reader.close()

    if is_lora:
        try:
            lora = LoraReader(path)
        except Exception as e:
            print(f"  LoRA parse failed: {e}")
            return
        print(f"  modules  : {len(lora.modules)}  kinds: {lora.kind_counts()}")
        alphas = sorted({round(m["alpha"], 4) for m in lora.modules.values()})
        print(f"  alpha/rank scales: {alphas[:10]}")
        lora.close()


def inspect(base_path, lora_paths):
    """Prints a structural summary and the module match report."""
    if base_path:
        print(f"=== BASE: {os.path.basename(base_path)} ===")
        _print_file_summary(base_path, is_lora=False)
    for p in lora_paths:
        print(f"\n=== LORA: {os.path.basename(p)} ===")
        _print_file_summary(p, is_lora=True)

    if not (base_path and lora_paths):
        return 0

    warnings = []
    job = LoraMergeJob(base_path, [(p, 1.0) for p in lora_paths],
                       out_path=None, inspect_only=True,
                       log_fn=warnings.append)
    try:
        report = job.run()
    except Exception:
        print("\n[ERROR] inspection failed:")
        print(traceback.format_exc())
        return 1

    print("\n--- match report ---")
    for row in report["loras"]:
        print(f"  {row['file']}: {row['modules']} module(s) matched, "
              f"{row['unmatched']} unmatched, kinds={row['kinds']}")
    print(f"  base tensors: {report['base_tensors']} | targeted weights: "
          f"{report['targets']}")
    if report["targets"] == 0:
        print("  WARNING: no LoRA module matched any base key - check formats.")
    for w in warnings:
        print(f"  {w}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA / LoKr files into a base diffusion model")
    parser.add_argument("--gui", action="store_true",
                        help="Force the graphical interface")
    parser.add_argument("--theme", default="dark", choices=["dark", "light"],
                        help="GUI theme (default: dark)")
    parser.add_argument("--model", help="Base model (.safetensors)")
    parser.add_argument("--lora", action="append",
                        help="LoRA file, optionally FILE:STRENGTH (repeatable)")
    parser.add_argument("--out", help="Output file")
    parser.add_argument("--format", default="fp8",
                        choices=["fp8", "bf16", "fp16", "int8", "auto"],
                        help="Output format (default: fp8)")
    parser.add_argument("--math-dtype", default="F32", choices=["F32", "F16", "BF16"],
                        help="Dtype used in merge calculations")
    parser.add_argument("--gpu-max-bytes", type=int, default=GPU_MAX_TENSOR_BYTES,
                        help="Max tensor size (bytes) processed on GPU")
    parser.add_argument("--keep-metadata", action="store_true", default=True,
                        help="Preserve the base model __metadata__")
    parser.add_argument("--no-keep-metadata", dest="keep_metadata",
                        action="store_false",
                        help="Do not preserve the base __metadata__")
    parser.add_argument("--meta-tag", default=None,
                        help="Custom tag added to the output metadata")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--inspect", nargs="*", default=None,
                        help="Inspect files and report LoRA/base matching, then exit")
    args = parser.parse_args()

    loras = [parse_lora_arg(v) for v in (args.lora or [])]

    # ---------------------------------------------------------- inspect mode
    if args.inspect is not None:
        if args.inspect:
            base = args.inspect[0]
            lora_paths = args.inspect[1:]
        elif args.model:
            base = args.model
            lora_paths = [p for p, _ in loras]
        else:
            parser.error("--inspect needs file paths, or --model / --lora.")
        return inspect(base, lora_paths)

    # ------------------------------------------------------------------- GUI
    if args.gui or (not args.model and not loras):
        if not _HAS_TK:
            print("tkinter is not available in this Python. Use CLI mode:")
            print("  python krea-2-lora-merge-tool-v1.py --model BASE.safetensors "
                  "--lora LORA.safetensors:1.0 --out out.safetensors --format fp8")
            return 1
        app = MergeApp(theme=args.theme)
        app.mainloop()
        return 0

    # ------------------------------------------------------------------- CLI
    if not args.model:
        parser.error("--model is required in CLI mode.")
    if not loras:
        parser.error("At least one --lora is required in CLI mode.")
    if not args.out:
        parser.error("--out is required in CLI mode.")

    print(f"Base: {os.path.basename(args.model)}")
    for path, strength in loras:
        print(f"LoRA: {os.path.basename(path)}  (strength {strength:g})")

    job = LoraMergeJob(
        args.model, loras, args.out,
        out_format=args.format,
        use_gpu=not args.cpu,
        math_dtype=DTYPES_ST.get(args.math_dtype, MATH_DTYPE),
        gpu_max_tensor_bytes=args.gpu_max_bytes,
        keep_metadata=args.keep_metadata,
        custom_meta_tag=args.meta_tag,
        log_fn=print,
        progress_fn=lambda c, t, n: print(f"\r[{c}/{t}] {n[:60]}", end=""),
    )
    try:
        job.run()
        print()
    except Exception:
        print("\n[ERROR] Merge failed:")
        print(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
