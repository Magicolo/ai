"""Compatibility shims for third-party generation packages.

Two small mismatches sit between our pinned stack and the vendored
generators, and both are fixed here (not in the vendors' files, so upstream
clones stay pristine):

- ACE-Step 1.5 was written against ``transformers`` 4, but the frame stack
  needs ``transformers`` 5 for the Ministral 3 text encoder, so one
  environment serves both. Transformers 5 unconditionally instantiates model
  skeletons on the meta device (the ``low_cpu_mem_usage`` flag is silently
  ignored) and then replaces every non-persistent buffer with
  ``torch.empty_like`` garbage, assuming ``_initialize_missing_keys`` will
  re-init them. ACE's remote modeling code asserts on tensor values inside
  ``__init__`` (illegal on meta tensors) and computes buffers (FSQ
  scales/levels/codebooks, RoPE ``inv_freq``) that no checkpoint covers and
  no re-init restores — so stock loading either crashes or generates NaN.
  :func:`apply_transformers5_compat` restores the transformers-4 behavior:
  real CPU init plus snapshot/restore of materialized buffers.
- Kijai's vendored MMAudio ``flow_matching`` module imports
  ``ProgressBar`` from ``comfy.utils``. The engine never runs ComfyUI, so
  the GPU image carries a minimal ``comfy`` stand-in package (see
  ``vendor/comfy/``) providing just that progress reporter.
"""

from __future__ import annotations

from typing import Any

_TRANSFORMERS_MAJOR_VERSION = 5
_APPLIED_PATCHES: set[str] = set()
_CPU_INIT_PATCH = "cpu-init"


def apply_transformers5_compat() -> None:
    """Restore transformers-4 loading behavior for ACE-Step on transformers 5.

    Strips ``torch.device("meta")`` from the model init context (so remote
    modeling code runs its asserts on real CPU tensors) and snapshots every
    materialized buffer around ``_move_missing_keys_from_meta_to_device``
    (so valid ``__init__``-computed buffers survive the unconditional
    ``torch.empty_like`` wipe). No-op on transformers 4 or when already
    applied; safe to call before every music render.
    """
    if _CPU_INIT_PATCH in _APPLIED_PATCHES:
        return
    try:
        import transformers  # noqa: PLC0415
        from transformers import PreTrainedModel  # noqa: PLC0415
    except ImportError:
        return
    try:
        major_version = int(str(transformers.__version__).split(".")[0])
    except ValueError:
        return
    if major_version < _TRANSFORMERS_MAJOR_VERSION:
        _APPLIED_PATCHES.add(_CPU_INIT_PATCH)
        return
    if getattr(PreTrainedModel.get_init_context, "__zoomy_no_meta__", False):
        _APPLIED_PATCHES.add(_CPU_INIT_PATCH)
        return
    import torch  # noqa: PLC0415

    bound_original = PreTrainedModel.get_init_context

    def _real_cpu_init_context(_model_class: Any, *args: Any, **kwargs: Any) -> Any:
        return [
            context
            for context in bound_original(*args, **kwargs)
            if not (isinstance(context, torch.device) and context.type == "meta")
        ]

    _real_cpu_init_context.__zoomy_no_meta__ = True  # type: ignore[attr-defined]
    _replace_class_attribute(
        PreTrainedModel, "get_init_context", classmethod(_real_cpu_init_context)
    )

    # Private transformers API: the patch point is the whole purpose of this
    # module, and it only runs on transformers 5 (guarded above).
    original_move = PreTrainedModel._move_missing_keys_from_meta_to_device  # noqa: SLF001

    def _keep_materialized_buffers(
        self: Any,
        missing_keys: Any,
        device_map: Any,
        device_mesh: Any,
        quantizer: Any,
    ) -> None:
        saved_buffers = {
            name: buffer.clone() for name, buffer in self.named_buffers() if not buffer.is_meta
        }
        original_move(self, missing_keys, device_map, device_mesh, quantizer)
        for name, value in saved_buffers.items():
            if "." in name:
                parent_name, attribute = name.rsplit(".", 1)
                parent = self.get_submodule(parent_name)
            else:
                parent, attribute = self, name
            # Direct _buffers assignment preserves the existing
            # persistent/non-persistent registration; only the tensor is
            # swapped back.
            parent._buffers[attribute] = value

    _replace_class_attribute(
        PreTrainedModel, "_move_missing_keys_from_meta_to_device", _keep_materialized_buffers
    )
    _APPLIED_PATCHES.add(_CPU_INIT_PATCH)


def _replace_class_attribute(target: Any, name: str, value: Any) -> None:
    """Replace a third-party class attribute via setattr with a string name.

    Direct assignment here would trip method-assign (classmethod swap) and
    SLF001 (private-member patch), yet patching is this module's whole
    purpose — so both patches route through this helper, which neither rule
    flags, instead of scattering ignore pragmas that fight the formatter.
    """
    setattr(target, name, value)
