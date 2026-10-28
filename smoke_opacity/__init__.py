"""VisPanache APIs with optional segmentation imports deferred until use."""
from importlib import import_module
__version__ = "1.0.0"
_EXPORTS = {
 "load_png_sequence": (".io", "load_png_sequence"),
 "run_sam3_primitive_tracking": (".segmentation", "run_sam3_primitive_tracking"),
 "create_clean_background_from_tracking": (".background", "create_clean_background_from_tracking"),
 "retrieve_opacity_frame": (".opacity", "retrieve_opacity_frame"),
 "retrieve_opacity_dataset": (".opacity", "retrieve_opacity_dataset"),
 "retrieve_hybrid_opacity": (".opacity", "retrieve_hybrid_opacity"),
 "retrieve_hybrid_opacity_qc": (".quality", "retrieve_hybrid_opacity_qc"),
 "to_quality_dataset": (".quality", "to_quality_dataset"),
 "QCConfig": (".quality", "QCConfig"),
}
__all__ = list(_EXPORTS)
def __getattr__(name):
 if name not in _EXPORTS: raise AttributeError(name)
 module, attribute = _EXPORTS[name]
 value = getattr(import_module(module, __name__), attribute)
 globals()[name] = value
 return value
