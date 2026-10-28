"""Numerical QC tests against the frozen hybrid function, without SAM3 execution."""
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, maximum_filter, median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke_opacity.quality import (QCConfig, QualityFlag as F, radiometric_diagnostics,
                     quality_from_diagnostics, retrieve_hybrid_opacity_qc, to_quality_dataset)


def load_frozen_retrieval():
    path = ROOT / 'smoke_opacity/opacity.py'
    manifest = json.loads((ROOT / 'docs/source_lineage.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['retrieval_sha256']
    from smoke_opacity.opacity import retrieve_hybrid_opacity
    return retrieve_hybrid_opacity


BASELINE = load_frozen_retrieval()


def scene(shape=(12, 18), img=.5, bg=.2, weight=0.):
    image = np.full((*shape, 3), img, dtype=np.float32)
    back = np.full_like(image, bg)
    return image, back, np.ones(shape, bool), np.full(shape, weight, np.float32)


def retrieve(*args, **kwargs):
    with warnings.catch_warnings():
        # Legacy empty-plume/support medians warn; QC must still handle them.
        warnings.simplefilter('ignore', RuntimeWarning)
        return retrieve_hybrid_opacity_qc(*args, retrieval=BASELINE, **kwargs)


class QualityTests(unittest.TestCase):
    def assert_flag(self, result, flag, where=None):
        a = result['quality_flag'] if where is None else result['quality_flag'][where]
        self.assertTrue(np.all((a & int(flag)) != 0), flag.name)

    def test_terrain_preserves_baseline_exactly(self):
        args = scene()
        raw = BASELINE(*args)[0]
        out = retrieve(*args)
        np.testing.assert_array_equal(out['opacity_baseline'], raw)
        np.testing.assert_array_equal(out['opacity'], raw)
        self.assertTrue(np.all(out['retrieval_regime'] == 1))
        self.assert_flag(out, F.SEGMENTATION_UNASSESSED)
        self.assert_flag(out, F.BACKGROUND_UNASSESSED)
        self.assertTrue(np.all(out['retrieval_confidence'] <= .5))

    def test_sky_attenuation(self):
        out = retrieve(*scene(img=.4, bg=.7, weight=1))
        self.assertTrue(np.all(out['retrieval_regime'] == 2))
        self.assertTrue(np.isfinite(out['opacity']).all())

    def test_bright_sky_is_missing_not_transparent(self):
        out = retrieve(*scene(img=.8, bg=.6, weight=1))
        self.assertTrue(np.all(out['retrieval_regime'] == 3))
        self.assertTrue(np.isnan(out['opacity']).all())
        self.assertTrue(np.all(out['opacity_baseline'] == 0))
        self.assertTrue(np.all(out['retrieval_confidence'] == 0))
        self.assert_flag(out, F.AMBIGUOUS_REGIME)

    def test_saturated_measurement_is_invalid(self):
        args = scene(img=.7)
        args[0][..., 0] = 1
        out = retrieve(*args)
        self.assertTrue(out['saturation_mask'].all())
        self.assert_flag(out, F.RGB_CLIPPING)
        self.assertTrue(np.isnan(out['opacity']).all())
        self.assertTrue(np.all(out['retrieval_regime'] == 4))

    def test_background_clipping_also_invalidates(self):
        args = scene(img=.6, bg=.8, weight=1)
        args[1][..., 1] = 1
        out = retrieve(*args)
        self.assertTrue(out['background_saturation_mask'].all())
        self.assertFalse(out['saturation_mask'].any())
        self.assert_flag(out, F.RGB_CLIPPING)
        self.assertTrue(np.isnan(out['opacity']).all())

    def test_near_clipping_has_low_confidence(self):
        out = retrieve(*scene(img=251/255, bg=.5))
        self.assertFalse(out['saturation_mask'].any())
        self.assertTrue(np.isfinite(out['opacity']).all())
        self.assert_flag(out, F.NEAR_CLIPPING)
        self.assertTrue(np.all(out['retrieval_confidence'] <= .25))

    def test_opacity_bound_is_not_sensor_saturation(self):
        out = retrieve(*scene(img=.85, bg=.2), gain_terrain=4.)
        self.assertTrue(np.all(out['opacity_baseline'] == 1))
        self.assert_flag(out, F.OPACITY_AT_BOUND)
        self.assertFalse(out['saturation_mask'].any())
        self.assertTrue(np.all((out['quality_flag'] & int(F.RGB_CLIPPING)) == 0))

    def test_unstable_denominator_remains_missing(self):
        out = retrieve(*scene(img=.97, bg=.94))
        self.assert_flag(out, F.UNSTABLE_DENOMINATOR)
        self.assertTrue(np.isnan(out['opacity']).all())
        self.assertTrue(np.all(out['opacity_baseline'] == 0))

    def test_zero_contrast_not_forced_zero(self):
        out = retrieve(*scene(img=.5, bg=.5))
        self.assertTrue(np.isnan(out['opacity']).all())
        self.assert_flag(out, F.AMBIGUOUS_REGIME)

    def test_terrain_darkening_unsupported(self):
        out = retrieve(*scene(img=.2, bg=.4))
        self.assertTrue(np.all(out['retrieval_regime'] == 4))
        self.assertTrue(np.isnan(out['opacity']).all())

    def test_outside_plume_is_undefined_even_for_clouds(self):
        args = scene(img=1, bg=.5, weight=1)
        args[2][:] = False
        out = retrieve(*args)
        self.assertTrue(np.all(out['retrieval_regime'] == 0))
        self.assertTrue(np.all(out['quality_flag'] == 0))
        self.assertTrue(np.isnan(out['opacity']).all())
        self.assertTrue(np.isnan(out['retrieval_confidence']).all())
        self.assertTrue(out['saturation_mask'].all())  # scene-wide measurement diagnostic

    def test_nonfinite_inputs_and_baseline(self):
        args = scene()
        args[0][0, 0] = np.nan
        out = retrieve(*args)
        self.assert_flag(out, F.NONFINITE_INPUT, (0, 0))
        self.assertTrue(np.isnan(out['opacity'][0, 0]))
        self.assertTrue(np.all(out['retrieval_regime'][1:] == 1))

    def test_unknown_ancillary_pixels_are_flagged(self):
        args = scene()
        confidence = np.ones(args[2].shape)
        confidence[0, 0] = np.nan
        out = retrieve(*args, segmentation_confidence=confidence,
                       background_quality=np.ones(args[2].shape))
        self.assert_flag(out, F.SEGMENTATION_UNASSESSED, (0, 0))
        self.assertEqual(out['retrieval_confidence'][0, 0], .5)
        self.assertEqual(out['retrieval_confidence'][1, 1], 1)

    def test_low_ancillary_quality_invalidates(self):
        args = scene()
        out = retrieve(*args, segmentation_confidence=np.full(args[2].shape, .3),
                       background_quality=np.full(args[2].shape, .2))
        self.assert_flag(out, F.SEGMENTATION_UNCERTAIN)
        self.assert_flag(out, F.BACKGROUND_UNRELIABLE)
        self.assertTrue(np.isnan(out['opacity']).all())

    def test_inpainting_recorded_without_claiming_known_error(self):
        args = scene()
        out = retrieve(*args, background_inpaint_mask=args[2])
        self.assert_flag(out, F.BACKGROUND_INPAINTED)
        self.assertTrue(np.isfinite(out['opacity']).all())
        self.assertTrue(np.all(out['retrieval_confidence'] <= .25))

    def test_transition_support_and_modifications(self):
        args = scene(weight=.5)
        raw, _, _, d = BASELINE(*args, return_diagnostics=True)
        d['seam_gap'][:] = True
        d['use_fallback'][:] = True
        d['sky_to_terrain_scale'][:] = 4
        out = quality_from_diagnostics(*args[:3], raw, d)
        self.assertTrue(np.all(out['retrieval_regime'] == 5))
        for bit in [F.TERRAIN_SKY_TRANSITION, F.INSUFFICIENT_SUPPORT, F.SEAM_REPAIRED,
                    F.SIGN_FALLBACK, F.CALIBRATION_AT_LIMIT]:
            self.assert_flag(out, bit)
        self.assertTrue(np.all(out['retrieval_confidence'] <= .25))

    def test_configurable_contrast_threshold(self):
        args = scene(img=.205, bg=.2)
        self.assertTrue(np.isnan(retrieve(*args)['opacity']).all())
        out = retrieve(*args, config=QCConfig(contrast_tolerance=.001))
        self.assertTrue(np.isfinite(out['opacity']).all())

    def test_uint8_and_float_equivalence(self):
        a = np.full((2, 3, 3), 251, dtype=np.uint8)
        b = np.full_like(a, 128)
        r1 = radiometric_diagnostics(a, b)
        r2 = radiometric_diagnostics(a.astype(np.float32)/255, b.astype(np.float32)/255)
        for k in r1:
            np.testing.assert_array_equal(r1[k], r2[k])

    def test_reject_ambiguous_scales_and_bad_configuration(self):
        for a in [np.full((2, 2, 3), 255.), np.ones((2, 2, 3), np.uint16)]:
            with self.assertRaises(ValueError): radiometric_diagnostics(a, a)
        for kw in [dict(near_clipping_level=1), dict(contrast_tolerance=-1),
                   dict(terrain_max_weight=.8), dict(transition_low_weight=1),
                   dict(radiometric_representation='linear_rgb')]:
            with self.assertRaises(ValueError): QCConfig(**kw)
        args = scene()
        with self.assertRaises(ValueError): retrieve(*args, background_quality=np.ones((1, 1)))
        with self.assertRaises(ValueError): retrieve(*args, segmentation_confidence=np.full(args[2].shape, 2.))
        bad = args[2].astype(float); bad[0, 0] = np.nan
        with self.assertRaises(ValueError): retrieve(args[0], args[1], bad, args[3])

    def test_netcdf_roundtrip_metadata_and_bitmask(self):
        import xarray as xr
        args = scene(img=1)
        out = retrieve(*args)
        provenance = dict(camera_id='synthetic', frame_id='unit_test', source_version='baseline SHA',
                          environment=sys.version, input_paths=['synthetic'])
        ds = to_quality_dataset(out, provenance=provenance)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'qc.nc'; ds.to_netcdf(p, engine='h5netcdf')
            with xr.open_dataset(p, engine="h5netcdf") as saved:
                np.testing.assert_array_equal(saved.quality_flag, out['quality_flag'])
                np.testing.assert_allclose(saved.opacity, out['opacity'], equal_nan=True)
                self.assertEqual(saved.quality_flag.dtype, np.dtype('uint32'))
                self.assertEqual(len(saved.quality_flag.attrs['flag_masks']), 16)
                self.assertEqual(saved.retrieval_regime.attrs['flag_meanings'].split()[3], 'bright_sky_plume_unsupported')
                self.assertIn('not a probability', saved.retrieval_confidence.attrs['comment'])
                self.assertEqual(json.loads(saved.attrs['provenance']), provenance)
        with self.assertRaises(ValueError): to_quality_dataset(out, provenance={})

    def test_baseline_exact_through_spatial_transition(self):
        rng = np.random.default_rng(42)
        shape = (32, 40)
        img = rng.uniform(.3, .85, (*shape, 3)).astype(np.float32)
        bg = np.full_like(img, .5)
        mask = rng.random(shape) > .15
        w = np.broadcast_to(np.linspace(0, 1, shape[1], dtype=np.float32), shape).copy()
        for kwargs in [dict(seam_repair=False), dict(gain_sky=3., scale_clip=(.6, 2.5),
                     support_inner_px=1, support_outer_px=25, min_support_pixels=15,
                     calibration_strength=.8)]:
            expected = BASELINE(img, bg, mask, w, **kwargs)[0]
            result = retrieve(img, bg, mask, w, **kwargs)
            np.testing.assert_array_equal(result['opacity_baseline'], expected)
            np.testing.assert_array_equal(result['opacity'][result['valid_opacity_mask']],
                                          expected[result['valid_opacity_mask']])
            self.assertTrue(np.isnan(result['opacity'][~result['valid_opacity_mask']]).all())
            ds = to_quality_dataset(result, provenance=dict(camera_id='synthetic', frame_id='transition',
                source_version='frozen', environment=sys.version, input_paths=['synthetic']))
            parameters = json.loads(ds.attrs['retrieval_parameters'])
            self.assertEqual(parameters['gain_sky'], kwargs.get('gain_sky', 1.))

    def test_qc_cannot_relax_legacy_denominator_guard(self):
        result = retrieve(*scene(img=.97, bg=.94), config=QCConfig(min_terrain_headroom=.001))
        self.assert_flag(result, F.UNSTABLE_DENOMINATOR)
        self.assertTrue(np.isnan(result['opacity']).all())

    def test_actual_qc_settings_and_ancillary_inputs_are_saved(self):
        args = scene(img=.205, bg=.2)
        config = QCConfig(contrast_tolerance=.001)
        q = np.full(args[2].shape, .8, dtype=np.float32)
        result = retrieve(*args, config=config, background_quality=q)
        provenance = dict(camera_id='synthetic', frame_id='settings', source_version='baseline',
                          environment=sys.version, input_paths=['synthetic'])
        ds = to_quality_dataset(result, provenance=provenance)
        self.assertEqual(json.loads(ds.attrs['qc_config'])['contrast_tolerance'], .001)
        np.testing.assert_array_equal(ds.background_quality, q)
        self.assertTrue(np.isnan(ds.segmentation_confidence).all())
        with self.assertRaises(ValueError):
            to_quality_dataset(result, provenance=provenance, config=QCConfig())

    def test_preserved_numeric_and_other_unchanged_sources(self):
        ledger = json.loads((ROOT/'docs/source_lineage.json').read_text())['files']
        for name, record in ledger.items():
            if record['unchanged']:
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), record['baseline_sha256'], name)
        for name in ['smoke_opacity/opacity.py', 'smoke_opacity/quality.py']:
            self.assertTrue(ledger[name]['unchanged'])



if __name__ == '__main__':
    unittest.main(verbosity=2)
