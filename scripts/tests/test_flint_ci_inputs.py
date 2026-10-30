"""Exercise the actual CI guard entry point; no runner or Apple build is started."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


class FlintCIGuardTests(unittest.TestCase):
    def call(self, inputs):
        return subprocess.run([sys.executable, str(ROOT / 'scripts/check_flint_ci_inputs.py')],
            env={**os.environ, 'FLINT_CI_INPUTS': json.dumps(inputs)}, capture_output=True, text=True)
    def test_opt_out_preserves_existing_delivery_modes(self):
        for values in [{}, {'flint': False, 'upload_private_testflight': True, 'probe_automatic_signing': True}]:
            result = self.call(values)
            self.assertEqual(result.returncode, 0); self.assertEqual(json.loads(result.stdout), {'flintEnabled': False})
    def test_full_unsigned_flint_validation_is_admitted(self):
        result = self.call({'flint': True, 'ornith_les': True, 'release_archive': True})
        self.assertEqual(result.returncode, 0); self.assertEqual(json.loads(result.stdout)['mode'], 'validation_only')
    def test_missing_legacy_owner_skips_or_signing_cannot_bypass_validation(self):
        for values in [{'flint': True}, {'flint': 'true'}] + [
            {'flint': True, 'ornith_les': True, key: True} for key in ['provisioning_only', 'check_delivery_tools',
            'regenerate_project_only', 'probe_automatic_signing', 'upload_private_testflight']]:
            with self.subTest(values=values): self.assertNotEqual(self.call(values).returncode, 0)
        self.assertNotEqual(self.call({'flint': True, 'ornith_les': True, 'release_archive': True,
            'upload_private_testflight': True}).returncode, 0)
    def test_delivery_requires_all_validation_and_signing(self):
        flags = {'flint': True, 'ornith_les': True, 'release_archive': True,
            'probe_automatic_signing': True, 'upload_private_testflight': True}
        result = self.call(flags)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)['mode'], 'private_delivery')
        self.assertTrue(json.loads(result.stdout)['credentialsNeeded'])
        for disabled in ['ornith_les', 'release_archive', 'probe_automatic_signing']:
            values = {**flags, disabled: False}
            with self.subTest(disabled=disabled): self.assertNotEqual(self.call(values).returncode, 0)
        for bypass in ['provisioning_only', 'check_delivery_tools', 'regenerate_project_only']:
            values = {**flags, bypass: True}
            with self.subTest(bypass=bypass): self.assertNotEqual(self.call(values).returncode, 0)
    def test_string_flags_cannot_be_misread_as_false(self):
        for key in ['ornith_les', 'provisioning_only', 'upload_private_testflight']:
            self.assertNotEqual(self.call({'flint': True, 'ornith_les': True, key: 'false'}).returncode, 0)


if __name__ == '__main__': unittest.main()
