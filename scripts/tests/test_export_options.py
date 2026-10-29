import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('export_options', ROOT / 'scripts/probe_automatic_export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportOptionsTests(unittest.TestCase):
    def test_default_existing_automatic_signing_behavior_is_preserved(self):
        options = export.export_options('FIXTURE123')
        self.assertEqual(options['signingStyle'], 'automatic')
        self.assertNotIn('provisioningProfiles', options)
        command = export.export_command(Path('archive'), Path('private'), Path('options'),
            {'ASC_KEY_PATH': 'fixture-key', 'ASC_KEY_ID': 'fixture-id', 'ASC_ISSUER_ID': 'fixture-issuer'})
        self.assertIn('-allowProvisioningUpdates', command)
        self.assertIn('-authenticationKeyPath', command)
        self.assertTrue(options['testFlightInternalTestingOnly'])
        self.assertFalse(options['manageAppVersionAndBuildNumber'])

    def test_manual_export_is_app_scoped_and_needs_no_api_private_key(self):
        profile = '12345678-1234-1234-1234-123456789012'
        options = export.export_options('FIXTURE123', profile)
        self.assertEqual(options['provisioningProfiles'], {'com.dhafen.flashlocal': profile})
        self.assertEqual(options['signingStyle'], 'manual')
        self.assertTrue(options['testFlightInternalTestingOnly'])
        command = export.export_command(Path('archive'), Path('private'), Path('options'), {}, manual=True)
        self.assertNotIn('-allowProvisioningUpdates', command)
        self.assertNotIn('-authenticationKeyPath', command)
        with self.assertRaises(ValueError): export.export_options('FIXTURE123', 'invalid')


if __name__ == '__main__': unittest.main()
