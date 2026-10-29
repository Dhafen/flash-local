import base64
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    spec = importlib.util.spec_from_file_location('eas_signing_gate', ROOT / 'scripts/sign_flint_eas.py')
    signing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signing)
finally:
    sys.path.pop(0)


class EASSigningGateTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('yaml'), 'EAS-only PyYAML not installed')
    def test_fresh_signing_worker_initializes_the_tested_python_runtime(self):
        import yaml
        commands = []
        for filename in ('flint-validation.yml', 'flint-release.yml'):
            job = yaml.safe_load((ROOT / '.eas/build' / filename).read_text())
            commands.append(job['build']['steps'][1]['run']['command'])
        self.assertEqual(commands[0], commands[1])

    def test_private_diagnostics_expose_only_redacted_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'private.log'
            path.write_text('private unrelated detail\nerror: fixture-password signing failed\n')
            summary = signing.private_error_summary(path, ['fixture-password'])
            self.assertEqual(summary, 'error: [withheld] signing failed')
            self.assertNotIn('unrelated', summary)

    def test_private_input_allows_only_app_scoped_manual_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            fixture = {'teamID': 'FIXTURE123', 'build': '246', 'p12': base64.b64encode(b'unit-pfx').decode(),
                       'password': 'unit-fixture-password', 'profile': base64.b64encode(b'unit-profile').decode()}
            path.write_text(json.dumps(fixture))
            self.assertEqual(signing.decode_private_file(path)['p12'], b'unit-pfx')
            fixture['apiPrivateKey'] = 'must-not-enter-EAS'
            path.write_text(json.dumps(fixture))
            with self.assertRaises(ValueError): signing.decode_private_file(path)

    def test_missing_failed_or_changed_gates_cannot_authorize_signing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / '.work/eas-output'; output.mkdir(parents=True)
            evidence = root / 'Experiments/Flint/evidence'; evidence.mkdir(parents=True)
            record = {'status': 'passed', 'stage': 'regression', 'host': {'sourceRevision': 'unit-source'}}
            checks = {'status': 'passed', 'sourceRevision': 'unit-source', 'steps': [{'exitCode': 0} for _ in range(30)]}
            qpack = {'status': 'passed', 'inputFingerprint': 'unit-inputs', 'steps': [{'exitCode': 0} for _ in range(6)]}
            (output / 'validation.json').write_text(json.dumps(record))
            (output / 'regression-checks.json').write_text(json.dumps(checks))
            (evidence / 'apple-qpack-controls.json').write_text(json.dumps(qpack))
            with patch.object(signing.subprocess, 'check_output', return_value='unit-source\n'), patch.object(signing, 'source_fingerprint', return_value='unit-inputs'):
                self.assertEqual(signing.admission(root), 'unit-source')
                checks['steps'][4]['exitCode'] = 65
                (output / 'regression-checks.json').write_text(json.dumps(checks))
                with self.assertRaises(ValueError): signing.admission(root)
                checks['steps'][4]['exitCode'] = 0
                (output / 'regression-checks.json').write_text(json.dumps(checks))
                qpack['inputFingerprint'] = 'another-input-set'
                (evidence / 'apple-qpack-controls.json').write_text(json.dumps(qpack))
                with self.assertRaises(ValueError): signing.admission(root)


if __name__ == '__main__': unittest.main()
