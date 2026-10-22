import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_qpack_validation_log import validate


class QPackValidationLogTests(unittest.TestCase):
    def fixture(self, split=False):
        rows = []
        for index, model in enumerate(('tiny-model', 'tiny-model-q4', 'tiny-model-q35')):
            prefix = 'ORNITH_KV_PLACEMENT '
            if split and index == 1:
                prefix += '✔ Test unrelatedProbe() passed after 0.1 seconds.\n'
            rows.append(prefix + f'model={model} allLogitBitsEqual=true gpuKVBitsEqual=true cpuMirrorEmpty=true syntheticOnly=true')
        return '\n'.join(rows) + '\n✔ Test omittedMirrorPreservesAllLogitBitsAndGPUKV(modelName:) with 3 test cases passed after 1 seconds.\n** TEST SUCCEEDED **\n'

    def testCompleteAndActualInterleavingAreBothAccepted(self):
        validate(self.fixture()); validate(self.fixture(split=True))

    def testMissingDuplicateOrFailedNumericalEvidenceIsRejected(self):
        text = self.fixture()
        for changed in [text.replace('ORNITH_KV_PLACEMENT ', '', 1), text + text,
                        text.replace('allLogitBitsEqual=true', 'allLogitBitsEqual=false', 1),
                        text.replace('gpuKVBitsEqual=true', 'gpuKVBitsEqual=false', 1),
                        text.replace('cpuMirrorEmpty=true', 'cpuMirrorEmpty=false', 1),
                        text.replace('with 3 test cases passed', 'with 2 test cases passed'),
                        text.replace('** TEST SUCCEEDED **', '** TEST FAILED **')]:
            with self.subTest(changed=changed[-150:]):
                with self.assertRaises(ValueError): validate(changed)


if __name__ == '__main__':
    unittest.main()
