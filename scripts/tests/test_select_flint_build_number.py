import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import select_flint_build_number as selector


class FlintBuildNumberTests(unittest.TestCase):
    def test_uses_both_processed_and_inflight_uploads(self):
        builds = [{'attributes': {'version': '247'}}]
        uploads = [{'attributes': {'cfBundleVersion': '250.2'}}]
        self.assertEqual(selector.select_number(builds, uploads, '246'), '251')
        self.assertEqual(selector.select_number(builds, [], '300'), '300')
        self.assertEqual(selector.select_number([], [], '1'), '1')

    def test_invalid_apple_or_ci_values_fail_closed(self):
        for value in [None, '', 'bad', '1.2.3.4', '-1']:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    selector.select_number([{'attributes': {'version': value}}], [], '1')
        for value in ['', '0', '-1', '1\n2']:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): selector.select_number([], [], value)
        with self.assertRaises(ValueError):
            selector.select_number([{'attributes': {'version': '2147483647'}}], [], '1')

    def test_checks_private_target_before_reading_builds(self):
        class API:
            def __init__(self): self.calls = []
            def all(self, route, query): self.calls.append(route); return []
        api = API()
        with patch.object(selector, 'private_target', side_effect=ValueError('not private')):
            with self.assertRaises(ValueError): selector.next_number(api, 'app', 'group', ['tester'], '1')
        self.assertEqual(api.calls, [])
        with patch.object(selector, 'private_target') as private:
            self.assertEqual(selector.next_number(api, 'app', 'group', ['tester'], '4'), '4')
        private.assert_called_once_with(api, 'app', 'group', ['tester'])
        self.assertEqual(api.calls, ['builds', 'apps/app/buildUploads'])


if __name__ == '__main__': unittest.main()
