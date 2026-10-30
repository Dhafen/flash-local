import datetime
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deliver_testflight as delivery
import probe_automatic_export as probe


class API:
    def __init__(self):
        self.rows = [{'id': 'own-upload', 'attributes': {'cfBundleShortVersionString': '0.1.0',
            'cfBundleVersion': '249', 'platform': 'IOS', 'state': {'state': 'AWAITING_UPLOAD'},
            'createdDate': '2026-10-30T10:01:16Z'}}]
        self.files = []
    def all(self, route, query=None):
        if route == 'apps/unit-app/buildUploads': return self.rows
        if route == 'buildUploads/own-upload/buildUploadFiles': return self.files
        raise AssertionError(route)


class XcodeReservationTests(unittest.TestCase):
    def setUp(self):
        self.api = API()
        self.start = datetime.datetime(2026, 10, 30, 10, 1, 8, tzinfo=datetime.timezone.utc)
        self.end = datetime.datetime(2026, 10, 30, 10, 1, 36, tzinfo=datetime.timezone.utc)
    def capture(self):
        return probe.capture_export_reservation(self.api, 'unit-app', '0.1.0', '249', self.start, self.end)
    def test_new_empty_reservation_is_bound_and_absent_is_supported(self):
        self.assertEqual(self.capture(), 'own-upload')
        self.api.rows = []
        self.assertIsNone(self.capture())
    def test_ambiguous_stale_wrong_or_nonempty_reservations_refused(self):
        original = self.api.rows[0]
        for attribute, value in [('cfBundleVersion', '248'), ('cfBundleShortVersionString', '2.0'),
                                 ('platform', 'MAC_OS'), ('state', {'state': 'PROCESSING'}),
                                 ('createdDate', '2020-01-01T00:00:00Z'), ('createdDate', '2026-10-30T10:01:16')]:
            old = original['attributes'][attribute]
            original['attributes'][attribute] = value
            with self.subTest(attribute=attribute, value=value):
                with self.assertRaises(ValueError): self.capture()
            original['attributes'][attribute] = old
        self.api.files = [{'id': 'unexpected'}]
        with self.assertRaises(ValueError): self.capture()
        self.api.files = []; self.api.rows.append(original)
        with self.assertRaises(ValueError): self.capture()
    def test_preexisting_build_or_upload_refused_before_export(self):
        with patch.object(delivery, 'find_build', return_value=None):
            with self.assertRaises(ValueError): probe.require_fresh_export_number(self.api, 'unit-app', '0.1.0', '249')
            self.api.rows = []
            probe.require_fresh_export_number(self.api, 'unit-app', '0.1.0', '249')
        with patch.object(delivery, 'find_build', return_value={'id': 'other'}):
            with self.assertRaises(ValueError): probe.require_fresh_export_number(self.api, 'unit-app', '0.1.0', '249')


if __name__ == '__main__': unittest.main()
