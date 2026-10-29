"""Protocol controls with fixture files, not Apple uploads or app inference."""
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import plistlib
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    import upload_apple_build as upload
finally:
    sys.path.pop(0)


def operation(offset, length):
    return {'offset': offset, 'length': length, 'method': 'PUT',
            'url': 'https://unit-storage.apple.com/fixture?signature=synthetic-only', 'requestHeaders': []}


class AppleBuildUploadTests(unittest.TestCase):
    def test_resume_committed_file_accepts_matching_apple_md5_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ipa = root / 'FlashLocal.ipa'; ipa.write_bytes(b'synthetic IPA fixture')
            source = 'a' * 40
            sha = hashlib.sha256(ipa.read_bytes()).hexdigest()
            seal = root / 'seal.json'
            seal.write_text(json.dumps({'bundleIdentifier': upload.BUNDLE, 'profileType': 'app-store',
                'sourceRevision': source, 'build': '246', 'workflowRunID': 'unit-run', 'version': '0.1.0',
                'usesNonExemptEncryption': False, 'signatureVerified': True, 'memoryEntitlementsVerified': True,
                'privacyManifestVerified': True, 'exportedLocally': True, 'internalTestingOnly': True,
                'ipaSHA256': sha}))
            journal = root / 'journal.json'
            journal.write_text(json.dumps({'sourceRevision': source, 'buildNumber': '246', 'ipaSHA256': sha,
                'buildUploadID': 'unit-upload', 'fileReservationAttempts': {'ASSET': True}}))
            api = Mock()
            api.request.return_value = {'data': {'attributes': {'bundleId': upload.BUNDLE}}}
            file_attributes = {'assetType': 'ASSET', 'fileName': ipa.name, 'fileSize': ipa.stat().st_size,
                'uti': 'com.apple.ipa', 'assetDeliveryState': {'state': 'COMPLETE'},
                'sourceFileChecksums': {'file': {'algorithm': 'MD5',
                    'hash': hashlib.md5(ipa.read_bytes(), usedforsecurity=False).hexdigest()}}}
            def all_records(route, query=None):
                if route.endswith('/buildUploadFiles'):
                    return [{'id': 'unit-file', 'attributes': file_attributes}]
                return [{'id': 'unit-upload', 'attributes': {'state': {'state': 'PROCESSING'}}}]
            api.all.side_effect = all_records
            args = (api, 'unit-app', ipa, None, seal, source, '246', 'unit-run', journal)
            with patch.object(upload, 'find_build', return_value=None), patch.object(upload, 'put_file') as put:
                self.assertTrue(upload.upload_build(*args)['binaryUploadCommitted'])
                put.assert_not_called()
                file_attributes['sourceFileChecksums']['file']['hash'] = '0' * 32
                with self.assertRaisesRegex(ValueError, 'checksum differs'):
                    upload.upload_build(*args)
                ipa.write_bytes(b'changed local fixture')
                with self.assertRaisesRegex(ValueError, 'IPA changed'):
                    upload.upload_build(*args)

    def test_existing_build_never_bypasses_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ipa = root / 'FlashLocal.ipa'
            ipa.write_bytes(b'unit fixture')
            api = Mock()
            api.request.return_value = {'data': {'attributes': {'bundleId': upload.BUNDLE}}}
            with patch.object(upload, 'verified_export', return_value={'version': '0.1.0', 'ipaSHA256': 'a' * 64}), \
                 patch.object(upload, 'find_build', return_value={'id': 'foreign-build'}):
                with self.assertRaisesRegex(ValueError, 'no matching local upload journal'):
                    upload.upload_build(api, 'unit-app', ipa, None, root / 'seal.json', 'b' * 40,
                                        '246', 'unit-run', root / 'journal.json')
            self.assertEqual(api.request.call_count, 1)  # Read app identity only; no mutation.
            api.all.assert_not_called()

    def test_existing_build_requires_apple_upload_link_and_exact_binary_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            ipa = Path(tmp) / 'FlashLocal.ipa'
            ipa.write_bytes(b'synthetic fixture, not a signed application')
            expected = hashlib.sha256(ipa.read_bytes()).hexdigest()
            report = {'buildUploadID': 'unit-upload', 'ipaSHA256': expected}

            class FixtureAPI:
                def __init__(self):
                    self.linked_build = 'unit-build'
                    self.checksum = expected
                    self.algorithm = 'SHA_256'
                    self.calls = []

                def request(self, route, method='GET', body=None, query=None):
                    self.calls.append(route)
                    return {'data': {'id': 'unit-upload', 'relationships': {
                        'build': {'data': {'type': 'builds', 'id': self.linked_build}}}}}

                def all(self, route, query=None):
                    self.calls.append(route)
                    return [{'attributes': {'assetType': 'ASSET', 'fileName': ipa.name,
                        'fileSize': ipa.stat().st_size, 'assetDeliveryState': {'state': 'COMPLETE'},
                        'sourceFileChecksums': {'file': {'algorithm': self.algorithm, 'hash': self.checksum}}}}]

            api = FixtureAPI()
            with self.assertRaisesRegex(ValueError, 'no matching local upload journal'):
                upload.confirm_build_provenance(api, 'unit-build', {}, ipa)
            self.assertEqual(api.calls, [])
            api.linked_build = 'different-build'
            with self.assertRaisesRegex(ValueError, 'does not belong'):
                upload.confirm_build_provenance(api, 'unit-build', report, ipa)
            api.linked_build = 'unit-build'
            api.checksum = '0' * 64
            with self.assertRaisesRegex(ValueError, 'does not match'):
                upload.confirm_build_provenance(api, 'unit-build', report, ipa)
            api.checksum = expected
            upload.confirm_build_provenance(api, 'unit-build', report, ipa)
            api.algorithm = 'MD5'
            api.checksum = hashlib.md5(ipa.read_bytes(), usedforsecurity=False).hexdigest()
            upload.confirm_build_provenance(api, 'unit-build', report, ipa)
            api.checksum = '0' * 32
            with self.assertRaisesRegex(ValueError, 'does not match'):
                upload.confirm_build_provenance(api, 'unit-build', report, ipa)
            ipa.write_bytes(b'changed local fixture')
            api.checksum = hashlib.md5(ipa.read_bytes(), usedforsecurity=False).hexdigest()
            with self.assertRaisesRegex(ValueError, 'SHA-256 export seal'):
                upload.confirm_build_provenance(api, 'unit-build', report, ipa)

    def test_rejects_omitted_overlapping_or_external_upload_instructions(self):
        for operations in ([operation(0, 3)], [operation(0, 3), operation(2, 2)],
                           [operation(1, 4)], [operation(0, 5) | {'method': 'POST'}],
                           [operation(0, 5) | {'url': 'https://apple.com.example.org/file'}],
                           [operation(0, 5) | {'url': 'http://unit-storage.apple.com/file'}],
                           [operation(0, 5) | {'requestHeaders': [{'name': 'x-unit', 'value': 'a\r\nb'}]}]):
            with self.assertRaises(ValueError):
                upload.upload_operations(operations, 5)
        self.assertEqual([row[:2] for row in upload.upload_operations(
            [operation(2, 3), operation(0, 2)], 5)], [(0, 2), (2, 3)])

    def test_real_loopback_put_streams_exact_ranges_without_api_credentials(self):
        observed = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_PUT(self):
                observed.append((self.rfile.read(int(self.headers['Content-Length'])),
                                 self.headers.get('Authorization')))
                self.send_response(503 if len(observed) == 1 else 204)
                self.end_headers()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'fixture.ipa'
                data = b'actual-loopback-fixture-' * 100000
                path.write_bytes(data)
                factory = lambda host, timeout: http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=timeout)
                with patch.object(upload.time, 'sleep'):
                    upload.put_file(path, [operation(0, 1700000), operation(1700000, len(data) - 1700000)], factory)
                self.assertEqual([row[0] for row in observed], [data[:1700000], data[:1700000], data[1700000:]])
                self.assertTrue(all(auth is None for _, auth in observed))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_seal_gates_mutations_and_reservations_precede_binary_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ipa = root / 'FlashLocal.ipa'; ipa.write_bytes(b'unit-fixture-ipa-not-a-signed-app')
            metadata = root / 'AppStoreInfo.plist'; metadata.write_bytes(plistlib.dumps({'unitFixture': True}))
            seal = root / 'seal.json'
            source = 'a' * 40
            record = {'bundleIdentifier': upload.BUNDLE, 'profileType': 'app-store', 'sourceRevision': source,
                'build': '246', 'workflowRunID': 'unit-run', 'version': '0.1.0', 'usesNonExemptEncryption': False,
                'signatureVerified': True, 'memoryEntitlementsVerified': True, 'privacyManifestVerified': True,
                'exportedLocally': True, 'internalTestingOnly': True, 'ipaSHA256': hashlib.sha256(ipa.read_bytes()).hexdigest()}
            seal.write_text(json.dumps(record))

            class FixtureAPI:
                def __init__(self):
                    self.calls = []
                    self.rows = []

                def all(self, route, query=None):
                    return self.rows if route.endswith('/buildUploadFiles') else []

                def request(self, route, method='GET', body=None, query=None):
                    self.calls.append((route, method, body))
                    if route == 'apps/unit-app':
                        return {'data': {'attributes': {'bundleId': upload.BUNDLE}}}
                    if route == 'builds':
                        return {'data': []}
                    if route == 'buildUploads':
                        return {'data': {'id': 'unit-upload', 'attributes': {}}}
                    if route == 'buildUploadFiles':
                        attrs = dict(body['data']['attributes'], assetDeliveryState={'state': 'AWAITING_UPLOAD'},
                                     uploadOperations=[operation(0, body['data']['attributes']['fileSize'])])
                        row = {'id': str(len(self.rows) + 1), 'attributes': attrs}
                        self.rows.append(row)
                        return {'data': row}
                    if method == 'PATCH':
                        return {'data': {}}
                    raise AssertionError('Unexpected protocol call')

            api = FixtureAPI()
            arguments = (api, 'unit-app', ipa, metadata, seal, source, '246', 'unit-run', root / 'journal.json')
            with patch.object(upload, 'put_file') as put:
                result = upload.upload_build(*arguments)
                self.assertEqual(put.call_count, 2)
            mutations = [(route, method, body) for route, method, body in api.calls if method != 'GET']
            self.assertEqual([(r, m) for r, m, _ in mutations], [('buildUploads', 'POST'),
                ('buildUploadFiles', 'POST'), ('buildUploadFiles', 'POST'),
                ('buildUploadFiles/1', 'PATCH'), ('buildUploadFiles/2', 'PATCH')])
            self.assertEqual(mutations[-1][2]['data']['attributes']['sourceFileChecksums']['file'],
                             {'algorithm': 'MD5', 'hash': hashlib.md5(ipa.read_bytes(), usedforsecurity=False).hexdigest()})
            self.assertTrue(result['binaryUploadCommitted'])
            self.assertFalse(result['processingVerified'])
            self.assertFalse(result['availableToInternalTesters'])

            ipa_only = FixtureAPI()
            with patch.object(upload, 'put_file') as put:
                result = upload.upload_build(ipa_only, 'unit-app', ipa, None, seal, source,
                    '246', 'unit-run', root / 'ipa-only-journal.json')
                self.assertEqual(put.call_count, 1)
            self.assertEqual([body['data']['attributes']['assetType'] for route, method, body in ipa_only.calls
                              if route == 'buildUploadFiles' and method == 'POST'], ['ASSET'])
            self.assertTrue(result['binaryUploadCommitted'])
            self.assertFalse(result['availableToInternalTesters'])

            class AmbiguousAPI(FixtureAPI):
                def request(self, route, method='GET', body=None, query=None):
                    value = super().request(route, method, body, query)
                    if route == 'buildUploads' and method == 'POST':
                        raise RuntimeError('Fixture lost the response after a potentially accepted POST')
                    return value

            uncertain = AmbiguousAPI()
            uncertain_args = (uncertain, *arguments[1:-1], root / 'uncertain-journal.json')
            with self.assertRaises(RuntimeError):
                upload.upload_build(*uncertain_args)
            with self.assertRaisesRegex(RuntimeError, 'creation is uncertain'):
                upload.upload_build(*uncertain_args)
            self.assertEqual(sum(method == 'POST' for _, method, _ in uncertain.calls), 1)

            ipa.write_bytes(b'changed export')
            api.calls.clear()
            with self.assertRaises(ValueError):
                upload.upload_build(*arguments)
            self.assertEqual(api.calls, [])


if __name__ == '__main__':
    unittest.main()
