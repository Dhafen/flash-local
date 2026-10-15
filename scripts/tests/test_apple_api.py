import base64
import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest
import io
import urllib.error
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('apple_api', ROOT / 'scripts/apple_api.py')
api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)


class AppleAPITests(unittest.TestCase):
    def test_transient_reads_retry_but_uncertain_writes_never_repeat(self):
        class Opener:
            def __init__(self): self.calls = 0
            def open(self, request, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise urllib.error.HTTPError(request.full_url, 503, 'unavailable', {},
                        io.BytesIO(b'{"errors":[{"code":"SERVICE_UNAVAILABLE"}]}'))
                return io.BytesIO(b'{"data":[]}')
        client = api.AppleAPI('unused', 'TESTKEY123', '00000000-0000-0000-0000-000000000001')
        client.token = lambda: 'test-token'
        client.opener = Opener()
        with patch.object(api.time, 'sleep'):
            self.assertEqual(client.request('apps'), {'data': []})
        self.assertEqual(client.opener.calls, 2)
        client.opener = Opener()
        with self.assertRaises(RuntimeError): client.request('betaGroups', method='POST', body={'data': {}})
        self.assertEqual(client.opener.calls, 1)

    def test_es256_der_preserves_leading_zero_padding(self):
        r = b'\x00\x80' + b'\x01' * 31
        s = b'\x7f' + b'\x02' * 31
        content = b'\x02' + bytes([len(r)]) + r + b'\x02' + bytes([len(s)]) + s
        self.assertEqual(api.raw_es256(b'\x30' + bytes([len(content)]) + content), r[1:] + s)
        for data in [b'', b'\x30\x06\x02\x01\x80\x02\x01\x01',
                     b'\x30\x07\x02\x02\x00\x01\x02\x01\x01']:
            with self.assertRaises(ValueError): api.raw_es256(data)

    def test_ids_and_pagination_never_forward_credentials_off_origin(self):
        for value in ['../apps', 'abc?x=y', '', 'a/b']:
            with self.assertRaises(ValueError): api.resource_id(value)
        for url in ['http://api.appstoreconnect.apple.com/v1/apps', 'https://example.com/v1/apps',
                    'https://api.appstoreconnect.apple.com.evil.test/v1/apps',
                    'https://api.appstoreconnect.apple.com/v1/apps#fragment']:
            with self.assertRaises(ValueError): api.apple_url(url)
        self.assertEqual(api.resource_id('123-abcd'), '123-abcd')

    def test_generated_jwt_signature_verifies_independently(self):
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec, utils
        except ImportError:
            self.skipTest('Independent signature verification uses cryptography in the local developer environment')
        openssl = shutil.which('openssl')
        if not openssl: self.skipTest('OpenSSL is required')
        key = ec.generate_private_key(ec.SECP256R1())
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'test.p8'
            path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption()))
            client = api.AppleAPI(path, 'TESTKEY123', '00000000-0000-0000-0000-000000000001', openssl)
            token = client.token()
            header, payload, signature = token.split('.')
            raw = base64.urlsafe_b64decode(signature + '==')
            self.assertEqual(len(raw), 64)
            der = utils.encode_dss_signature(int.from_bytes(raw[:32], 'big'), int.from_bytes(raw[32:], 'big'))
            key.public_key().verify(der, (header + '.' + payload).encode(), ec.ECDSA(hashes.SHA256()))
            claims = json.loads(base64.urlsafe_b64decode(payload + '=='))
            self.assertEqual(claims['aud'], 'appstoreconnect-v1')
            self.assertLessEqual(claims['exp'] - claims['iat'], 610)


if __name__ == '__main__': unittest.main()
