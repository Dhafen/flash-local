"""Small App Store Connect client. Credentials stay in memory/private files."""
import base64
import json
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ORIGIN = 'https://api.appstoreconnect.apple.com'


def resource_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9-]{1,100}', value):
        raise ValueError('Invalid Apple resource ID')
    return value


def raw_es256(der):
    if len(der) < 8 or len(der) > 72 or der[0] != 0x30 or der[1] != len(der) - 2:
        raise ValueError('Invalid P-256 DER signature')
    offset, result = 2, bytearray()
    for _ in range(2):
        if offset + 2 > len(der) or der[offset] != 2:
            raise ValueError('Missing signature integer')
        length = der[offset + 1]; offset += 2
        value = der[offset:offset + length]; offset += length
        if not 1 <= length <= 33 or len(value) != length or value[0] & 0x80:
            raise ValueError('Invalid positive signature integer')
        if length > 1 and value[0] == 0 and value[1] < 0x80:
            raise ValueError('Noncanonical signature integer')
        number = int.from_bytes(value, 'big')
        if not 0 < number < 1 << 256:
            raise ValueError('Signature integer exceeds P-256')
        result.extend(number.to_bytes(32, 'big'))
    if offset != len(der):
        raise ValueError('Trailing signature data')
    return bytes(result)


def apple_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme, parsed.netloc) != ('https', 'api.appstoreconnect.apple.com') or not parsed.path.startswith('/v1/') or parsed.fragment:
        raise ValueError('Apple API pagination escaped its origin')
    return url


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Unexpected Apple API redirect')


class AppleAPI:
    def __init__(self, key_path, key_id, issuer, openssl='openssl'):
        import uuid
        if not re.fullmatch(r'[A-Z0-9]{10}', key_id):
            raise ValueError('Invalid Apple API key ID')
        uuid.UUID(issuer)
        self.key_path, self.key_id, self.issuer = pathlib.Path(key_path), key_id, issuer
        self.openssl = openssl
        self.opener = urllib.request.build_opener(NoRedirect())

    def token(self):
        def encode(value):
            return base64.urlsafe_b64encode(json.dumps(value, separators=(',', ':')).encode()).rstrip(b'=')
        now = int(time.time())
        message = encode({'alg': 'ES256', 'kid': self.key_id, 'typ': 'JWT'}) + b'.' + encode({
            'iss': self.issuer, 'iat': now - 10, 'exp': now + 600, 'aud': 'appstoreconnect-v1'})
        signed = subprocess.run([self.openssl, 'dgst', '-sha256', '-sign', str(self.key_path)],
                                input=message, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if signed.returncode:
            raise RuntimeError('Apple JWT signing failed; private key diagnostics withheld')
        return (message + b'.' + base64.urlsafe_b64encode(raw_es256(signed.stdout)).rstrip(b'=')).decode()

    def request(self, route, method='GET', body=None, query=None):
        url = route if route.startswith('https://') else ORIGIN + '/v1/' + route
        apple_url(url)
        if query:
            url += ('&' if '?' in url else '?') + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method,
            headers={'Authorization': 'Bearer ' + self.token(), 'Content-Type': 'application/json'})
        for attempt in range(4):
            try:
                with self.opener.open(request, timeout=60) as response:
                    result = response.read((4 << 20) + 1)
                if len(result) > 4 << 20:
                    raise RuntimeError('Apple API response exceeded its bound')
                return json.loads(result) if result else {}
            except urllib.error.HTTPError as error:
                status = error.code
                if method == 'GET' and status in [429, 500, 502, 503, 504] and attempt < 3:
                    error.close(); time.sleep(2 ** attempt); continue
                try:
                    parsed = json.loads(error.read(65536))
                    codes = ','.join(str(item.get('code', 'unknown')) for item in parsed.get('errors', [])
                        if re.fullmatch(r'[A-Z0-9_.-]{1,100}', str(item.get('code', ''))))
                except Exception:
                    codes = 'unavailable'
                finally:
                    error.close()
                raise RuntimeError(f'Apple API HTTP {status}, codes={codes}') from None
            except urllib.error.URLError:
                if method == 'GET' and attempt < 3:
                    time.sleep(2 ** attempt); continue
                raise RuntimeError('Apple API connection unavailable; no mutation retry was attempted') from None

    def all(self, route, query=None):
        values, seen = [], set()
        for page in range(50):
            result = self.request(route, query=query if page == 0 else None)
            values.extend(result['data'])
            route = result.get('links', {}).get('next')
            if not route:
                return values
            apple_url(route)
            if route in seen:
                raise RuntimeError('Apple API pagination repeated a page')
            seen.add(route)
        raise RuntimeError('Apple API pagination exceeded 50 pages')
