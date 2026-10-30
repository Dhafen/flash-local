"""Upload a sealed device IPA through Apple's documented Build Uploads API.

This runs on Windows too. The API key stays on the operator's computer.
https://developer.apple.com/documentation/appstoreconnectapi/build-uploads
No certificate import, new cloud build, Transporter install or store release.
"""
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
import urllib.parse

from apple_api import resource_id
from deliver_testflight import BUNDLE, find_build, verified_export


def upload_operations(values, size):
    """Require an exact partition of this file before sending any bytes."""
    if not isinstance(values, list) or not 1 <= len(values) <= 10000:
        raise ValueError('Missing bounded Apple upload operations')
    result = []
    for value in values:
        offset, length = value.get('offset'), value.get('length')
        if type(offset) is not int or type(length) is not int or offset < 0 or not 0 < length <= size:
            raise ValueError('Invalid Apple upload byte range')
        url = urllib.parse.urlsplit(value['url'])
        if (value.get('method') != 'PUT' or url.scheme != 'https' or not url.hostname
                or not url.hostname.endswith('.apple.com') or url.port not in (None, 443)
                or url.username or url.password or url.fragment):
            raise ValueError('Upload operation is not an HTTPS Apple storage PUT')
        headers = {}
        supplied = value.get('requestHeaders', [])
        if not isinstance(supplied, list) or len(supplied) > 64:
            raise ValueError('Unbounded Apple upload headers')
        for header in supplied:
            name, content = header['name'], header['value']
            if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9-]{1,100}', name)
                    or not isinstance(content, str) or len(content) > 8192
                    or any(c in content for c in '\r\n') or name.lower() in headers
                    or name.lower() in ('host', 'cookie', 'connection', 'transfer-encoding', 'proxy-authorization')):
                raise ValueError('Unexpected Apple upload header')
            headers[name.lower()] = content
        if 'content-length' in headers and headers['content-length'] != str(length):
            raise ValueError('Upload Content-Length conflicts with file range')
        headers['content-length'] = str(length)
        result.append((offset, length, url, headers))
    result.sort(key=lambda item: item[0])
    end = 0
    for offset, length, _, _ in result:
        if offset != end:
            raise ValueError('Apple upload ranges overlap or omit file bytes')
        end += length
    if end != size:
        raise ValueError('Apple upload ranges do not cover the complete file')
    return result


def put_file(path, operations, connection_factory=http.client.HTTPSConnection):
    """Stream one MiB at a time; never attach an App Store API bearer token."""
    operations = upload_operations(operations, path.stat().st_size)
    with path.open('rb') as stream:
        for offset, length, url, headers in operations:
            # Repeating the same storage PUT is idempotent. JSON API mutations
            # are deliberately not retried by AppleAPI.
            for attempt in range(3):
                connection = connection_factory(url.hostname, timeout=120)
                try:
                    stream.seek(offset)
                    target = urllib.parse.urlunsplit(('', '', url.path or '/', url.query, ''))
                    connection.putrequest('PUT', target, skip_accept_encoding=True)
                    for name, value in headers.items():
                        connection.putheader(name, value)
                    connection.endheaders()
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(1 << 20, remaining))
                        if not chunk:
                            raise ValueError('Upload source truncated after validation')
                        connection.send(chunk)
                        remaining -= len(chunk)
                    response = connection.getresponse()
                    status = response.status
                    response.read(65536)
                    if status in (200, 201, 204):
                        break
                    if status not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                        raise RuntimeError('Apple storage PUT failed: HTTP ' + str(status))
                except (OSError, http.client.HTTPException):
                    if attempt == 2:
                        raise RuntimeError('Apple storage connection failed; API state must be checked before retry') from None
                finally:
                    connection.close()
                time.sleep(2 ** attempt)


def digest(path, algorithm='sha256'):
    # Apple's older delivery service reports MD5 for transfer integrity.
    # This is not used instead of the sealed SHA-256 identity check.
    result = hashlib.new(algorithm, usedforsecurity=algorithm.lower() != 'md5')
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def matches_apple_checksum(path, local_sha256, checksum):
    if digest(path) != local_sha256:
        raise ValueError('Local file no longer matches its SHA-256 export seal')
    algorithm = checksum.get('algorithm')
    if algorithm == 'SHA_256':
        expected = local_sha256
    elif algorithm == 'MD5':
        expected = digest(path, 'md5')
    else:
        raise ValueError('Apple did not provide a supported file checksum')
    return checksum.get('hash') == expected


def confirm_build_provenance(api, build_id, report, ipa, deadline=None):
    """Bind an Apple build to this journal's upload and verified IPA bytes.

    A processed build can be visible before Apple populates its upload/build
    relationship. Retry the read only, within a bounded deadline. A conflicting
    identity, malformed relationship or different binary remains a hard stop.
    """
    build_id = resource_id(build_id)
    upload_id = report.get('buildUploadID')
    if not upload_id:
        raise ValueError('Existing Apple build has no matching local upload journal')
    upload_id = resource_id(upload_id)
    deadline = time.monotonic() + 300 if deadline is None else deadline
    announced = False
    while True:
        upload = api.request('buildUploads/' + upload_id, query={'include': 'build'})['data']
        if upload.get('id') != upload_id:
            raise ValueError('Apple build does not belong to the recorded upload')
        relationships = upload.get('relationships', {})
        if not isinstance(relationships, dict) or not isinstance(relationships.get('build', {}), dict):
            raise ValueError('Invalid Apple upload relationship')
        linked_build = relationships.get('build', {}).get('data')
        if linked_build is not None:
            if not isinstance(linked_build, dict) or linked_build.get('type') != 'builds' or linked_build.get('id') != build_id:
                raise ValueError('Apple build does not belong to the recorded upload')
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Apple upload/build relationship is not yet available; do not re-upload blindly')
        if not announced:
            print('Waiting for Apple to link the processed build to its recorded upload.')
            announced = True
        time.sleep(min(10, remaining))
    files = api.all('buildUploads/' + upload_id + '/buildUploadFiles', query={'limit': '200'})
    binary = [row['attributes'] for row in files if row['attributes'].get('assetType') == 'ASSET']
    if len(binary) != 1:
        raise ValueError('Recorded upload does not contain exactly one binary')
    attrs = binary[0]
    checksum = attrs.get('sourceFileChecksums', {}).get('file') or {}
    if ((attrs.get('fileName'), attrs.get('fileSize')) != (ipa.name, ipa.stat().st_size)
            or attrs.get('assetDeliveryState', {}).get('state') not in ('UPLOAD_COMPLETE', 'COMPLETE')
            or not matches_apple_checksum(ipa, report['ipaSHA256'], checksum)):
        raise ValueError('Apple upload binary does not match the sealed export')


def upload_build(api, app_id, ipa, metadata, seal_path, source, number, run_id, journal):
    """Resume only this exact version and sealed file; never create duplicates."""
    resource_id(app_id)
    seal = verified_export(ipa, seal_path, source, number, run_id)
    if metadata is not None:
        if not metadata.is_file() or not 0 < metadata.stat().st_size <= 2 * 1024 * 1024:
            raise ValueError('Supplied Xcode metadata must be a bounded regular file')
        import plistlib
        parsed = plistlib.loads(metadata.read_bytes())
        if not isinstance(parsed, dict):
            raise ValueError('Invalid Xcode export metadata')
    if api.request('apps/' + app_id)['data']['attributes']['bundleId'] != BUNDLE:
        raise ValueError('Wrong App Store app')
    prior = json.loads(journal.read_text(encoding='utf-8')) if journal.is_file() else {}
    if prior and (prior.get('sourceRevision'), prior.get('buildNumber'), prior.get('ipaSHA256')) != (source, number, seal['ipaSHA256']):
        raise ValueError('Existing delivery journal belongs to a different sealed export')
    report = prior | {'sourceRevision': source, 'buildNumber': number, 'version': seal['version'],
              'ipaSHA256': seal['ipaSHA256'], 'binaryUploadCommitted': False,
              'processingVerified': False, 'availableToInternalTesters': False}

    def save():
        journal.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    save()
    existing_build = find_build(api, app_id, number)
    if existing_build is not None:
        confirm_build_provenance(api, existing_build['id'], prior, ipa)
        report['existingBuildObserved'] = True
        save()
        return report
    found = api.all('apps/' + app_id + '/buildUploads', query={
        'filter[cfBundleVersion]': number, 'filter[cfBundleShortVersionString]': seal['version'],
        'filter[platform]': 'IOS', 'limit': '200'})
    if len(found) > 1:
        raise ValueError('Existing build upload is ambiguous; inspect Apple before retry')
    xcode_reservation = False
    if found:
        upload = found[0]
        # A matching number does not establish identity. Only resume a record
        # created by this sealed export's journal, including an uncertain POST,
        # or the new empty reservation independently bound by the Xcode export.
        expected = prior.get('buildUploadID')
        xcode_reservation = (not prior and seal.get('xcodeReservationVerified') is True and
                             seal.get('xcodeReservedUploadID') == upload.get('id'))
        if expected != upload.get('id') and not (expected is None and prior.get('createUploadAttempted') is True) and not xcode_reservation:
            raise ValueError('Existing Apple upload has no matching local delivery journal')
        if upload['attributes'].get('state', {}).get('state') == 'FAILED':
            raise RuntimeError('Existing Apple build upload failed; inspect its errors before retry')
        if xcode_reservation and upload['attributes'].get('state', {}).get('state') != 'AWAITING_UPLOAD':
            raise ValueError('Xcode reservation is no longer awaiting upload')
    else:
        if prior.get('createUploadAttempted'):
            raise RuntimeError('Previous upload creation is uncertain; inspect Apple before another creation attempt')
        report['createUploadAttempted'] = True
        save()  # An uncertain POST must be reconciled with Apple on the next run.
        upload = api.request('buildUploads', method='POST', body={'data': {
            'type': 'buildUploads', 'attributes': {'cfBundleShortVersionString': seal['version'],
                'cfBundleVersion': number, 'platform': 'IOS'},
            'relationships': {'app': {'data': {'type': 'apps', 'id': app_id}}}}})['data']
    identifier = resource_id(upload['id'])
    existing = api.all('buildUploads/' + identifier + '/buildUploadFiles', query={'limit': '200'})
    if xcode_reservation and existing:
        raise ValueError('Xcode reservation acquired unexpected files')
    report['buildUploadID'] = identifier
    if xcode_reservation: report['xcodeReservationAdopted'] = True
    save()
    # Apple's Build Uploads API supports an IPA asset without a description
    # file. AppStoreInfo.plist is mandatory for Windows Transporter, which
    # this direct API path does not use. Apple processing/audience still gates
    # any private tester association in main().
    specifications = [(ipa, 'ASSET', 'com.apple.ipa')]
    if metadata is not None:
        specifications.insert(0, (metadata, 'ASSET_DESCRIPTION',
            'com.apple.binary-property-list' if metadata.read_bytes().startswith(b'bplist') else 'com.apple.xml-property-list'))
    if any(row['attributes'].get('assetType') not in ('ASSET', 'ASSET_DESCRIPTION') for row in existing):
        raise ValueError('Unexpected additional upload assets')
    reserved = []
    for path, kind, uti in specifications:
        rows = [row for row in existing if row['attributes'].get('assetType') == kind]
        if len(rows) > 1:
            raise ValueError('Duplicate Apple file reservations')
        checksum = digest(path)
        if rows:
            row = rows[0]
            attrs = row['attributes']
            if (attrs.get('fileSize'), attrs.get('fileName'), attrs.get('uti')) != (path.stat().st_size, path.name, uti):
                raise ValueError('Existing Apple reservation belongs to a different file')
        else:
            if prior.get('fileReservationAttempts', {}).get(kind):
                raise RuntimeError('Previous file reservation is uncertain; inspect Apple before another reservation')
            report.setdefault('fileReservationAttempts', {})[kind] = True
            report['reservingAsset'] = kind
            save()
            row = api.request('buildUploadFiles', method='POST', body={'data': {
                'type': 'buildUploadFiles', 'attributes': {'fileName': path.name,
                    'fileSize': path.stat().st_size, 'uti': uti, 'assetType': kind},
                'relationships': {'buildUpload': {'data': {'type': 'buildUploads', 'id': identifier}}}}})['data']
        reserved.append((path, kind, row, checksum))
    # Reserve both files before committing either; commit the binary last.
    for path, kind, row, checksum in reserved:
        file_id = resource_id(row['id'])
        attrs = row['attributes']
        state = attrs.get('assetDeliveryState', {}).get('state')
        report['currentAsset'] = kind
        report['currentFileID'] = file_id
        save()
        if state in ('UPLOAD_COMPLETE', 'COMPLETE'):
            if not matches_apple_checksum(path, checksum, attrs.get('sourceFileChecksums', {}).get('file') or {}):
                raise ValueError('Existing committed file checksum differs from this export')
            continue
        if state != 'AWAITING_UPLOAD':
            raise ValueError('Unexpected Apple file upload state')
        put_file(path, attrs.get('uploadOperations'))
        if digest(path) != checksum:
            raise ValueError('Local export changed during upload; refusing commit')
        api.request('buildUploadFiles/' + file_id, method='PATCH', body={'data': {
            'type': 'buildUploadFiles', 'id': file_id, 'attributes': {'uploaded': True,
                # Apple's build upload endpoint rejected SHA_256 for this
                # asset. Its transport checksum uses MD5; the immutable local
                # export remains bound to the verified SHA-256 seal above.
                'sourceFileChecksums': {'file': {'algorithm': 'MD5', 'hash': digest(path, 'md5')}}}}})
    report['binaryUploadCommitted'] = True
    save()
    return report


def main():
    import argparse
    import os
    import shutil
    from apple_api import AppleAPI
    from deliver_testflight import private_target, wait_valid, attach_and_confirm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ipa', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, help='Optional AppStoreInfo.plist, when emitted by Xcode')
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--build', required=True)
    parser.add_argument('--export-run', required=True)
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    api = AppleAPI(Path(os.environ['ASC_KEY_PATH']), os.environ['ASC_KEY_ID'],
        os.environ['ASC_ISSUER_ID'], openssl=shutil.which('openssl') or 'C:/Program Files/Git/usr/bin/openssl.exe')
    app_id = resource_id(os.environ['APP_STORE_APP_ID'])
    group_id = resource_id(os.environ['TESTFLIGHT_INTERNAL_GROUP_ID'])
    allowed = json.loads(os.environ['TESTFLIGHT_ALLOWED_TESTER_IDS'])
    private_target(api, app_id, group_id, allowed)
    report = upload_build(api, app_id, args.ipa, args.metadata, args.seal, args.source,
                          args.build, args.export_run, args.journal)

    def save():
        args.journal.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    deadline = time.monotonic() + 1800
    build = wait_valid(api, app_id, args.build, report['version'], deadline)
    confirm_build_provenance(api, build['id'], report, args.ipa, deadline=deadline)
    report.update(processingVerified=True, buildID=build['id'], buildAudienceType='INTERNAL_ONLY')
    save()
    private_target(api, app_id, group_id, allowed)
    if build['attributes'].get('usesNonExemptEncryption') is None:
        api.request('builds/' + build['id'], method='PATCH', body={'data': {'type': 'builds', 'id': build['id'],
            'attributes': {'usesNonExemptEncryption': False}}})
    state = attach_and_confirm(api, build, group_id, deadline)
    report.update(availableToInternalTesters=True, groupAssociationVerified=True, internalBuildState=state)
    save()
    print('Apple confirms VALID internal-only build available to the existing private group.')


if __name__ == '__main__':
    main()
