"""Read Apple state and select an unused monotonic Flint build number.

This is a read-only preflight. No Apple write, signing or build is performed.
"""
import base64
import json
import os
from pathlib import Path
import re
import tempfile

from apple_api import AppleAPI, resource_id
from deliver_testflight import private_target


def select_number(builds, uploads, run_number):
    if not re.fullmatch(r'[1-9][0-9]*', str(run_number)):
        raise ValueError('Invalid CI run number')
    maximum = 0
    for row, field in [(row, 'version') for row in builds] + [(row, 'cfBundleVersion') for row in uploads]:
        value = row.get('attributes', {}).get(field)
        if not isinstance(value, str) or not re.fullmatch(r'[0-9]+(?:\.[0-9]+){0,2}', value):
            raise ValueError('Unsupported Apple build string; refusing to guess a new number')
        maximum = max(maximum, int(value.split('.')[0]))
    candidate = max(maximum + 1, int(run_number))
    if not 1 <= candidate <= 2_147_483_647:
        raise ValueError('No safe numeric build number available')
    return str(candidate)


def next_number(api, app_id, group_id, allowed, run_number):
    resource_id(app_id)
    private_target(api, app_id, group_id, allowed)
    builds = api.all('builds', query={'filter[app]': app_id, 'limit': '200'})
    uploads = api.all('apps/' + app_id + '/buildUploads', query={'filter[platform]': 'IOS', 'limit': '200'})
    return select_number(builds, uploads, run_number)


def main():
    encoded = os.environ['ASC_PRIVATE_KEY_B64']
    key = base64.b64decode(encoded, validate=True)
    if not 0 < len(key) <= 16384:
        raise ValueError('Invalid signing key length')
    with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP']) as temporary:
        path = Path(temporary) / 'AuthKey.p8'
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(key)
        api = AppleAPI(path, os.environ['ASC_KEY_ID'], os.environ['ASC_ISSUER_ID'])
        number = next_number(api, os.environ['APP_STORE_APP_ID'], os.environ['TESTFLIGHT_INTERNAL_GROUP_ID'],
                             json.loads(os.environ['TESTFLIGHT_ALLOWED_TESTER_IDS']), os.environ['GITHUB_RUN_NUMBER'])
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as stream:
        stream.write('number=' + number + '\n')
    print('Apple private target verified; selected unused build number ' + number)


if __name__ == '__main__':
    main()
