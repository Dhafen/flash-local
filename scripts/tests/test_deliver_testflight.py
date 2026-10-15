import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import time
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('delivery', SCRIPTS / 'deliver_testflight.py')
delivery = importlib.util.module_from_spec(spec); spec.loader.exec_module(delivery)


class FakeApple:
    def __init__(self):
        self.bundle = delivery.BUNDLE
        self.group = {'id': 'group-one', 'attributes': {'isInternalGroup': True,
            'hasAccessToAllBuilds': False, 'publicLinkEnabled': False}}
        self.testers = [{'id': 'own-tester'}]
        self.builds = [{'id': 'build-one', 'attributes': {'version': '42', 'processingState': 'VALID',
            'buildAudienceType': 'INTERNAL_ONLY', 'expired': False}}]
        self.linked = []
        self.posts = []

    def all(self, route, query=None):
        if route == 'apps/app-one/betaGroups': return [self.group]
        if route == 'betaGroups/group-one/relationships/betaTesters': return self.testers
        if route == 'builds': return self.builds
        if route == 'builds/build-one/relationships/betaGroups': return self.linked
        raise AssertionError(route)

    def request(self, route, method='GET', body=None):
        if method == 'POST':
            self.posts.append((route, body)); self.linked = [{'id': 'group-one'}]; return {}
        if route == 'apps/app-one': return {'data': {'attributes': {'bundleId': self.bundle}}}
        if route.endswith('/preReleaseVersion'):
            return {'data': {'attributes': {'version': '0.1.0', 'platform': 'IOS'}}}
        if route.endswith('/buildBetaDetail'):
            return {'data': {'attributes': {'internalBuildState': 'IN_BETA_TESTING'}}}
        raise AssertionError(route)


class PrivateDeliveryTests(unittest.TestCase):
    def test_only_own_tester_and_nonpublic_nonautomatic_group_can_receive_build(self):
        api = FakeApple()
        delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        for key in ['publicLinkEnabled', 'hasAccessToAllBuilds']:
            api.group['attributes'][key] = True
            with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
            api.group['attributes'][key] = False
        api.testers.append({'id': 'someone-else'})
        with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        self.assertEqual(api.posts, [], 'Privacy rejection must happen before any access mutation')

    def test_wrong_app_external_group_and_missing_own_tester_refused(self):
        api = FakeApple(); api.bundle = 'com.example.other'
        with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        api.bundle = delivery.BUNDLE; api.group['attributes']['isInternalGroup'] = False
        with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        api.group['attributes']['isInternalGroup'] = True
        with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', [])

    def test_null_public_link_is_inapplicable_only_for_internal_groups_without_a_url(self):
        api = FakeApple()
        api.group['attributes'].update(publicLinkEnabled=None, publicLink=None)
        delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        api.group['attributes']['publicLink'] = 'https://testflight.apple.com/join/unexpected'
        with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        api.group['attributes'].update(publicLink=None, isInternalGroup=False)
        with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        api.group['attributes'].update(isInternalGroup=True, hasAccessToAllBuilds=None)
        with self.assertRaises(ValueError): delivery.private_target(api, 'app-one', 'group-one', ['own-tester'])
        self.assertEqual(api.posts, [])

    def test_processed_internal_build_is_attached_once_and_observed_in_testing(self):
        api = FakeApple()
        build = delivery.wait_valid(api, 'app-one', '42', '0.1.0', time.monotonic() + 1)
        self.assertEqual(delivery.attach_and_confirm(api, build, 'group-one', time.monotonic() + 1), 'IN_BETA_TESTING')
        delivery.attach_and_confirm(api, build, 'group-one', time.monotonic() + 1)
        self.assertEqual(len(api.posts), 1, 'An observed existing association must not be resent')
        api.linked.append({'id': 'unexpected-group'})
        with self.assertRaises(ValueError): delivery.attach_and_confirm(api, build, 'group-one', time.monotonic() + 1)

    def test_expired_failed_wrong_audience_and_ambiguous_builds_refused(self):
        for change in [{'expired': True}, {'processingState': 'FAILED'}, {'buildAudienceType': 'APP_STORE_ELIGIBLE'}]:
            api = FakeApple(); api.builds[0]['attributes'].update(change)
            with self.assertRaises((ValueError, RuntimeError)):
                delivery.wait_valid(api, 'app-one', '42', '0.1.0', time.monotonic() + 1)
        api = FakeApple(); api.builds.append(api.builds[0])
        with self.assertRaises(ValueError): delivery.find_build(api, 'app-one', '42')

    def test_changed_ipa_or_source_cannot_reuse_signature_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            ipa = pathlib.Path(folder) / 'app.ipa'; ipa.write_bytes(b'only a validation-binding fixture')
            path = pathlib.Path(folder) / 'seal.json'
            seal = {'signatureVerified': True, 'memoryEntitlementsVerified': True, 'exportedLocally': True,
                    'privacyManifestVerified': True,
                    'internalTestingOnly': True, 'usesNonExemptEncryption': False,
                    'bundleIdentifier': delivery.BUNDLE, 'profileType': 'app-store',
                    'sourceRevision': 'a' * 40, 'build': '42', 'workflowRunID': '123',
                    'ipaSHA256': hashlib.sha256(ipa.read_bytes()).hexdigest()}
            path.write_text(json.dumps(seal))
            delivery.verified_export(ipa, path, 'a' * 40, '42', '123')
            missing_privacy = dict(seal); missing_privacy.pop('privacyManifestVerified')
            path.write_text(json.dumps(missing_privacy))
            with self.assertRaisesRegex(ValueError, 'privacyManifestVerified'):
                delivery.verified_export(ipa, path, 'a' * 40, '42', '123')
            path.write_text(json.dumps(seal))
            with self.assertRaises(ValueError): delivery.verified_export(ipa, path, 'b' * 40, '42', '123')
            with self.assertRaises(ValueError): delivery.verified_export(ipa, path, 'a' * 40, '43', '123')
            ipa.write_bytes(b'changed after signing check')
            with self.assertRaises(ValueError): delivery.verified_export(ipa, path, 'a' * 40, '42', '123')


if __name__ == '__main__': unittest.main()
