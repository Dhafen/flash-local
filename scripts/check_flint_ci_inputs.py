"""Require complete Flint validation before any optional Apple delivery."""
import json
import os


def validate(inputs):
    if not isinstance(inputs, dict) or type(inputs.get('flint', False)) is not bool:
        raise ValueError('Expected boolean workflow inputs')
    if not inputs.get('flint', False): return {'flintEnabled': False}
    for key in ['ornith_les', 'release_archive', 'provisioning_only', 'check_delivery_tools', 'regenerate_project_only',
                'probe_automatic_signing', 'upload_private_testflight']:
        if type(inputs.get(key, False)) is not bool: raise ValueError('Expected boolean workflow input: ' + key)
    if not inputs.get('ornith_les', False): raise ValueError('Flint app validation also requires ornith_les=true')
    if any(inputs.get(key, False) for key in ['provisioning_only', 'check_delivery_tools', 'regenerate_project_only']):
        raise ValueError('Flint requires the complete native, package, app and iOS validation gates')
    signing = inputs.get('probe_automatic_signing', False)
    upload = inputs.get('upload_private_testflight', False)
    if signing and not inputs.get('release_archive', False):
        raise ValueError('Flint signing requires the complete Release archive')
    if upload and not signing:
        raise ValueError('Flint delivery requires a verified signing export')
    mode = 'private_delivery' if upload else ('signing_probe' if signing else 'validation_only')
    return {'flintEnabled': True, 'mode': mode, 'credentialsNeeded': signing}


if __name__ == '__main__':
    print(json.dumps(validate(json.loads(os.environ['FLINT_CI_INPUTS']))))
