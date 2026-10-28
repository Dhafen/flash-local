"""Guard the first opt-in Apple validation; existing release modes are unchanged."""
import json
import os


def validate(inputs):
    if not isinstance(inputs, dict) or type(inputs.get('flint', False)) is not bool:
        raise ValueError('Expected boolean workflow inputs')
    if not inputs.get('flint', False): return {'flintEnabled': False}
    for key in ['ornith_les', 'provisioning_only', 'check_delivery_tools', 'regenerate_project_only',
                'probe_automatic_signing', 'upload_private_testflight']:
        if type(inputs.get(key, False)) is not bool: raise ValueError('Expected boolean workflow input: ' + key)
    if not inputs.get('ornith_les', False): raise ValueError('Flint app validation also requires ornith_les=true')
    if any(inputs.get(key, False) for key in ['provisioning_only', 'check_delivery_tools', 'regenerate_project_only']):
        raise ValueError('Flint requires the complete native, package, app and iOS validation gates')
    if inputs.get('probe_automatic_signing', False) or inputs.get('upload_private_testflight', False):
        raise ValueError('First Flint run is validation-only; signing/upload stays gated until Apple checks and UI review pass')
    return {'flintEnabled': True, 'mode': 'validation_only', 'credentialsNeeded': False}


if __name__ == '__main__':
    print(json.dumps(validate(json.loads(os.environ['FLINT_CI_INPUTS']))))
