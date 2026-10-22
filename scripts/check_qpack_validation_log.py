#!/usr/bin/env python3
"""Validate all three successful KV-placement probes despite XCTest output interleaving.

Swift Testing can insert a progress line between the marker prefix and its
payload. All numerical assertions and xcodebuild's exit status remain required.
"""
import argparse
from pathlib import Path
import re


def validate(text):
    if '** TEST SUCCEEDED **' not in text or '** TEST FAILED **' in text:
        raise ValueError('The native test run did not succeed')
    if text.count('ORNITH_KV_PLACEMENT') != 3:
        raise ValueError('Expected exactly three KV placement markers')
    for model in ('tiny-model', 'tiny-model-q4', 'tiny-model-q35'):
        payload = f'model={model} allLogitBitsEqual=true gpuKVBitsEqual=true cpuMirrorEmpty=true syntheticOnly=true'
        if text.count(payload) != 1:
            raise ValueError(f'Missing or duplicate successful KV placement payload for {model}')
    if not re.search(r'Test omittedMirrorPreservesAllLogitBitsAndGPUKV\(modelName:\) with 3 test cases passed', text):
        raise ValueError('All three parameterized KV placement cases must pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    validate(args.log.read_text(encoding='utf-8'))
    print('QPACK_KV_LOG allThreeCasesPassed=true exactPayloads=true interleavedProgressAccepted=true')


if __name__ == '__main__':
    main()
