"""Compare only generated tiny Ornith graphs against the existing CPU pin."""
import json
import math
import pathlib
import subprocess
from llama_reference_source import prepare, REVISION


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    work = root / '.work'
    records = []
    with (work/'les-reference.log').open('w') as output:
        def run(command):
            subprocess.run([str(x) for x in command], cwd=root, check=True, stdout=output, stderr=subprocess.STDOUT)
        try:
            source = prepare(root, run)
            build = work/'reference-build'
            run(['cmake', '-S', root/'Tests/ReferenceHarness', '-B', build, f'-DLLAMA_SOURCE={source}',
                '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_ARCHITECTURES=arm64', '-DGGML_NATIVE=OFF',
                '-DGGML_CPU_ARM_ARCH=armv8-a', '-DGGML_ACCELERATE=OFF', '-DGGML_CPU_KLEIDIAI=OFF', '-DGGML_LLAMAFILE=OFF'])
            run(['cmake', '--build', build, '--target', 'flash-local-reference', 'flash-local-projection-probe-control',
                'flash-local-checkpoint-control', 'flash-local-dequantized-control', '--parallel', '3'])
            run([build/'flash-local-projection-probe-control'])
            for mixed in [False, True]:
                suffix = '-mixed' if mixed else ''
                model = root/'.test-fixtures'/f'synthetic-ornith{suffix}.gguf'
                if model.stat().st_size > 32 << 20:
                    raise ValueError('Only bounded generated random fixtures may run')
                run([build/'flash-local-checkpoint-control', model])
                run([build/'flash-local-dequantized-control', model, work/f'les-dequantized-f32{suffix}.gguf'])
                oracle = work/f'les-oracle{suffix}.json'
                run([build/'flash-local-reference', model, oracle])
                expected = json.loads(oracle.read_text())['logits']
                candidate = json.loads((work/f'les-control-{"mixed" if mixed else "f32"}.json').read_text())['cpuLogits']
                if len(expected) != 5 or len(candidate) != 5 or any(len(a) != len(b) for a, b in zip(expected, candidate)):
                    raise ValueError('Incomplete full-vocabulary comparison')
                pairs = [(a, b) for row, other in zip(expected, candidate) for a, b in zip(row, other)]
                if any(not math.isfinite(a) or not math.isfinite(b) for a, b in pairs):
                    raise ValueError('Non-finite reference value')
                maximum = max(abs(a-b) for a, b in pairs)
                relative = math.sqrt(sum((a-b)**2 for a, b in pairs) / max(sum(a*a for a, _ in pairs), 1e-30))
                same = all(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
                           for a, b in zip(expected, candidate))
                if maximum > .01 or relative > .001 or not same:
                    raise ValueError(f'Pinned reference mismatch: maxAbs={maximum} relativeL2={relative} sameGreedy={same}')
                records.append(dict(mixed=mixed, positions=5, maximumAbsoluteError=maximum, relativeL2=relative, sameGreedy=same))
        except Exception:
            output.flush(); print((work/'les-reference.log').read_text(errors='replace')[-10000:]); raise
    (work/'les-reference-check.json').write_text(json.dumps(dict(referenceRevision=REVISION,
        syntheticOnly=True, trainedModelNumericsVerified=False, comparisons=records), indent=2)+'\n')
    print('ORNITH_LES_REFERENCE fullGraph=true positions=10 mixedQ4KQ6K=true sameGreedy=true syntheticOnly=true')


if __name__ == '__main__':
    main()
