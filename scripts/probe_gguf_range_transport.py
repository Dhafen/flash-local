"""Exercise the real URLSession transport early, with eight Ornith header bytes.

The source is compiled unchanged apart from replacing its package import with
the two error cases it uses. This is a live Mac preflight, not a model test.
"""
import pathlib
import subprocess
import tempfile

from fetch_ornith_metadata import MODEL_BYTES, URL


ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = r'''
import Foundation

@main struct Probe {
    static func main() async throws {
        guard CommandLine.arguments.count == 3,
              let url = URL(string: CommandLine.arguments[1]),
              let total = UInt64(CommandLine.arguments[2]) else {
            throw NSError(domain: "GGUF range probe arguments", code: 1)
        }
        let transport = GGUFHTTPRangeTransport()
        defer { transport.close() }
        let first = try await transport.fetch(url: url, range: 0..<4, totalBytes: total, etag: nil)
        let second = try await transport.fetch(url: url, range: 4..<8, totalBytes: total, etag: first.etag)
        guard first.bytes == Data("GGUF".utf8), second.bytes == Data([3, 0, 0, 0]),
              second.etag == first.etag else {
            throw NSError(domain: "GGUF range probe identity", code: 2)
        }
        transport.close()
        var rejected = false
        do { _ = try await transport.fetch(url: url, range: 0..<4, totalBytes: total, etag: first.etag) }
        catch is CancellationError { rejected = true }
        guard rejected else { throw NSError(domain: "GGUF range probe close", code: 3) }
        print("GGUF_URLSESSION_PREFLIGHT source=ornith bytes=8 exact206=true sameETag=true closed=true noTensorPayload=true")
    }
}
'''


def standalone_transport(source):
    expected = 'import GGUFCore\n'
    if source.count(expected) != 1:
        raise ValueError('Unexpected GGUF transport import; refusing a partial probe')
    return source.replace(expected, 'enum GGUFError: Error { case invalid(String), limit(String) }\n', 1)


def main():
    transport = ROOT / 'Sources/MinirunKit/Download/GGUFHTTPRangeTransport.swift'
    with tempfile.TemporaryDirectory(prefix='flash-local-range-probe-') as directory:
        root = pathlib.Path(directory)
        copied = root / 'Transport.swift'
        copied.write_text(standalone_transport(transport.read_text(encoding='utf-8')), encoding='utf-8')
        harness = root / 'Probe.swift'
        harness.write_text(HARNESS, encoding='utf-8')
        binary = root / 'range-probe'
        subprocess.run(['xcrun', 'swiftc', '-parse-as-library', str(copied), str(harness), '-o', str(binary)],
                       check=True, timeout=120)
        subprocess.run([str(binary), URL, str(MODEL_BYTES)], check=True, timeout=120)


if __name__ == '__main__':
    main()
