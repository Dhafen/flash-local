import Foundation

public protocol QPackByteSource: Sendable {
    /// nil range requests one small auxiliary file. Weight requests always
    /// have an exact bounded range and require an exact HTTP 206 response.
    func data(name: String, range: Range<Int>?, totalByteCount: Int) async throws -> Data
}

public struct HTTPQPackByteSource: QPackByteSource {
    let base: URL
    public init(manifest: MLXSourceManifest) throws {
        guard manifest.repository == "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
              manifest.revision.count == 40,
              manifest.revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              let url = URL(string: "https://huggingface.co/\(manifest.repository)/resolve/\(manifest.revision)/") else {
            throw OrnithPerformanceError.invalidSource("source must be immutable and public")
        }
        base = url
    }

    public func data(name: String, range: Range<Int>?, totalByteCount: Int) async throws -> Data {
        guard MLXSourceIndex.safeFilename(name), totalByteCount > 0,
              range.map({ !$0.isEmpty && $0.lowerBound >= 0 && $0.upperBound <= totalByteCount && $0.count <= 8 << 20 }) ?? (totalByteCount <= 32 << 20) else {
            throw OrnithPerformanceError.invalidSource("unbounded download request")
        }
        return try await BoundedHTTPTransfer().fetch(
            url: base.appendingPathComponent(name), range: range, totalByteCount: totalByteCount)
    }
}

/// A data delegate validates the response *before accepting any body*. Using
/// data(for:) first would allow a server ignoring Range to buffer a 5 GB shard.
final class BoundedHTTPTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Data, Error>?
    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var body = Data()
    private var finished = false
    private var cancelled = false
    private var responseAccepted = false
    private var expectedRange: Range<Int>?
    private var expectedTotal = 0
    private var expectedCount = 0
    private let configuration: URLSessionConfiguration

    init(configuration: URLSessionConfiguration = .ephemeral) {
        self.configuration = configuration
    }

    func fetch(url: URL, range: Range<Int>?, totalByteCount: Int) async throws -> Data {
        try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
                request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
                if let range {
                    request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
                }
                lock.lock()
                guard !cancelled else {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                self.continuation = continuation
                expectedRange = range
                expectedTotal = totalByteCount
                expectedCount = range?.count ?? totalByteCount
                let queue = OperationQueue()
                queue.maxConcurrentOperationCount = 1
                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
                let task = session.dataTask(with: request)
                self.session = session
                self.task = task
                lock.unlock()
                task.resume()
            }
        }, onCancel: {
            self.lock.lock()
            self.cancelled = true
            self.lock.unlock()
            self.finish(.failure(CancellationError()))
        })
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse,
              http.value(forHTTPHeaderField: "Content-Encoding").map({ $0 == "identity" }) ?? true,
              http.expectedContentLength < 0 || http.expectedContentLength == expectedCount else {
            completionHandler(.cancel)
            finish(.failure(OrnithPerformanceError.integrity("encoded/oversized/short HTTP response")))
            return
        }
        if let range = expectedRange {
            guard http.statusCode == 206,
                  http.value(forHTTPHeaderField: "Content-Range") == "bytes \(range.lowerBound)-\(range.upperBound - 1)/\(expectedTotal)" else {
                completionHandler(.cancel)
                finish(.failure(OrnithPerformanceError.integrity("server ignored the exact byte range")))
                return
            }
        } else if http.statusCode != 200 {
            completionHandler(.cancel)
            finish(.failure(OrnithPerformanceError.integrity("auxiliary HTTP status")))
            return
        }
        lock.lock()
        responseAccepted = true
        lock.unlock()
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        guard !finished, responseAccepted, data.count <= expectedCount - body.count else {
            lock.unlock()
            finish(.failure(OrnithPerformanceError.integrity("HTTP body exceeds the requested bound")))
            return
        }
        body.append(data)
        lock.unlock()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish(.failure(error)); return }
        lock.lock()
        let complete = responseAccepted && body.count == expectedCount
        let result = body
        lock.unlock()
        finish(complete ? .success(result) : .failure(OrnithPerformanceError.integrity("short HTTP body")))
    }

    private func finish(_ result: Result<Data, Error>) {
        lock.lock()
        guard !finished, let continuation else { lock.unlock(); return }
        finished = true
        self.continuation = nil
        let session = self.session
        self.session = nil
        self.task = nil
        body = Data()
        lock.unlock()
        continuation.resume(with: result)
        session?.invalidateAndCancel()
    }
}
