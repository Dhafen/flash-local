import Foundation
import GGUFCore

public struct GGUFRangeChunk: Sendable {
    public let bytes: Data
    public let etag: String
    public init(bytes: Data, etag: String) { self.bytes = bytes; self.etag = etag }
}

public protocol GGUFRangeTransport: Sendable {
    func fetch(url: URL, range: Range<UInt64>, totalBytes: UInt64, etag: String?) async throws -> GGUFRangeChunk
    func close()
}

/// One bounded response at a time. Header admission happens BEFORE allowing
/// the body, so an origin ignoring Range cannot fill RAM with a 56 GB file.
/// The delegate keeps at most one 8 MiB response and rejects excess bytes.
public final class GGUFHTTPRangeTransport: NSObject, GGUFRangeTransport, URLSessionDataDelegate, @unchecked Sendable {
    public static let maximumChunkBytes = 8 << 20
    private final class Pending {
        let range: Range<UInt64>
        let total: UInt64
        let expectedETag: String?
        let continuation: CheckedContinuation<GGUFRangeChunk, Error>
        let task: URLSessionDataTask
        var data = Data()
        var etag: String?
        var error: Error?
        init(range: Range<UInt64>, total: UInt64, expectedETag: String?,
             continuation: CheckedContinuation<GGUFRangeChunk, Error>, task: URLSessionDataTask) {
            self.range = range; self.total = total; self.expectedETag = expectedETag
            self.continuation = continuation; self.task = task
        }
    }
    private let lock = NSLock()
    private var session: URLSession?
    private var pending: Pending?
    private var stopped = false

    public override init() {
        super.init()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil; configuration.httpCookieStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.httpMaximumConnectionsPerHost = 1
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 300
        let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1; queue.qualityOfService = .utility
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
    }

    public func close() {
        lock.lock(); stopped = true
        let current = session; session = nil
        lock.unlock()
        current?.invalidateAndCancel()
    }

    public func fetch(url: URL, range: Range<UInt64>, totalBytes: UInt64, etag: String?) async throws -> GGUFRangeChunk {
        guard url.scheme == "https", range.lowerBound < range.upperBound, range.upperBound <= totalBytes,
              range.upperBound - range.lowerBound <= UInt64(Self.maximumChunkBytes) else {
            throw GGUFError.invalid("bounded HTTPS model range")
        }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                var request = URLRequest(url: url)
                request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
                request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
                if let etag { request.setValue(etag, forHTTPHeaderField: "If-Range") }
                lock.lock()
                guard !stopped, let session, pending == nil else {
                    lock.unlock(); continuation.resume(throwing: CancellationError()); return
                }
                let task = session.dataTask(with: request)
                pending = Pending(range: range, total: totalBytes, expectedETag: etag, continuation: continuation, task: task)
                lock.unlock(); task.resume()
            }
        } onCancel: { self.close() }
    }

    static func admit(_ response: HTTPURLResponse, range: Range<UInt64>, total: UInt64, etag: String?) throws -> String {
        guard range.lowerBound < range.upperBound, range.upperBound <= total else {
            throw GGUFError.invalid("HTTP response range geometry")
        }
        let expected = "bytes \(range.lowerBound)-\(range.upperBound - 1)/\(total)"
        guard response.statusCode == 206,
              response.value(forHTTPHeaderField: "Content-Range") == expected,
              response.value(forHTTPHeaderField: "Content-Encoding").map({ $0.lowercased() == "identity" }) ?? true,
              let received = response.value(forHTTPHeaderField: "ETag"),
              (2...1024).contains(received.utf8.count), received.hasPrefix("\""), received.hasSuffix("\""),
              etag == nil || etag == received else {
            throw GGUFError.invalid("server changed model identity or did not honor the exact HTTP range")
        }
        if let length = response.value(forHTTPHeaderField: "Content-Length") {
            guard UInt64(length) == range.upperBound - range.lowerBound else {
                throw GGUFError.invalid("HTTP range Content-Length")
            }
        }
        return received
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                           completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        lock.lock(); defer { lock.unlock() }
        guard let pending, pending.task.taskIdentifier == dataTask.taskIdentifier else { completionHandler(.cancel); return }
        do {
            guard let response = response as? HTTPURLResponse else { throw GGUFError.invalid("non-HTTP model response") }
            pending.etag = try Self.admit(response, range: pending.range, total: pending.total, etag: pending.expectedETag)
            pending.data.reserveCapacity(Int(pending.range.upperBound - pending.range.lowerBound))
            completionHandler(.allow)
        } catch { pending.error = error; completionHandler(.cancel) }
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock(); defer { lock.unlock() }
        guard let pending, pending.task.taskIdentifier == dataTask.taskIdentifier, pending.error == nil else { return }
        let expected = Int(pending.range.upperBound - pending.range.lowerBound)
        guard pending.etag != nil, data.count <= expected - pending.data.count else {
            pending.error = GGUFError.limit("HTTP range body exceeded the admitted chunk"); dataTask.cancel(); return
        }
        pending.data.append(data)
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        guard let value = pending, value.task.taskIdentifier == task.taskIdentifier else { lock.unlock(); return }
        pending = nil; lock.unlock()
        if let failure = value.error ?? error { value.continuation.resume(throwing: failure); return }
        guard let etag = value.etag, value.data.count == Int(value.range.upperBound - value.range.lowerBound) else {
            value.continuation.resume(throwing: GGUFError.invalid("truncated HTTP model range")); return
        }
        value.continuation.resume(returning: GGUFRangeChunk(bytes: value.data, etag: etag))
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                           newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        lock.lock(); defer { lock.unlock() }
        guard let pending, pending.task.taskIdentifier == task.taskIdentifier, request.url?.scheme == "https" else {
            completionHandler(nil); return
        }
        var redirected = request
        redirected.setValue("bytes=\(pending.range.lowerBound)-\(pending.range.upperBound - 1)", forHTTPHeaderField: "Range")
        redirected.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        if let etag = pending.expectedETag { redirected.setValue(etag, forHTTPHeaderField: "If-Range") }
        completionHandler(redirected)
    }
}
