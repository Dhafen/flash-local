import Foundation
import XCTest
@testable import OrnithPerformance

final class BoundedHTTPTests: XCTestCase {
    private func makeFixture(_ handler: @escaping StubProtocol.Handler) -> (transfer: BoundedHTTPTransfer, url: URL) {
        // A cancelled URLSession can enter startLoading after its caller has
        // finished. Never let that old request use a later test's handler.
        let url = URL(string: "https://fixture.invalid/\(UUID().uuidString)")!
        StubProtocol.configure(handler, for: url)
        addTeardownBlock { StubProtocol.removeHandler(for: url) }
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        return (BoundedHTTPTransfer(configuration: config), url)
    }

    func testExact206RangeIsAcceptedAndBodyRemainsBounded() async throws {
        let fixture = makeFixture { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=2-5")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Accept-Encoding"), "identity")
            return (206, ["Content-Range": "bytes 2-5/8", "Content-Length": "4"], Data([2, 3, 4, 5]))
        }
        let data = try await fixture.transfer.fetch(url: fixture.url, range: 2..<6, totalByteCount: 8)
        XCTAssertEqual(data, Data([2, 3, 4, 5]))
    }

    func testIgnoredRangeWrongRangeEncodingAndOversizedBodiesAreRejected() async throws {
        let cases: [(Int, [String: String], Data)] = [
            (200, ["Content-Length": "4"], Data(count: 4)),
            (206, ["Content-Range": "bytes 0-3/8", "Content-Length": "4"], Data(count: 4)),
            (206, ["Content-Range": "bytes 2-5/8", "Content-Encoding": "gzip"], Data(count: 4)),
            (206, ["Content-Range": "bytes 2-5/8", "Content-Length": "8"], Data(count: 8)),
            (206, ["Content-Range": "bytes 2-5/8"], Data(count: 8)),
            (206, ["Content-Range": "bytes 2-5/8"], Data(count: 3)),
        ]
        for (status, headers, body) in cases {
            let fixture = makeFixture { _ in (status, headers, body) }
            do {
                _ = try await fixture.transfer.fetch(url: fixture.url, range: 2..<6, totalByteCount: 8)
                XCTFail("invalid range/length response must fail")
            } catch { }
        }
    }

    func testCancellationCompletesEvenWhenServerNeverSendsAResponse() async throws {
        let fixture = makeFixture { _ in nil }
        let task = Task { try await fixture.transfer.fetch(url: fixture.url, range: 0..<4, totalByteCount: 8) }
        try await Task.sleep(nanoseconds: 20_000_000)
        task.cancel()
        do { _ = try await task.value; XCTFail("cancellation expected") }
        catch is CancellationError { }
    }

    func testOverlappingTransfersKeepTheirOwnRangeHandlers() async throws {
        let first = makeFixture { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=2-5")
            return (206, ["Content-Range": "bytes 2-5/8", "Content-Length": "4"], Data([2, 3, 4, 5]))
        }
        let second = makeFixture { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=0-3")
            return (206, ["Content-Range": "bytes 0-3/8", "Content-Length": "4"], Data([0, 1, 2, 3]))
        }
        async let a = first.transfer.fetch(url: first.url, range: 2..<6, totalByteCount: 8)
        async let b = second.transfer.fetch(url: second.url, range: 0..<4, totalByteCount: 8)
        let (aData, bData) = try await (a, b)
        XCTAssertEqual(aData, Data([2, 3, 4, 5]))
        XCTAssertEqual(bData, Data([0, 1, 2, 3]))
    }
}

private final class StubProtocol: URLProtocol, @unchecked Sendable {
    typealias Reply = (Int, [String: String], Data)
    typealias Handler = @Sendable (URLRequest) -> Reply?
    // URLProtocol requires synchronous callbacks on Foundation-owned threads.
    // All registry access holds this lock; invoke a copied handler outside it.
    // Each URL belongs to one fixture, including its delayed cancellation work.
    static let lock = NSLock()
    static var handlers: [URL: Handler] = [:]
    static func configure(_ handler: @escaping Handler, for url: URL) {
        lock.lock(); defer { lock.unlock() }
        handlers[url] = handler
    }
    static func removeHandler(for url: URL) {
        lock.lock(); defer { lock.unlock() }
        handlers.removeValue(forKey: url)
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock()
        let handler = request.url.flatMap { Self.handlers[$0] }
        Self.lock.unlock()
        guard let (status, headers, body) = handler?(request) else { return }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() { }
}
