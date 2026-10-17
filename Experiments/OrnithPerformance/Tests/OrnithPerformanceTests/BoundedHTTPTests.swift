import Foundation
import XCTest
@testable import OrnithPerformance

final class BoundedHTTPTests: XCTestCase {
    override func tearDown() { StubProtocol.configure(nil); super.tearDown() }

    func transfer() -> BoundedHTTPTransfer {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        return BoundedHTTPTransfer(configuration: config)
    }

    func testExact206RangeIsAcceptedAndBodyRemainsBounded() async throws {
        StubProtocol.configure { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=2-5")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Accept-Encoding"), "identity")
            return (206, ["Content-Range": "bytes 2-5/8", "Content-Length": "4"], Data([2, 3, 4, 5]))
        }
        let data = try await transfer().fetch(url: URL(string: "https://fixture.invalid/model")!,
                                              range: 2..<6, totalByteCount: 8)
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
            StubProtocol.configure { _ in (status, headers, body) }
            do {
                _ = try await transfer().fetch(url: URL(string: "https://fixture.invalid/model")!,
                                                range: 2..<6, totalByteCount: 8)
                XCTFail("invalid range/length response must fail")
            } catch { }
        }
    }

    func testCancellationCompletesEvenWhenServerNeverSendsAResponse() async throws {
        StubProtocol.configure { _ in nil }
        let transfer = transfer()
        let task = Task { try await transfer.fetch(url: URL(string: "https://fixture.invalid/stall")!,
                                                   range: 0..<4, totalByteCount: 8) }
        try await Task.sleep(nanoseconds: 20_000_000)
        task.cancel()
        do { _ = try await task.value; XCTFail("cancellation expected") }
        catch is CancellationError { }
    }
}

private final class StubProtocol: URLProtocol, @unchecked Sendable {
    typealias Reply = (Int, [String: String], Data)
    static let lock = NSLock()
    static var handler: ((URLRequest) -> Reply?)?
    static func configure(_ handler: ((URLRequest) -> Reply?)?) {
        lock.lock(); defer { lock.unlock() }
        self.handler = handler
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock()
        let handler = Self.handler
        Self.lock.unlock()
        guard let (status, headers, body) = handler?(request) else { return }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() { }
}
