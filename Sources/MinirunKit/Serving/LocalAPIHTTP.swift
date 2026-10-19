import Foundation

struct LocalHTTPRequest {
    let method, path: String
    let headers: [String: String]
    let body: Data
}

/// One bounded HTTP/1.1 request per connection. No ambiguous framing, request
/// compression, chunked uploads or pipelining enters the inference owner.
struct LocalHTTPParser {
    private var buffer = Data()
    mutating func append(_ data: Data) throws -> LocalHTTPRequest? {
        buffer.append(data)
        guard buffer.count <= 262144 + 16384 else { throw LocalAPIError(status: 413, message: "Request too large") }
        guard let boundary = buffer.range(of: Data([13, 10, 13, 10])) else {
            if buffer.count > 16384 { throw LocalAPIError(status: 431, message: "Headers too large") }
            return nil
        }
        guard boundary.lowerBound <= 16384,
              let text = String(data: buffer[..<boundary.lowerBound], encoding: .utf8) else {
            throw LocalAPIError(status: 400, message: "Invalid HTTP headers")
        }
        let lines = text.components(separatedBy: "\r\n")
        let start = (lines.first ?? "").split(separator: " ", omittingEmptySubsequences: false)
        guard start.count == 3, start[2] == "HTTP/1.1", start[1].hasPrefix("/"),
              start[0] == "GET" || start[0] == "POST" else {
            throw LocalAPIError(status: 400, message: "Expected HTTP/1.1 GET or POST")
        }
        var headers: [String: String] = [:]
        let tokenCharacters = Set("abcdefghijklmnopqrstuvwxyz0123456789!#$%&'*+-.^_`|~".utf8)
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { throw LocalAPIError(status: 400, message: "Invalid header") }
            let name = line[..<colon].lowercased(), value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            guard !name.isEmpty, name.utf8.allSatisfy({ tokenCharacters.contains($0) }),
                  !value.contains("\r"), !value.contains("\n"), headers[name] == nil else {
                throw LocalAPIError(status: 400, message: "Duplicate or malformed header")
            }
            headers[name] = value
        }
        guard headers["host"]?.isEmpty == false, headers["transfer-encoding"] == nil,
              headers["content-encoding"] == nil || headers["content-encoding"] == "identity" else {
            throw LocalAPIError(status: 400, message: "Unsupported request framing")
        }
        let length: Int
        if let field = headers["content-length"] {
            guard !field.isEmpty, field.utf8.allSatisfy({ (48...57).contains($0) }),
                  let count = Int(field), count <= 262144 else { throw LocalAPIError(status: 413, message: "Invalid body length") }
            length = count
        } else {
            guard start[0] == "GET" else { throw LocalAPIError(status: 411, message: "Content-Length is required") }
            length = 0
        }
        let available = buffer.count - boundary.upperBound
        guard available <= length else { throw LocalAPIError(status: 400, message: "Pipelining is unsupported") }
        guard available == length else { return nil }
        return .init(method: String(start[0]), path: String(start[1]), headers: headers,
            body: Data(buffer[boundary.upperBound...]))
    }
}

enum LocalAPIWire {
    static func json(_ object: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) }
    static func head(status: Int = 200, type: String = "application/json", length: Int? = nil) -> Data {
        let reason = status == 200 ? "OK" : "Error"
        var value = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: \(type); charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n"
        if let length { value += "Content-Length: \(length)\r\n" }
        if status == 401 { value += "WWW-Authenticate: Bearer\r\n" }
        if status == 429 { value += "Retry-After: 2\r\n" }
        return Data((value + "\r\n").utf8)
    }
    static func chunk(id: String, created: Int, delta: [String: Any], finish: String? = nil) throws -> Data {
        let value: [String: Any] = ["id": id, "object": "chat.completion.chunk", "created": created,
            "model": LocalChatRequest.modelID, "usage": NSNull(),
            "choices": [["index": 0, "delta": delta, "finish_reason": finish as Any? ?? NSNull()]]]
        return try sse(value)
    }
    static func sse(_ object: [String: Any]) throws -> Data {
        var data = Data("data: ".utf8); data.append(try json(object)); data.append(Data("\n\n".utf8)); return data
    }
    static func authenticated(_ header: String?, key: String) -> Bool {
        let expected = Array(("Bearer " + key).utf8), actual = Array((header ?? "").utf8)
        guard actual.count == expected.count else { return false }
        var difference: UInt8 = 0
        for index in expected.indices { difference |= actual[index] ^ expected[index] }
        return difference == 0
    }
}
