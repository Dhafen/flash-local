import CoreFoundation
import Foundation

/// Immutable wire values. Schema objects stay inside synchronous validation;
/// Foundation's untyped JSON never crosses the inference executor boundary.
public struct LocalToolDefinition: Equatable, Sendable {
    public let name, description, parametersJSON: String
    public let strict: Bool
    public var json: [String: Any] {
        ["type": "function", "function": ["name": name, "description": description,
            "parameters": (try? JSONSerialization.jsonObject(with: Data(parametersJSON.utf8))) ?? [:], "strict": strict]]
    }
    static func parse(_ value: [String: Any]) throws -> Self {
        guard value["type"] as? String == "function", Set(value.keys).isSubset(of: ["type", "function"]),
              let function = value["function"] as? [String: Any],
              Set(function.keys).isSubset(of: ["name", "description", "parameters", "strict"]),
              let name = function["name"] as? String, LocalToolCall.validName(name) else {
            throw LocalAPIError.invalid("tools", "Expected a named function tool")
        }
        let description = function["description"] as? String ?? ""
        guard description.utf8.count <= 8192,
              function["description"] == nil || function["description"] is NSNull || function["description"] is String else {
            throw LocalAPIError.invalid("tools", "Tool descriptions must be bounded text")
        }
        var strict = false
        if let value = function["strict"], !(value is NSNull) {
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw LocalAPIError.invalid("tools", "strict must be a boolean")
            }
            strict = number.boolValue
        }
        let parameters = function["parameters"] as? [String: Any] ?? ["type": "object", "properties": [:], "additionalProperties": false]
        guard function["parameters"] == nil || function["parameters"] is [String: Any],
              parameters["type"] as? String == "object" else {
            throw LocalAPIError.invalid("tools", "Function parameters must have type object")
        }
        if let properties = parameters["properties"] as? [String: Any], !properties.keys.allSatisfy(LocalToolCall.validName) {
            throw LocalAPIError.invalid("tools", "Top-level argument names must fit the model's named-parameter format")
        }
        let data = try JSONSerialization.data(withJSONObject: parameters, options: [.sortedKeys, .fragmentsAllowed])
        guard data.count <= 32768 else { throw LocalAPIError.invalid("tools", "Function schema exceeds 32 KiB") }
        var schema = LocalToolSchema(root: parameters)
        try schema.inspect(parameters)
        return .init(name: name, description: description, parametersJSON: String(decoding: data, as: UTF8.self), strict: strict)
    }
    public func validate(arguments: String) throws {
        guard arguments.utf8.count <= 65536,
              let object = try? JSONSerialization.jsonObject(with: Data(arguments.utf8)) as? [String: Any],
              let root = try? JSONSerialization.jsonObject(with: Data(parametersJSON.utf8)) else {
            throw LocalAPIError.invalid("tool_calls", "Function arguments must be a JSON object of at most 64 KiB")
        }
        var schema = LocalToolSchema(root: root)
        try schema.boundValue(object)
        try schema.validate(object, against: root)
    }
    fileprivate func xmlArgument(_ text: String, name: String) throws -> Any {
        let root = try JSONSerialization.jsonObject(with: Data(parametersJSON.utf8)) as! [String: Any]
        let property = (root["properties"] as? [String: Any])?[name] ?? root["additionalProperties"] ?? true
        // A declared string remains a string, including numeric-looking text.
        // Non-string schemas may accept JSON scalars, arrays or nested objects.
        var candidates: [Any] = [text]
        let json = ["True": "true", "False": "false", "None": "null"][text] ?? text
        if let value = try? JSONSerialization.jsonObject(with: Data(json.utf8), options: [.fragmentsAllowed]) { candidates.append(value) }
        for candidate in candidates {
            var schema = LocalToolSchema(root: root)
            if (try? schema.validate(candidate, against: property)) != nil { return candidate }
        }
        throw LocalAPIError.invalid("tool_calls", "XML parameter does not match its declared schema")
    }
}

public struct LocalToolCall: Codable, Equatable, Sendable {
    public struct Function: Codable, Equatable, Sendable { public let name, arguments: String }
    public let id: String
    public let type: String
    public let function: Function
    public var json: [String: Any] {
        ["id": id, "type": type, "function": ["name": function.name, "arguments": function.arguments]]
    }
    public static func validName(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.count <= 64 && value.utf8.allSatisfy {
            (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || $0 == 95 || $0 == 45
        }
    }
    static func validID(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.count <= 128 && value.utf8.allSatisfy {
            (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || $0 == 95 || $0 == 45
        }
    }
    static func parse(_ value: [String: Any]) throws -> Self {
        guard Set(value.keys).isSubset(of: ["id", "type", "function"]), value["type"] as? String == "function",
              let id = value["id"] as? String, validID(id), let function = value["function"] as? [String: Any],
              Set(function.keys) == ["name", "arguments"], let name = function["name"] as? String, validName(name),
              let arguments = function["arguments"] as? String, arguments.utf8.count <= 65536,
              (try? JSONSerialization.jsonObject(with: Data(arguments.utf8))) is [String: Any] else {
            throw LocalAPIError.invalid("messages", "Malformed function-call history")
        }
        return .init(id: id, type: "function", function: .init(name: name, arguments: arguments))
    }
}

public enum LocalToolChoice: Equatable, Sendable {
    case none, auto, required, named(String)
    public var enabled: Bool { self != .none }
}

/// Parse only the model's completed tool envelope. Never repair arguments,
/// invent a call, select a tool by substring, or emit an unvalidated action.
public enum LocalToolOutput {
    public static func parse(_ text: String, tools: [LocalToolDefinition], choice: LocalToolChoice,
                             parallel: Bool) throws -> (content: String, calls: [LocalToolCall]) {
        guard choice.enabled, !tools.isEmpty else { return (text, []) }
        var remaining = text[...], content = "", calls: [LocalToolCall] = []
        while let open = remaining.range(of: "<tool_call>") {
            content += remaining[..<open.lowerBound]
            let tail = remaining[open.upperBound...]
            guard let close = tail.range(of: "</tool_call>") else { throw malformed() }
            let payload = String(tail[..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard payload.utf8.count <= 65536 else { throw malformed() }
            let value: [String: Any]
            if payload.hasPrefix("<function=") { value = try xml(payload, tools: tools) }
            else if let object = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] { value = object }
            else { throw malformed() }
            guard
                  Set(value.keys) == ["name", "arguments"], let name = value["name"] as? String,
                  let definition = tools.first(where: { $0.name == name }), let arguments = value["arguments"] as? [String: Any] else {
                throw malformed()
            }
            if case .named(let expected) = choice, name != expected { throw malformed() }
            let data = try JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys])
            let encoded = String(decoding: data, as: UTF8.self)
            do { try definition.validate(arguments: encoded) } catch { throw malformed() }
            calls.append(.init(id: "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""),
                type: "function", function: .init(name: name, arguments: encoded)))
            guard calls.count <= 16, parallel || calls.count == 1 else { throw malformed() }
            remaining = tail[close.upperBound...]
        }
        if !calls.isEmpty, !remaining.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { throw malformed() }
        content += remaining
        guard !content.contains("</tool_call>"), !content.contains("<tool_call"), !content.contains("</tool_call") else { throw malformed() }
        if calls.isEmpty, choice == .required { throw malformed() }
        if calls.isEmpty, case .named = choice { throw malformed() }
        return (calls.isEmpty ? text : content.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }
    private static func xml(_ text: String, tools: [LocalToolDefinition]) throws -> [String: Any] {
        guard let end = text.firstIndex(of: ">"), text.hasSuffix("</function>") else { throw malformed() }
        let name = String(text[text.index(text.startIndex, offsetBy: 10)..<end])
        guard let tool = tools.first(where: { $0.name == name }) else { throw malformed() }
        var body = String(text[text.index(after: end)...].dropLast("</function>".count))
        var arguments: [String: Any] = [:]
        while !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            body = body.trimmingCharacters(in: .whitespacesAndNewlines)
            guard body.hasPrefix("<parameter="), let end = body.firstIndex(of: ">") else { throw malformed() }
            let key = String(body[body.index(body.startIndex, offsetBy: 11)..<end])
            guard LocalToolCall.validName(key), arguments[key] == nil,
                  let close = body.range(of: "</parameter>", range: body.index(after: end)..<body.endIndex) else { throw malformed() }
            var value = String(body[body.index(after: end)..<close.lowerBound])
            // Remove only the template's delimiting newlines. Preserve source
            // indentation and meaningful whitespace inside string arguments.
            if value.hasPrefix("\r\n") { value.removeFirst(2) } else if value.hasPrefix("\n") { value.removeFirst() }
            if value.hasSuffix("\r\n") { value.removeLast(2) } else if value.hasSuffix("\n") { value.removeLast() }
            do { arguments[key] = try tool.xmlArgument(value, name: key) } catch { throw malformed() }
            guard arguments.count <= 128 else { throw malformed() }
            body = String(body[close.upperBound...])
        }
        return ["name": name, "arguments": arguments]
    }
    private static func malformed() -> LocalAPIError {
        .init(status: 502, message: "The local model did not produce a complete, declared, schema-valid function call", parameter: "tool_calls")
    }
}

/// Deliberately bounded JSON Schema subset, including local references and
/// common Pydantic unions. Unsupported validation keywords fail explicitly.
private struct LocalToolSchema {
    let root: Any
    var operations = 0
    private mutating func step(_ depth: Int) throws {
        operations += 1
        guard depth <= 24, operations <= 8192 else { throw invalid("Schema or arguments exceed validation bounds") }
    }
    private func invalid(_ message: String) -> LocalAPIError { .invalid("tools", message) }
    private func boolean(_ value: Any) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }
    private func number(_ value: Any) -> Double? {
        guard boolean(value) == nil, let number = value as? NSNumber, number.doubleValue.isFinite else { return nil }
        return number.doubleValue
    }
    mutating func boundValue(_ value: Any, depth: Int = 0) throws {
        try step(depth)
        if let object = value as? [String: Any] {
            guard object.count <= 128 else { throw invalid("Too many argument properties") }
            for child in object.values { try boundValue(child, depth: depth + 1) }
        } else if let array = value as? [Any] {
            guard array.count <= 4096 else { throw invalid("Argument array is too large") }
            for child in array { try boundValue(child, depth: depth + 1) }
        }
    }
    private func reference(_ path: String) throws -> Any {
        guard path.hasPrefix("#/"), path.utf8.count <= 512 else { throw invalid("Only local schema references are supported") }
        var current = root
        for item in path.dropFirst(2).split(separator: "/", omittingEmptySubsequences: false) {
            let key = item.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
            guard let object = current as? [String: Any], let value = object[key] else { throw invalid("Unresolved local schema reference") }
            current = value
        }
        return current
    }
    private func equal(_ left: Any, _ right: Any) -> Bool {
        if let a = left as? [String: Any], let b = right as? [String: Any] {
            return Set(a.keys) == Set(b.keys) && a.allSatisfy { key, value in equal(value, b[key]!) }
        }
        if let a = left as? [Any], let b = right as? [Any] {
            return a.count == b.count && zip(a, b).allSatisfy { equal($0.0, $0.1) }
        }
        if boolean(left) != nil || boolean(right) != nil { return boolean(left) == boolean(right) && boolean(left) != nil }
        if let a = left as? NSNumber, let b = right as? NSNumber { return a.compare(b) == .orderedSame }
        if let a = left as? String, let b = right as? String { return a == b }
        return left is NSNull && right is NSNull
    }
    mutating func inspect(_ raw: Any, depth: Int = 0, references: Set<String> = []) throws {
        try step(depth)
        if boolean(raw) != nil { return }
        guard let schema = raw as? [String: Any] else { throw invalid("Schema nodes must be objects or booleans") }
        let keys: Set<String> = ["type", "properties", "required", "additionalProperties", "items", "enum", "const",
            "anyOf", "oneOf", "allOf", "$defs", "definitions", "$ref", "title", "description", "default", "format",
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems"]
        guard Set(schema.keys).isSubset(of: keys) else { throw invalid("Unsupported JSON Schema validation keyword") }
        if let value = schema["type"] {
            let types = (value as? String).map { [$0] } ?? (value as? [String] ?? [])
            guard !types.isEmpty, Set(types).count == types.count,
                  Set(types).isSubset(of: ["object", "array", "string", "number", "integer", "boolean", "null"]) else { throw invalid("Invalid schema type") }
        }
        if let value = schema["$ref"] {
            guard let path = value as? String else { throw invalid("Invalid schema reference") }
            let target = try reference(path)
            if !references.contains(path) { try inspect(target, depth: depth + 1, references: references.union([path])) }
        }
        for key in ["properties", "$defs", "definitions"] where schema[key] != nil {
            guard let children = schema[key] as? [String: Any], children.count <= 128 else { throw invalid("Invalid schema properties") }
            for child in children.values { try inspect(child, depth: depth + 1, references: references) }
        }
        if let required = schema["required"] {
            guard let names = required as? [String], names.count <= 128, Set(names).count == names.count else { throw invalid("Invalid required properties") }
        }
        for key in ["items", "additionalProperties"] where schema[key] != nil { try inspect(schema[key]!, depth: depth + 1, references: references) }
        for key in ["anyOf", "oneOf", "allOf"] where schema[key] != nil {
            guard let branches = schema[key] as? [Any], !branches.isEmpty, branches.count <= 16 else { throw invalid("Invalid schema union") }
            for branch in branches { try inspect(branch, depth: depth + 1, references: references) }
        }
        if let values = schema["enum"] { guard let values = values as? [Any], !values.isEmpty, values.count <= 256 else { throw invalid("Invalid enum") } }
        for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] where schema[key] != nil {
            guard number(schema[key]!) != nil else { throw invalid("Invalid numeric bound") }
        }
        for key in ["minLength", "maxLength", "minItems", "maxItems"] where schema[key] != nil {
            guard let count = number(schema[key]!), count >= 0, count <= 262144, count.rounded() == count else { throw invalid("Invalid size bound") }
        }
    }
    mutating func validate(_ value: Any, against raw: Any, depth: Int = 0) throws {
        try step(depth)
        if let allowed = boolean(raw) { if !allowed { throw invalid("Schema rejects this argument") }; return }
        guard let schema = raw as? [String: Any] else { throw invalid("Invalid schema") }
        if let path = schema["$ref"] as? String { try validate(value, against: reference(path), depth: depth + 1) }
        if let rawType = schema["type"] {
            let types = (rawType as? String).map { [$0] } ?? (rawType as? [String] ?? [])
            let valid = types.contains { type in
                switch type {
                case "null": value is NSNull
                case "boolean": boolean(value) != nil
                case "number": number(value) != nil
                case "integer": number(value).map { $0.rounded(.towardZero) == $0 } ?? false
                case "string": value is String
                case "array": value is [Any]
                case "object": value is [String: Any]
                default: false
                }
            }
            guard valid else { throw invalid("Argument type differs from schema") }
        }
        if let allowed = schema["enum"] as? [Any], !allowed.contains(where: { equal(value, $0) }) { throw invalid("Argument differs from enum") }
        if let expected = schema["const"], !equal(value, expected) { throw invalid("Argument differs from const") }
        for key in ["anyOf", "oneOf", "allOf"] {
            guard let branches = schema[key] as? [Any] else { continue }
            var matched = 0
            for branch in branches { do { try validate(value, against: branch, depth: depth + 1); matched += 1 } catch {
                if operations > 8192 { throw error }
            } }
            guard key == "anyOf" ? matched > 0 : key == "oneOf" ? matched == 1 : matched == branches.count else { throw invalid("Argument does not match schema union") }
        }
        if let object = value as? [String: Any] {
            guard object.count <= 128 else { throw invalid("Too many argument properties") }
            let properties = schema["properties"] as? [String: Any] ?? [:]
            for name in schema["required"] as? [String] ?? [] where object[name] == nil { throw invalid("A required argument is missing") }
            for (name, child) in object {
                if let childSchema = properties[name] { try validate(child, against: childSchema, depth: depth + 1) }
                else if let additional = schema["additionalProperties"] { try validate(child, against: additional, depth: depth + 1) }
            }
        }
        if let list = value as? [Any] {
            guard list.count <= 4096 else { throw invalid("Argument array is too large") }
            try bounds(Double(list.count), schema: schema, minimum: "minItems", maximum: "maxItems")
            if let items = schema["items"] { for item in list { try validate(item, against: items, depth: depth + 1) } }
        }
        if let string = value as? String { try bounds(Double(string.unicodeScalars.count), schema: schema, minimum: "minLength", maximum: "maxLength") }
        if let value = number(value) {
            try bounds(value, schema: schema, minimum: "minimum", maximum: "maximum")
            if let minimum = schema["exclusiveMinimum"].flatMap(number), value <= minimum { throw invalid("Argument is below its exclusive bound") }
            if let maximum = schema["exclusiveMaximum"].flatMap(number), value >= maximum { throw invalid("Argument is above its exclusive bound") }
        }
    }
    private func bounds(_ value: Double, schema: [String: Any], minimum: String, maximum: String) throws {
        if let bound = schema[minimum].flatMap(number), value < bound { throw invalid("Argument is below its schema bound") }
        if let bound = schema[maximum].flatMap(number), value > bound { throw invalid("Argument exceeds its schema bound") }
    }
}
