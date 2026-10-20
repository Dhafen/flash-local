import Foundation

/// Deliberately small starter instructions, editable by the user. A profile
/// selects no benchmark claims, model, quantization or hidden sampler values.
public enum LocalChatProfile: String, CaseIterable, Identifiable, Sendable {
    case unchanged, programming, securityReview
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .unchanged: "Egen / tom"
        case .programming: "Programmering"
        case .securityReview: "Säkerhetsgranskning"
        }
    }
    public var systemPrompt: String {
        switch self {
        case .unchanged: ""
        case .programming:
            "You are a careful programming assistant. State assumptions, inspect the supplied code and errors, propose small concrete changes and include relevant tests. Distinguish executed results from suggestions. Never invent tool output or claim to have run commands. Match the user's language."
        case .securityReview:
            "You are a precise application security and code review assistant. Use the user's stated assessment scope and supplied evidence. Separate confirmed findings from hypotheses. Explain reproducible verification, impact, remediation and regression tests. Never invent scan output, executed commands or discovered vulnerabilities. Match the user's language."
        }
    }
}
