import SwiftUI

public struct QueryView: View {
    @Bindable public var service: RAGAPIService
    @State private var queryText: String = "What is the formula and standard constant for Reciprocal Rank Fusion?"
    @State private var rerankTopK: Double = 5.0
    @Binding public var selectedCitationIndex: Int?

    public init(service: RAGAPIService, selectedCitationIndex: Binding<Int?>) {
        self.service = service
        self._selectedCitationIndex = selectedCitationIndex
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Quick Suggestion Chips
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        SuggestionButton("RRF Formula", action: {
                            queryText = "What is the formula and standard constant for Reciprocal Rank Fusion?"
                        })
                        SuggestionButton("Why 2-Stage Retrieval?", action: {
                            queryText = "Why do production RAG systems use a two-stage retrieve and rerank architecture?"
                        })
                        SuggestionButton("HNSW Hyperparameters", action: {
                            queryText = "What are the key hyperparameters controlling trade-offs in HNSW index construction and search?"
                        })
                        SuggestionButton("QLoRA Innovations", action: {
                            queryText = "What three innovations were introduced by QLoRA for 4-bit fine-tuning?"
                        })
                    }
                }

                // Search Input Glass Bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)

                    TextField("Ask an AI engineering question...", text: $queryText)
                        .textFieldStyle(.plain)
                        .onSubmit {
                            Task {
                                await service.executeQuery(query: queryText, rerankK: Int(rerankTopK))
                            }
                        }

                    Button {
                        Task {
                            await service.executeQuery(query: queryText, rerankK: Int(rerankTopK))
                        }
                    } label: {
                        if service.isLoading {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Text("Run Query")
                                .fontWeight(.semibold)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(service.isLoading)
                }
                .padding(12)
                .background(Color.secondary.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 12))

                // Error Message if present
                if let error = service.errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                // Response Card
                if let response = service.latestResponse {
                    // Telemetry HUD
                    HStack(spacing: 16) {
                        TelemetryBadge(label: "Retrieval", value: String(format: "%.0f ms", response.retrievalLatencyMs))
                        TelemetryBadge(label: "Total Latency", value: String(format: "%.0f ms", response.totalLatencyMs))
                        TelemetryBadge(label: "Faithfulness", value: String(format: "%.0f%%", response.faithfulnessScore * 100), color: .green)
                        TelemetryBadge(label: "Citation Precision", value: String(format: "%.0f%%", response.citationPrecision * 100), color: .blue)
                    }
                    .padding(12)
                    .background(Color.secondary.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                    // Answer Box
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Label("Grounded Answer", systemImage: "checkmark.seal.fill")
                                .font(.headline)
                                .foregroundStyle(.blue)

                            Spacer()

                            Text(String(format: "Confidence: %.2f", response.confidenceScore))
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }

                        Text(response.answer)
                            .font(.body)
                            .lineSpacing(4)
                            .foregroundStyle(.primary)

                        Divider()

                        // Citations Provenance
                        Text("Citations & Provenance")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)

                        ForEach(response.citations) { cit in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("[\(cit.citationIndex)] \(cit.sourceTitle) • \(cit.sectionTitle)")
                                        .font(.caption.weight(.medium))
                                        .foregroundStyle(.primary)

                                    Spacer()

                                    Text(cit.isGrounded ? "✓ Grounded" : "⚠ Ungrounded")
                                        .font(.caption2.weight(.bold))
                                        .foregroundStyle(cit.isGrounded ? .green : .red)
                                }

                                Text("\"\(cit.quote)\"")
                                    .font(.caption)
                                    .italic()
                                    .foregroundStyle(.secondary)
                                    .padding(.leading, 8)
                                    .overlay(alignment: .leading) {
                                        Rectangle()
                                            .fill(Color.blue)
                                            .frame(width: 2)
                                    }
                            }
                            .padding(10)
                            .background(Color.secondary.opacity(0.06))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .onTapGesture {
                                selectedCitationIndex = cit.citationIndex
                            }
                        }
                    }
                    .padding(20)
                    .background(Color.secondary.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
            }
            .padding(24)
        }
        .navigationTitle("Studio Query")
    }
}

public struct SuggestionButton: View {
    public let title: String
    public let action: () -> Void

    public init(_ title: String, action: @escaping () -> Void) {
        self.title = title
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Text(title)
                .font(.caption.weight(.medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.secondary.opacity(0.12))
                .foregroundStyle(.primary)
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

public struct TelemetryBadge: View {
    public let label: String
    public let value: String
    public var color: Color = .primary

    public var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
            Text(value)
                .font(.caption.monospaced().weight(.semibold))
                .foregroundStyle(color)
        }
    }
}
