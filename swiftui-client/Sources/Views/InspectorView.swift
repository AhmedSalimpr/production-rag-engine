import SwiftUI

public struct InspectorView: View {
    public let chunks: [RetrievedChunk]
    public var selectedChunkIndex: Int?

    public init(chunks: [RetrievedChunk], selectedChunkIndex: Int? = nil) {
        self.chunks = chunks
        self.selectedChunkIndex = selectedChunkIndex
    }

    public var body: some View {
        List {
            Section("Retrieved Chunks & RRF Fusion (\(chunks.count))") {
                if chunks.isEmpty {
                    ContentUnavailableView(
                        "No Chunks Retrieved",
                        systemImage: "doc.text.magnifyingglass",
                        description: Text("Run a query to inspect dense and sparse fusion candidates.")
                    )
                } else {
                    ForEach(Array(chunks.enumerated()), id: \.element.id) { index, chunk in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(chunk.sourceTitle)
                                    .font(.headline)
                                    .foregroundStyle(.primary)

                                Spacer()

                                if let rrf = chunk.rrfScore {
                                    Text(String(format: "RRF: %.4f", rrf))
                                        .font(.caption2.monospaced())
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background(Color.blue.opacity(0.15))
                                        .foregroundStyle(.blue)
                                        .clipShape(Capsule())
                                }
                            }

                            HStack(spacing: 12) {
                                if let dense = chunk.denseRank {
                                    Text("Dense: #\(dense)")
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.secondary)
                                }
                                if let sparse = chunk.sparseRank {
                                    Text("BM25: #\(sparse)")
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.secondary)
                                }
                                if let tokens = chunk.tokenCount {
                                    Text("\(tokens) tokens")
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.tertiary)
                                }
                            }

                            Text(chunk.content)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(6)
                                .padding(8)
                                .background(Color.secondary.opacity(0.08))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .padding(.vertical, 4)
                        .listRowBackground(
                            (selectedChunkIndex == index + 1)
                                ? Color.blue.opacity(0.12)
                                : Color.clear
                        )
                    }
                }
            }
        }
        .navigationTitle("Chunk Inspector")
    }
}
