import Foundation

public struct Citation: Codable, Identifiable, Sendable {
    public var id: Int { citationIndex }
    public let citationIndex: Int
    public let sourceId: String
    public let sourceTitle: String
    public let sectionTitle: String
    public let quote: String
    public let isGrounded: Bool
    public let matchConfidence: Double

    enum CodingKeys: String, CodingKey {
        case citationIndex = "citation_index"
        case sourceId = "source_id"
        case sourceTitle = "source_title"
        case sectionTitle = "section_title"
        case quote
        case isGrounded = "is_grounded"
        case matchConfidence = "match_confidence"
    }
}

public struct RetrievedChunk: Codable, Identifiable, Sendable {
    public var id: String { chunkId }
    public let chunkId: String
    public let docId: String
    public let sourceTitle: String
    public let sectionTitle: String
    public let content: String
    public let tokenCount: Int?
    public let rrfScore: Double?
    public let denseRank: Int?
    public let sparseRank: Int?
    public let finalRank: Int?

    enum CodingKeys: String, CodingKey {
        case chunkId = "chunk_id"
        case docId = "doc_id"
        case sourceTitle = "source_title"
        case sectionTitle = "section_title"
        case content
        case tokenCount = "token_count"
        case rrfScore = "rrf_score"
        case denseRank = "dense_rank"
        case sparseRank = "sparse_rank"
        case finalRank = "final_rank"
    }
}

public struct RAGResponse: Codable, Sendable {
    public let query: String
    public let answer: String
    public let citations: [Citation]
    public let retrievedChunks: [RetrievedChunk]
    public let confidenceScore: Double
    public let faithfulnessScore: Double
    public let citationPrecision: Double
    public let retrievalLatencyMs: Double
    public let totalLatencyMs: Double

    enum CodingKeys: String, CodingKey {
        case query
        case answer
        case citations
        case retrievedChunks = "retrieved_chunks"
        case confidenceScore = "confidence_score"
        case faithfulnessScore = "faithfulness_score"
        case citationPrecision = "citation_precision"
        case retrievalLatencyMs = "retrieval_latency_ms"
        case totalLatencyMs = "total_latency_ms"
    }
}

public struct EvalScorecard: Codable, Sendable {
    public let totalTestCases: Int
    public let recallAt5: Double
    public let mrr: Double
    public let meanFaithfulness: Double
    public let meanCitationPrecision: Double
    public let p50LatencyMs: Double
    public let p95LatencyMs: Double
    public let costPer1kQueriesUsd: Double

    enum CodingKeys: String, CodingKey {
        case totalTestCases = "total_test_cases"
        case recallAt5 = "recall_at_5"
        case mrr
        case meanFaithfulness = "mean_faithfulness"
        case meanCitationPrecision = "mean_citation_precision"
        case p50LatencyMs = "p50_latency_ms"
        case p95LatencyMs = "p95_latency_ms"
        case costPer1kQueriesUsd = "cost_per_1k_queries_usd"
    }
}
