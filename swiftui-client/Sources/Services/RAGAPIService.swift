import Foundation
import Observation

@Observable
public final class RAGAPIService: @unchecked Sendable {
    public var baseURL: URL = URL(string: "http://127.0.0.1:8000")!
    public var isLoading: Bool = false
    public var errorMessage: String? = nil
    public var latestResponse: RAGResponse? = nil
    public var latestScorecard: EvalScorecard? = nil

    public init() {}

    public func executeQuery(query: String, rerankK: Int = 5) async {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        isLoading = true
        errorMessage = nil

        let endpoint = baseURL.appendingPathComponent("api/query")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let payload: [String: Any] = [
            "query": query,
            "dense_top_k": 15,
            "sparse_top_k": 15,
            "rerank_top_k": rerankK,
            "dense_weight": 1.0,
            "sparse_weight": 1.0
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }

            let decoder = JSONDecoder()
            let result = try decoder.decode(RAGResponse.self, from: data)
            
            await MainActor.run {
                self.latestResponse = result
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.errorMessage = "Failed to query RAG API: \(error.localizedDescription)"
                self.isLoading = false
            }
        }
    }

    public func fetchMetrics() async {
        let endpoint = baseURL.appendingPathComponent("api/metrics")
        do {
            let (data, _) = try await URLSession.shared.data(from: endpoint)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            if let scorecardDict = json?["scorecard"] {
                let scorecardData = try JSONSerialization.data(withJSONObject: scorecardDict)
                let scorecard = try JSONDecoder().decode(EvalScorecard.self, from: scorecardData)
                await MainActor.run {
                    self.latestScorecard = scorecard
                }
            }
        } catch {
            print("Failed to fetch scorecard: \(error)")
        }
    }
}
