import SwiftUI

public enum NavigationTab: String, Hashable, CaseIterable {
    case query = "Studio Query"
    case scorecard = "Eval Scorecard"
}

public struct ContentView: View {
    @State private var service = RAGAPIService()
    @State private var selectedTab: NavigationTab? = .query
    @State private var selectedCitationIndex: Int? = nil

    public init() {}

    public var body: some View {
        NavigationSplitView {
            List(NavigationTab.allCases, id: \.self, selection: $selectedTab) { tab in
                NavigationLink(value: tab) {
                    Label(
                        tab.rawValue,
                        systemImage: tab == .query ? "bubble.left.and.text.bubble.right.fill" : "chart.line.uptrend.xyaxis"
                    )
                }
            }
            .navigationTitle("RAG Studio")
            .listStyle(.sidebar)
        } content: {
            switch selectedTab {
            case .query, .none:
                QueryView(service: service, selectedCitationIndex: $selectedCitationIndex)
            case .scorecard:
                ScorecardView(scorecard: service.latestScorecard)
            }
        } detail: {
            InspectorView(
                chunks: service.latestResponse?.retrievedChunks ?? [],
                selectedChunkIndex: selectedCitationIndex
            )
        }
        .task {
            await service.fetchMetrics()
        }
    }
}
