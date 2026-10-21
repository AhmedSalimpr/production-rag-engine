import SwiftUI

public struct ScorecardView: View {
    public let scorecard: EvalScorecard?

    public init(scorecard: EvalScorecard?) {
        self.scorecard = scorecard
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Automated LLMOps Benchmark")
                    .font(.title2.bold())
                    .foregroundStyle(.primary)

                Text("Quantitative evaluation across golden technical QA pairs. Demonstrating high-signal production readiness.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if let sc = scorecard {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 160))], spacing: 16) {
                        MetricTile(
                            title: "Recall@5",
                            value: String(format: "%.1f%%", sc.recallAt5 * 100),
                            subtitle: "Target chunk in top-5",
                            color: .green
                        )
                        MetricTile(
                            title: "MRR",
                            value: String(format: "%.3f", sc.mrr),
                            subtitle: "Mean Reciprocal Rank",
                            color: .blue
                        )
                        MetricTile(
                            title: "Faithfulness",
                            value: String(format: "%.1f%%", sc.meanFaithfulness * 100),
                            subtitle: "Grounded assertions",
                            color: .green
                        )
                        MetricTile(
                            title: "Citation Precision",
                            value: String(format: "%.1f%%", sc.meanCitationPrecision * 100),
                            subtitle: "Verbatim quote validity",
                            color: .blue
                        )
                        MetricTile(
                            title: "p95 Latency",
                            value: String(format: "%.1f ms", sc.p95LatencyMs),
                            subtitle: "End-to-end pipeline",
                            color: .primary
                        )
                        MetricTile(
                            title: "Cost / 1k Queries",
                            value: String(format: "$%.4f", sc.costPer1kQueriesUsd),
                            subtitle: "Token budget model",
                            color: .purple
                        )
                    }
                } else {
                    ContentUnavailableView(
                        "No Benchmark Loaded",
                        systemImage: "chart.bar.xaxis",
                        description: Text("Connect to the FastAPI service to view the evaluation scorecard.")
                    )
                }
            }
            .padding(24)
        }
        .navigationTitle("Eval Scorecard")
    }
}

public struct MetricTile: View {
    public let title: String
    public let value: String
    public let subtitle: String
    public let color: Color

    public init(title: String, value: String, subtitle: String, color: Color) {
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.color = color
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)

            Text(value)
                .font(.system(.title, design: .monospaced, weight: .bold))
                .foregroundStyle(color)

            Text(subtitle)
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
