import SectionTitle from "../../components/SectionTitle";
import ResultCard from "../../components/ResultCard";

export default function DashboardPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-14">
      <SectionTitle title="Results Dashboard" subtitle="Track recent recommendations and detection history." />

      <div className="grid gap-6 md:grid-cols-2">
        <ResultCard title="Latest Crop Recommendation" description="Rice • 92% confidence • Soil fertility high" />
        <ResultCard title="Latest Disease Detection" description="Tomato Late Blight • 89% confidence" />
        <ResultCard title="Latest Q&A Session" description="Water tomatoes every 2-3 days in warm climates." />
        <ResultCard title="Model Health" description="Crop and disease models loaded. RAG retrieval active." />
      </div>

      <div className="mt-8 rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
        <h4 className="text-base font-semibold text-slate-900">Insights</h4>
        <p className="mt-2 text-sm text-slate-600">
          Export results, monitor confidence trends, and review predictions for quality assurance. This view can be connected to
          backend logs in future iterations.
        </p>
      </div>
    </div>
  );
}
