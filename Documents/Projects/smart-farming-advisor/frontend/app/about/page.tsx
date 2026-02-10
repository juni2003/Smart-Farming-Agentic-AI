import SectionTitle from "../../components/SectionTitle";

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-14">
      <SectionTitle title="About Smart Farming Advisor" subtitle="An agentic AI system for precision agriculture." />

      <div className="space-y-6 text-slate-700">
        <p>
          Smart Farming Advisor combines machine learning, computer vision, and retrieval-augmented generation to help farmers make
          better decisions. The system is composed of three specialized tools that are orchestrated by an intelligent routing agent.
        </p>
        <div className="grid gap-6 md:grid-cols-3">
          <div className="rounded-2xl border border-leaf-100 bg-white p-5 shadow-soft">
            <h3 className="text-base font-semibold text-slate-900">Crop Recommendation</h3>
            <p className="mt-2 text-sm">Predicts the best crop using NPK, climate, and engineered soil metrics.</p>
          </div>
          <div className="rounded-2xl border border-leaf-100 bg-white p-5 shadow-soft">
            <h3 className="text-base font-semibold text-slate-900">Disease Detection</h3>
            <p className="mt-2 text-sm">Uses a ResNet50 model to classify plant leaf diseases with high accuracy.</p>
          </div>
          <div className="rounded-2xl border border-leaf-100 bg-white p-5 shadow-soft">
            <h3 className="text-base font-semibold text-slate-900">Farming Q&A</h3>
            <p className="mt-2 text-sm">Retrieves answers from a curated knowledge base with optional LLM generation.</p>
          </div>
        </div>

        <div className="rounded-2xl bg-slate-900 p-6 text-white">
          <h4 className="text-lg font-semibold">System Architecture</h4>
          <p className="mt-2 text-sm text-slate-200">
            User query → Intent routing → Crop tool / Disease tool / RAG tool → Response + confidence and evidence.
          </p>
        </div>

        <div className="rounded-2xl border border-sun-200 bg-sun-50 p-6">
          <h4 className="text-lg font-semibold text-slate-900">Limitations</h4>
          <p className="mt-2 text-sm">
            Predictions depend on data quality and image clarity. Always validate recommendations with local agronomy knowledge before
            large-scale decisions.
          </p>
        </div>
      </div>
    </div>
  );
}
