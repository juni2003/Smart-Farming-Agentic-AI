import Link from "next/link";
import FeatureCard from "../components/FeatureCard";
import MetricBadge from "../components/MetricBadge";
import SectionTitle from "../components/SectionTitle";

export default function HomePage() {
  return (
    <div>
      <section className="bg-hero-gradient">
        <div className="mx-auto flex max-w-6xl flex-col gap-10 px-6 py-16 md:flex-row md:items-center">
          <div className="flex-1">
            <p className="mb-4 inline-flex items-center gap-2 rounded-full bg-white px-3 py-1 text-xs font-semibold text-leaf-700 shadow-soft">
              🌾 AI for precision agriculture
            </p>
            <h1 className="text-4xl font-bold leading-tight text-slate-900 md:text-5xl">
              Smart Farming Advisor
            </h1>
            <p className="mt-4 text-lg text-slate-700">
              Make confident farming decisions with crop recommendations, disease detection, and a trusted knowledge assistant.
            </p>
            <div className="mt-6 flex flex-wrap gap-4">
              <Link href="/crop" className="rounded-full bg-leaf-600 px-6 py-3 text-sm font-semibold text-white shadow-soft">
                Try Crop Advisor
              </Link>
              <Link href="/disease" className="rounded-full border border-leaf-200 bg-white px-6 py-3 text-sm font-semibold text-leaf-700 shadow-soft">
                Detect Disease
              </Link>
              <Link href="/qa" className="rounded-full border border-sun-200 bg-white px-6 py-3 text-sm font-semibold text-sun-700 shadow-soft">
                Ask a Question
              </Link>
            </div>
            <div className="mt-8 grid grid-cols-2 gap-4 md:grid-cols-3">
              <MetricBadge label="Crop Accuracy" value="99.39%" />
              <MetricBadge label="Disease Accuracy" value="98.97%" />
              <MetricBadge label="RAG Hit Rate" value="100%" />
            </div>
          </div>
          <div className="flex-1">
            <div className="rounded-3xl border border-white/60 bg-white/80 p-6 shadow-soft">
              <img src="/images/crops.svg" alt="Crops illustration" className="mb-6 w-full rounded-2xl" />
              <div className="grid gap-4">
                <div className="rounded-2xl bg-white p-4 shadow-soft">
                  <p className="text-xs text-slate-500">Live crop insights</p>
                  <p className="mt-1 text-lg font-semibold text-leaf-800">Rice • 92% confidence</p>
                </div>
                <div className="rounded-2xl bg-white p-4 shadow-soft">
                  <p className="text-xs text-slate-500">Disease detection</p>
                  <p className="mt-1 text-lg font-semibold text-slate-900">Tomato Late Blight</p>
                </div>
                <div className="rounded-2xl bg-white p-4 shadow-soft">
                  <p className="text-xs text-slate-500">Q&A assistant</p>
                  <p className="mt-1 text-lg font-semibold text-slate-900">“Water tomatoes every 2-3 days.”</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16">
        <SectionTitle title="Key Capabilities" subtitle="Three tools, one intelligent advisor." />
        <div className="grid gap-6 md:grid-cols-3">
          <FeatureCard
            icon="🌱"
            title="Crop Recommendation"
            description="Predict the best crops using soil nutrients and climate features with ML-powered accuracy."
          />
          <FeatureCard
            icon="🍃"
            title="Disease Detection"
            description="Upload leaf images to detect plant diseases and get confidence-ranked diagnoses."
          />
          <FeatureCard
            icon="🤖"
            title="Farming Q&A"
            description="Ask questions and receive trusted, retrieval-based answers from a curated knowledge base."
          />
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-16">
        <div className="gradient-card rounded-3xl p-10 shadow-soft">
          <SectionTitle
            title="Ready to test the full system?"
            subtitle="Run the crop advisor, detect diseases, or chat with the knowledge base in minutes."
          />
          <div className="flex flex-wrap gap-4">
            <Link href="/crop" className="rounded-full bg-leaf-600 px-6 py-3 text-sm font-semibold text-white shadow-soft">
              Start Crop Test
            </Link>
            <Link href="/disease" className="rounded-full bg-sun-500 px-6 py-3 text-sm font-semibold text-white shadow-soft">
              Start Disease Test
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
