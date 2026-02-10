import SectionTitle from "../../components/SectionTitle";

export default function ModelsPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-14">
      <SectionTitle title="Models & Datasets" subtitle="Understand how predictions are generated." />

      <div className="space-y-6 text-slate-700">
        <div className="rounded-2xl border border-leaf-100 bg-white p-6 shadow-soft">
          <h3 className="text-lg font-semibold">Crop Recommendation Model</h3>
          <p className="mt-2 text-sm">Random Forest with engineered soil and climate features. Accuracy: 99.39%.</p>
        </div>
        <div className="rounded-2xl border border-leaf-100 bg-white p-6 shadow-soft">
          <h3 className="text-lg font-semibold">Disease Detection Model</h3>
          <p className="mt-2 text-sm">ResNet50 transfer learning model trained on PlantVillage images. Accuracy: 98.97%.</p>
        </div>
        <div className="rounded-2xl border border-leaf-100 bg-white p-6 shadow-soft">
          <h3 className="text-lg font-semibold">RAG Q&A Pipeline</h3>
          <p className="mt-2 text-sm">FAISS vector store with Sentence Transformers and optional Gemini response generation.</p>
        </div>
        <div className="rounded-2xl bg-sun-50 p-6">
          <h4 className="text-base font-semibold">Datasets</h4>
          <p className="mt-2 text-sm">Crop recommendation dataset, PlantVillage disease images, and curated farming FAQ Q&A pairs.</p>
        </div>
      </div>
    </div>
  );
}
