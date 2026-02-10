"use client";

import { useState } from "react";
import SectionTitle from "../../components/SectionTitle";
import FileUploader from "../../components/FileUploader";
import ConfidenceBar from "../../components/ConfidenceBar";
import TopKList from "../../components/TopKList";
import { detectDisease } from "../../lib/api";

export default function DiseasePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!file) {
      setError("Please upload a leaf image.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await detectDisease(file);
      setResult(response?.data ?? response);
    } catch (err: any) {
      setError(err?.message ?? "Failed to detect disease.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-6 py-14">
      <SectionTitle title="Disease Detection" subtitle="Upload a leaf image to identify plant diseases." />

      <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr]">
        <div className="rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
          <FileUploader file={file} onChange={setFile} />
          <button
            onClick={handleSubmit}
            className="mt-6 w-full rounded-full bg-leaf-600 px-6 py-3 text-sm font-semibold text-white shadow-soft hover:bg-leaf-700"
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Detect Disease"}
          </button>
          {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
        </div>

        <div className="rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
          {result ? (
            <div className="space-y-5">
              <div>
                <p className="text-sm text-slate-500">Predicted disease</p>
                <h3 className="text-2xl font-semibold text-leaf-700">{result?.predicted_disease ?? "Unknown"}</h3>
              </div>
              <ConfidenceBar value={result?.confidence ?? 0} />
              <div>
                <p className="mb-2 text-sm font-semibold text-slate-700">Top predictions</p>
                <TopKList
                  items={(result?.top_predictions ?? []).map((item: any) => ({
                    label: item.disease,
                    value: item.probability
                  }))}
                />
              </div>
              <div className="rounded-2xl bg-sun-50 p-4 text-xs text-slate-700">
                Tip: Use a well-lit, close-up leaf photo without blur for best results.
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500">Results will appear here after upload.</div>
          )}
        </div>
      </div>
    </div>
  );
}
