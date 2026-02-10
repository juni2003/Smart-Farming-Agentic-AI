"use client";

import { useState } from "react";
import InputField from "../../components/InputField";
import SectionTitle from "../../components/SectionTitle";
import ConfidenceBar from "../../components/ConfidenceBar";
import TopKList from "../../components/TopKList";
import { recommendCrop } from "../../lib/api";
import { toNumber } from "../../lib/validators";

export default function CropPage() {
  const [form, setForm] = useState({
    N: "90",
    P: "42",
    K: "43",
    temperature: "20",
    humidity: "82",
    ph: "6.5",
    rainfall: "202"
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (key: string, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const payload = {
        N: toNumber(form.N),
        P: toNumber(form.P),
        K: toNumber(form.K),
        temperature: toNumber(form.temperature),
        humidity: toNumber(form.humidity),
        ph: toNumber(form.ph),
        rainfall: toNumber(form.rainfall)
      };
      const response = await recommendCrop(payload);
      setResult(response?.data ?? response);
    } catch (err: any) {
      setError(err?.message ?? "Failed to fetch recommendation.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-6 py-14">
      <SectionTitle title="Crop Recommendation" subtitle="Enter soil and climate values to get the best crop match." />

      <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr]">
        <div className="rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
          <div className="grid gap-4 md:grid-cols-2">
            <InputField label="Nitrogen (N)" name="N" value={form.N} onChange={(val) => handleChange("N", val)} />
            <InputField label="Phosphorus (P)" name="P" value={form.P} onChange={(val) => handleChange("P", val)} />
            <InputField label="Potassium (K)" name="K" value={form.K} onChange={(val) => handleChange("K", val)} />
            <InputField label="Temperature (°C)" name="temperature" value={form.temperature} onChange={(val) => handleChange("temperature", val)} />
            <InputField label="Humidity (%)" name="humidity" value={form.humidity} onChange={(val) => handleChange("humidity", val)} />
            <InputField label="Soil pH" name="ph" value={form.ph} onChange={(val) => handleChange("ph", val)} />
            <InputField label="Rainfall (mm)" name="rainfall" value={form.rainfall} onChange={(val) => handleChange("rainfall", val)} />
          </div>
          <button
            onClick={handleSubmit}
            className="mt-6 w-full rounded-full bg-leaf-600 px-6 py-3 text-sm font-semibold text-white shadow-soft hover:bg-leaf-700"
            disabled={loading}
          >
            {loading ? "Running..." : "Get Recommendation"}
          </button>
          {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
        </div>

        <div className="rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
          {result ? (
            <div className="space-y-5">
              <div>
                <p className="text-sm text-slate-500">Recommended crop</p>
                <h3 className="text-2xl font-semibold text-leaf-700">{result?.recommended_crop ?? "Unknown"}</h3>
              </div>
              <ConfidenceBar value={result?.confidence ?? 0} />
              <div>
                <p className="mb-2 text-sm font-semibold text-slate-700">Top alternatives</p>
                <TopKList
                  items={(result?.top_3_recommendations ?? []).map((item: any) => ({
                    label: item.crop,
                    value: item.probability
                  }))}
                />
              </div>
              <div className="rounded-2xl bg-leaf-50 p-4 text-xs text-slate-700">
                Tip: Improve confidence by using accurate soil test values and recent climate data.
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500">
              Results will appear here after you submit the form.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
