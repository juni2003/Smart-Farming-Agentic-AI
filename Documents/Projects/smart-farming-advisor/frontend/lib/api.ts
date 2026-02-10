const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5000";

export type CropPayload = {
  N: number;
  P: number;
  K: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
};

export async function recommendCrop(payload: CropPayload) {
  const response = await fetch(`${API_BASE}/api/crop/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function detectDisease(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/disease/predict`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

export async function askQuestion(question: string, useLLM = false) {
  const response = await fetch(`${API_BASE}/api/qa`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, use_llm: useLLM })
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}
