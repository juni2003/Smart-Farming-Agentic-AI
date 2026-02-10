"use client";

import { useState } from "react";
import SectionTitle from "../../components/SectionTitle";
import ChatBubble from "../../components/ChatBubble";
import { askQuestion } from "../../lib/api";

export default function QAPage() {
  const [question, setQuestion] = useState("How often should I water tomato plants?");
  const [chat, setChat] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<any[]>([]);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setChat((prev) => [...prev, { role: "user", text: question }]);

    try {
      const response = await askQuestion(question, false);
      setChat((prev) => [...prev, { role: "assistant", text: response.answer }]);
      setSources(response.retrieved_docs || []);
    } catch (err: any) {
      setError(err?.message ?? "Failed to answer question.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl px-6 py-14">
      <SectionTitle title="Farming Q&A" subtitle="Ask practical farming questions and get instant guidance." />

      <div className="rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
        <div className="space-y-4">
          {chat.length === 0 && (
            <div className="text-sm text-slate-500">Your answers will appear here.</div>
          )}
          {chat.map((message, idx) => (
            <ChatBubble key={idx} role={message.role} text={message.text} />
          ))}
        </div>

        <div className="mt-6 flex flex-col gap-3 md:flex-row">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            className="flex-1 rounded-full border border-leaf-100 bg-white px-5 py-3 text-sm shadow-soft focus:border-leaf-400 focus:outline-none"
            placeholder="Ask a farming question..."
          />
          <button
            onClick={handleAsk}
            className="rounded-full bg-leaf-600 px-6 py-3 text-sm font-semibold text-white shadow-soft"
            disabled={loading}
          >
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>
        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
      </div>

      <div className="mt-8 rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
        <h4 className="text-sm font-semibold text-slate-900">Retrieved Knowledge</h4>
        <div className="mt-4 space-y-3 text-sm text-slate-600">
          {sources.length === 0 && <p>No sources yet. Ask a question to view snippets.</p>}
          {sources.map((doc, idx) => (
            <div key={idx} className="rounded-2xl bg-leaf-50 p-4">
              <p className="text-xs text-slate-500">Relevance: {(doc.relevance_score * 100).toFixed(1)}%</p>
              <p className="mt-1">{doc.content}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
