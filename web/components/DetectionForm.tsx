"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { DetectionResult } from "@/types/detection";

interface Props {
  onResult: (result: DetectionResult, imageUrl: string, caption: string) => void;
}

export default function DetectionForm({ onResult }: Props) {
  const router = useRouter();
  const [image, setImage] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!image) { setError("Please select an image."); return; }
    setError(null);
    setLoading(true);
    try {
      const token = sessionStorage.getItem("token");
      if (!token) { router.push("/"); return; }

      const form = new FormData();
      form.append("image", image);
      form.append("text", text);

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });

      if (res.status === 401) { sessionStorage.removeItem("token"); router.push("/"); return; }
      if (!res.ok) { const b = await res.json().catch(() => ({})); setError(b.error ?? "Detection failed."); return; }

      const result: DetectionResult = await res.json();
      onResult(result, URL.createObjectURL(image), text);
    } catch {
      setError("An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full">
      <div>
        <label className="block text-sm font-medium mb-1">Image</label>
        <input
          type="file" accept="image/*" required
          onChange={(e) => setImage(e.target.files?.[0] ?? null)}
          className="block w-full text-sm text-gray-600 file:mr-3 file:py-1 file:px-3 file:rounded file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-1">Caption</label>
        <textarea
          value={text} onChange={(e) => setText(e.target.value)} rows={3}
          className="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
          placeholder="Enter the image caption…"
        />
      </div>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      <button
        type="submit" disabled={loading}
        className="bg-blue-600 text-white rounded px-4 py-2 text-sm font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
      >
        {loading && <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
        {loading ? "Analyzing…" : "Analyze"}
      </button>
    </form>
  );
}
