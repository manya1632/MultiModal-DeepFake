"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import DetectionForm from "@/components/DetectionForm";
import ResultVisualization from "@/components/ResultVisualization";
import type { DetectionResult } from "@/types/detection";

export default function DetectPage() {
  const router = useRouter();
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [caption, setCaption] = useState<string>("");

  useEffect(() => {
    if (!sessionStorage.getItem("token")) router.push("/");
  }, [router]);

  return (
    <main className="min-h-screen p-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Deepfake Detection</h1>
          <button
            onClick={() => { sessionStorage.removeItem("token"); router.push("/"); }}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            Sign out
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h2 className="text-lg font-semibold mb-4">Submit for Analysis</h2>
            <DetectionForm
              onResult={(res, url, cap) => { setResult(res); setImageUrl(url); setCaption(cap); }}
            />
          </div>

          {result && (
            <div>
              <h2 className="text-lg font-semibold mb-4">Results</h2>
              <ResultVisualization result={result} imageUrl={imageUrl} caption={caption} />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
