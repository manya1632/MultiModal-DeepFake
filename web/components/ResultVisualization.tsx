"use client";

import { useEffect, useRef } from "react";
import type { DetectionResult } from "@/types/detection";

interface Props {
  result: DetectionResult;
  imageUrl: string;
  caption: string;
}

export default function ResultVisualization({ result, imageUrl, caption }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      if (result.label === "fake" && result.bbox) {
        const [cx, cy, w, h] = result.bbox;
        const x = (cx - w / 2) * img.naturalWidth;
        const y = (cy - h / 2) * img.naturalHeight;
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = Math.max(2, img.naturalWidth / 200);
        ctx.strokeRect(x, y, w * img.naturalWidth, h * img.naturalHeight);
      }
    };
    img.src = imageUrl;
  }, [imageUrl, result]);

  const tokens = caption.split(/\s+/);
  const fakeSet = new Set(result.fake_token_positions);

  return (
    <div className="flex flex-col gap-6">
      {/* Label + scores */}
      <div className="flex flex-wrap gap-3 items-center">
        <span className={`px-3 py-1 rounded-full text-sm font-semibold ${result.label === "fake" ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}>
          {result.label === "fake" ? "Fake" : "Real"}
        </span>
        <span className="text-sm text-gray-600">
          Trust Score: <strong>{(result.trust_score * 100).toFixed(1)}%</strong>
        </span>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${result.watermark_valid ? "bg-blue-100 text-blue-700" : "bg-yellow-100 text-yellow-700"}`}>
          Watermark {result.watermark_valid ? "Valid" : "Invalid"}
        </span>
      </div>

      {/* Detailed scores */}
      <div className="grid grid-cols-2 gap-2 text-sm text-gray-700">
        <div className="bg-gray-100 rounded p-2">
          <p className="text-xs text-gray-500 mb-0.5">HAMMER Score</p>
          <p className="font-medium">{(result.hammer_score * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-gray-100 rounded p-2">
          <p className="text-xs text-gray-500 mb-0.5">Watermark Score</p>
          <p className="font-medium">{(result.watermark_score * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Image with bounding box */}
      <div>
        <p className="text-xs text-gray-500 mb-1">
          {result.label === "fake" && result.bbox ? "Manipulated region highlighted" : "Image"}
        </p>
        <canvas ref={canvasRef} className="max-w-full rounded border border-gray-200" />
      </div>

      {/* Token highlights */}
      {caption && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Caption</p>
          <p className="text-sm leading-relaxed">
            {tokens.map((token, i) => (
              <span key={i}>
                <span className={fakeSet.has(i) ? "bg-red-200 text-red-800 rounded px-0.5" : ""}>
                  {token}
                </span>
                {i < tokens.length - 1 ? " " : ""}
              </span>
            ))}
          </p>
        </div>
      )}
    </div>
  );
}
