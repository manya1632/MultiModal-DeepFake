export interface DetectionResult {
  label: "real" | "fake";
  trust_score: number;
  hammer_score: number;
  watermark_score: number;
  watermark_valid: boolean;
  bbox: [number, number, number, number] | null; // [cx, cy, w, h] normalized
  fake_token_positions: number[];
}
