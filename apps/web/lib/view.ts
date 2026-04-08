import type { AnalysisResultRecord, SentimentLabel } from "../../../packages/schemas/ts";

export function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
  }).format(new Date(value));
}

export function sentimentTone(label: string): "positive" | "neutral" | "negative" {
  if (label === "positive") {
    return "positive";
  }
  if (label === "negative") {
    return "negative";
  }
  return "neutral";
}

export function labelToChinese(label: SentimentLabel): string {
  if (label === "positive") {
    return "积极";
  }
  if (label === "negative") {
    return "消极";
  }
  return "中性";
}

export function buildTrendSeries(items: AnalysisResultRecord[]): Array<{
  day: string;
  positive: number;
  neutral: number;
  negative: number;
}> {
  const bucket = new Map<string, { day: string; positive: number; neutral: number; negative: number }>();

  items.forEach((item) => {
    const day = item.created_at.slice(0, 10);
    if (!bucket.has(day)) {
      bucket.set(day, { day, positive: 0, neutral: 0, negative: 0 });
    }
    const entry = bucket.get(day);
    if (!entry) {
      return;
    }
    if (item.final_label === "positive") {
      entry.positive += 1;
    } else if (item.final_label === "negative") {
      entry.negative += 1;
    } else {
      entry.neutral += 1;
    }
  });

  return [...bucket.values()].sort((left, right) => left.day.localeCompare(right.day)).slice(-10);
}
