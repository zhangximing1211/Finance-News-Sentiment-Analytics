import type { ReactNode } from "react";

type MetricCardProps = {
  label: string;
  value: string;
  subtitle?: string;
  emphasis?: boolean;
  children?: ReactNode;
};

export function MetricCard({ label, value, subtitle, emphasis = false, children }: MetricCardProps) {
  return (
    <article className={`metric-card${emphasis ? " metric-card-emphasis" : ""}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      {subtitle ? <p className="metric-subtitle">{subtitle}</p> : null}
      {children}
    </article>
  );
}
