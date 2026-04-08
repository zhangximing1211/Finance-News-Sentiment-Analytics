type StatusPillProps = {
  tone?: "positive" | "neutral" | "negative" | "info" | "warning";
  children: string;
};

export function StatusPill({ tone = "info", children }: StatusPillProps) {
  return <span className={`status-pill status-pill-${tone}`}>{children}</span>;
}
