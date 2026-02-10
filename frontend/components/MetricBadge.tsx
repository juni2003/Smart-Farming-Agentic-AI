type Props = {
  label: string;
  value: string;
};

export default function MetricBadge({ label, value }: Props) {
  return (
    <div className="rounded-xl bg-white/70 px-4 py-3 shadow-soft">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="text-lg font-semibold text-leaf-800">{value}</p>
    </div>
  );
}
