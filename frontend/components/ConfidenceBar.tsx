type Props = {
  value: number;
};

export default function ConfidenceBar({ value }: Props) {
  const percent = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
        <span>Confidence</span>
        <span>{percent}%</span>
      </div>
      <div className="h-3 w-full rounded-full bg-leaf-100">
        <div className="h-3 rounded-full bg-leaf-600" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
