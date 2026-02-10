type Item = {
  label: string;
  value: number;
};

type Props = {
  items: Item[];
};

export default function TopKList({ items }: Props) {
  return (
    <div className="space-y-2">
      {items.map((item) => (
        <div key={item.label} className="flex items-center justify-between rounded-xl border border-leaf-100 bg-white px-4 py-2">
          <span className="text-sm font-medium text-slate-700">{item.label}</span>
          <span className="text-sm text-slate-600">{Math.round(item.value * 100)}%</span>
        </div>
      ))}
    </div>
  );
}
