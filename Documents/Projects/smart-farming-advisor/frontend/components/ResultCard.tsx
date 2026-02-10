type Props = {
  title: string;
  description: string;
};

export default function ResultCard({ title, description }: Props) {
  return (
    <div className="rounded-2xl border border-leaf-100 bg-white p-5 shadow-soft">
      <h4 className="text-base font-semibold text-slate-900">{title}</h4>
      <p className="mt-2 text-sm text-slate-600">{description}</p>
    </div>
  );
}
