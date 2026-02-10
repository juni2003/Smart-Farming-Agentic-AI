type Props = {
  title: string;
  description: string;
  icon: string;
};

export default function FeatureCard({ title, description, icon }: Props) {
  return (
    <div className="rounded-2xl border border-leaf-100 bg-white p-6 shadow-soft">
      <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-leaf-100 text-2xl">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
      <p className="mt-2 text-sm text-slate-600">{description}</p>
    </div>
  );
}
