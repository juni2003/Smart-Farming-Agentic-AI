type Props = {
  label: string;
  name: string;
  type?: string;
  placeholder?: string;
  value: string | number;
  onChange: (value: string) => void;
};

export default function InputField({ label, name, type = "number", placeholder, value, onChange }: Props) {
  return (
    <label className="flex flex-col gap-2 text-sm text-slate-700">
      {label}
      <input
        name={name}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-xl border border-leaf-100 bg-white px-4 py-2 text-slate-900 shadow-soft focus:border-leaf-400 focus:outline-none"
      />
    </label>
  );
}
