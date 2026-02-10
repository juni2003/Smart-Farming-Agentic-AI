type Props = {
  file: File | null;
  onChange: (file: File | null) => void;
};

export default function FileUploader({ file, onChange }: Props) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-dashed border-leaf-200 bg-white p-6 text-sm text-slate-600">
      <input
        type="file"
        accept="image/*"
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      {file && (
        <div className="rounded-xl bg-leaf-50 p-3 text-slate-700">
          Selected: {file.name}
        </div>
      )}
    </div>
  );
}
