import SectionTitle from "../../components/SectionTitle";

export default function ContactPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-14">
      <SectionTitle title="Contact & Feedback" subtitle="Share feedback or request new features." />

      <div className="rounded-3xl border border-leaf-100 bg-white p-6 shadow-soft">
        <form className="grid gap-4">
          <label className="text-sm text-slate-700">
            Name
            <input className="mt-2 w-full rounded-xl border border-leaf-100 px-4 py-2 shadow-soft" placeholder="Your name" />
          </label>
          <label className="text-sm text-slate-700">
            Email
            <input className="mt-2 w-full rounded-xl border border-leaf-100 px-4 py-2 shadow-soft" placeholder="you@email.com" />
          </label>
          <label className="text-sm text-slate-700">
            Message
            <textarea
              className="mt-2 w-full rounded-xl border border-leaf-100 px-4 py-2 shadow-soft"
              rows={5}
              placeholder="Tell us about your experience..."
            />
          </label>
          <button className="rounded-full bg-leaf-600 px-6 py-3 text-sm font-semibold text-white shadow-soft">
            Send Feedback
          </button>
        </form>
      </div>
    </div>
  );
}
