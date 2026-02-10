import Link from "next/link";

const links = [
  { href: "/", label: "Home" },
  { href: "/crop", label: "Crop" },
  { href: "/disease", label: "Disease" },
  { href: "/qa", label: "Q&A" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/models", label: "Models" },
  { href: "/about", label: "About" },
  { href: "/contact", label: "Contact" }
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-leaf-100 bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-leaf-800">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-leaf-100 text-leaf-700">🌾</span>
          Smart Farming Advisor
        </Link>
        <nav className="hidden gap-4 text-sm font-medium text-slate-700 md:flex">
          {links.map((link) => (
            <Link key={link.href} href={link.href} className="hover:text-leaf-700">
              {link.label}
            </Link>
          ))}
        </nav>
        <Link
          href="/crop"
          className="rounded-full bg-leaf-600 px-4 py-2 text-sm font-semibold text-white shadow-soft hover:bg-leaf-700"
        >
          Try Now
        </Link>
      </div>
    </header>
  );
}
