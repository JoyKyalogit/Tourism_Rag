import { Link, useLocation, useParams } from "react-router-dom";
import { CATEGORIES, getDestinationBySlug } from "../data/catalog";

function LogoMark({ className = "h-8 w-8" }) {
  return (
    <div
      className={`flex shrink-0 items-center justify-center rounded-lg bg-emerald-700 font-display text-xs font-bold text-white ${className}`}
    >
      TA
    </div>
  );
}

function navClass(active) {
  return `shrink-0 rounded-full px-3.5 py-2 text-sm font-medium transition ${
    active
      ? "bg-white/90 text-emerald-900 shadow-sm"
      : "text-slate-800 hover:bg-white/70 hover:text-emerald-900"
  }`;
}

export default function TopNav() {
  const location = useLocation();
  const { placeSlug } = useParams();
  const path = location.pathname;
  const place = placeSlug ? getDestinationBySlug(placeSlug) : null;
  const placeCategorySlug = place
    ? CATEGORIES.find((c) => c.type === place.type)?.slug
    : null;

  return (
    <header className="sticky top-0 z-30 shrink-0 border-b border-white/30 bg-white/55 backdrop-blur-md">
      <div className="flex h-14 items-center gap-3 px-4 lg:px-6">
        <Link to="/" className="flex shrink-0 items-center gap-2">
          <LogoMark />
          <span className="hidden font-display text-sm font-semibold tracking-wide text-slate-900 sm:inline">
            Travel Africa
          </span>
        </Link>

        <nav
          className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto scrollbar-none"
          aria-label="Main"
        >
          <Link to="/" className={navClass(path === "/")}>
            Home
          </Link>
          {CATEGORIES.map((cat) => {
            const href = `/explore/${cat.slug}`;
            const active =
              path === href || path.startsWith(`${href}/`) || placeCategorySlug === cat.slug;
            return (
              <Link key={cat.slug} to={href} className={navClass(active)}>
                {cat.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
