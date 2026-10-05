import { Link, useLocation, useNavigate } from "react-router-dom";

function CompassIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <circle cx="12" cy="12" r="9" />
      <path d="M16 8l-2.5 7.5L8 16l2.5-7.5L16 8z" fill="currentColor" stroke="none" opacity="0.8" />
    </svg>
  );
}

function MapIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l5.447 2.724A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
    </svg>
  );
}

function HeartIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z" />
    </svg>
  );
}

const NAV_ITEMS = [
  { id: "explore", label: "Explore", icon: CompassIcon, to: "/" },
  { id: "itineraries", label: "Itineraries", icon: MapIcon, to: "/?tab=itineraries" },
  { id: "favorites", label: "Favorites", icon: HeartIcon, to: "/" },
];

export default function Sidebar({ onPlanTrip }) {
  const location = useLocation();
  const navigate = useNavigate();
  const onHome = location.pathname === "/";
  const activeNav =
    new URLSearchParams(location.search).get("tab") === "itineraries" ? "itineraries" : "explore";

  return (
    <aside className="hidden w-52 shrink-0 flex-col border-r border-white/30 bg-white/55 px-4 py-6 text-slate-800 backdrop-blur-md lg:flex">
      <Link to="/" className="mb-8 flex items-center gap-2.5 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-700 font-display text-xs font-bold text-white">
          TA
        </div>
        <div>
          <p className="font-display text-xl font-semibold text-slate-900">Travel Africa</p>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-emerald-700">AI Travel</p>
        </div>
      </Link>

      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map(({ id, label, icon: Icon, to }) => {
          const active = onHome && activeNav === id;
          return (
            <Link
              key={id}
              to={to}
              className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                active
                  ? "bg-emerald-100/80 text-emerald-900"
                  : "text-slate-600 hover:bg-emerald-50 hover:text-emerald-900"
              }`}
            >
              <Icon className={`h-4 w-4 ${active ? "text-emerald-600" : ""}`} />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto pt-8">
        <button
          type="button"
          onClick={onPlanTrip ?? (() => navigate("/?tab=itineraries"))}
          className="btn-gradient w-full rounded-xl py-2.5 text-sm font-semibold text-white"
        >
          Plan My Trip
        </button>
      </div>
    </aside>
  );
}
