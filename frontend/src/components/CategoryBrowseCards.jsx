import { Link } from "react-router-dom";
import { CATEGORIES, getDestinationsByCategory } from "../data/catalog";

export default function CategoryBrowseCards() {
  return (
    <section className="mb-8">
      <div className="mb-4 flex items-center gap-2">
        <span className="text-brand-400">🧭</span>
        <h2 className="font-display text-lg font-bold text-white">Browse by category</h2>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CATEGORIES.map((cat) => {
          const count = getDestinationsByCategory(cat.slug).length;
          return (
            <Link
              key={cat.slug}
              to={`/explore/${cat.slug}`}
              className="group relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 shadow-lg backdrop-blur-md transition duration-300 hover:scale-[1.02] hover:border-emerald-400/40 hover:shadow-emerald-500/10"
            >
              <div className="relative h-36 overflow-hidden">
                <img
                  src={cat.cover}
                  alt={cat.label}
                  className="h-full w-full object-cover transition duration-500 group-hover:scale-110"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent" />
                <span className="absolute left-3 top-3 text-2xl">{cat.emoji}</span>
                <span className="absolute bottom-3 right-3 rounded-full bg-black/50 px-2.5 py-0.5 text-xs text-white backdrop-blur-sm">
                  {count} places
                </span>
              </div>
              <div className="p-4">
                <h3 className="font-display text-base font-bold text-white">{cat.label}</h3>
                <p className="mt-1 line-clamp-2 text-xs text-slate-400">{cat.description}</p>
                <p className="mt-3 text-xs font-semibold text-emerald-400 group-hover:underline">
                  View all {cat.label.toLowerCase()} →
                </p>
              </div>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
