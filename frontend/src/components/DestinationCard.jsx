import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { getCategory, getPriceKes, getRating } from "../utils/destination";
import { googleMapsUrl } from "../utils/parseResponse";
import { toSlug } from "../data/catalog";

const CATEGORY_STYLES = {
  Beach: "bg-sky-500/90 text-white shadow-sky-500/30",
  Safari: "bg-amber-500/90 text-white shadow-amber-500/30",
  City: "bg-violet-500/90 text-white shadow-violet-500/30",
  Adventure: "bg-orange-500/90 text-white shadow-orange-500/30",
  Nature: "bg-emerald-500/90 text-white shadow-emerald-500/30",
};

export default function DestinationCard({
  destination,
  budget,
  isSelected,
  isHovered,
  onExplore,
  onHover,
  onFavorite,
}) {
  const [fav, setFav] = useState(false);
  const category = destination.type || getCategory(destination);
  const rating = getRating(destination.name);
  const price = getPriceKes(destination, budget);
  const active = isSelected || isHovered;
  const slug = destination.slug || toSlug(destination.name);

  return (
    <article
      onMouseEnter={() => onHover?.(destination)}
      onMouseLeave={() => onHover?.(null)}
      className={`group overflow-hidden rounded-2xl border bg-white/5 shadow-lg backdrop-blur-md transition duration-300 hover:scale-[1.02] hover:shadow-xl hover:shadow-brand-500/10 ${
        active
          ? "border-brand-400/50 ring-2 ring-brand-400/30 shadow-brand-500/20"
          : "border-white/10 hover:border-white/20"
      }`}
    >
      <div className="relative h-48 overflow-hidden">
        <img
          src={destination.image}
          alt={destination.name}
          className="h-full w-full object-cover transition duration-500 group-hover:scale-110"
          loading="lazy"
          onError={(e) => {
            e.currentTarget.src =
              "https://upload.wikimedia.org/wikipedia/commons/8/83/Nairobi_skyline_P1000019.jpg";
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />

        <span
          className={`absolute left-3 top-3 rounded-lg px-2.5 py-1 text-[11px] font-bold uppercase tracking-wide shadow-lg ${CATEGORY_STYLES[category] || CATEGORY_STYLES.Nature}`}
        >
          {category}
        </span>

        <button
          type="button"
          onClick={() => {
            setFav(!fav);
            onFavorite?.(destination, !fav);
          }}
          className="absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-black/40 text-white backdrop-blur-md transition hover:scale-110 hover:bg-black/60"
          aria-label="Favorite"
        >
          <svg
            className="h-4 w-4"
            fill={fav ? "currentColor" : "none"}
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z" />
          </svg>
        </button>
      </div>

      <div className="p-5">
        <h3 className="font-display text-lg font-bold text-white">{destination.name}</h3>

        <p className="mt-1 flex items-center gap-1.5 text-xs text-slate-400">
          <svg className="h-3.5 w-3.5 shrink-0 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a2 2 0 01-2.828 0l-4.243-4.243a8 8 0 1111.314 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          {destination.location}
        </p>

        <p className="mt-3 line-clamp-2 text-sm leading-relaxed text-slate-300">
          {destination.description}
        </p>

        <div className="mt-4 flex items-center justify-between border-t border-white/10 pt-4">
          <div className="flex items-center gap-1 text-sm">
            <span className="text-amber-400">★</span>
            <span className="font-semibold text-white">{rating}</span>
          </div>
          <p className="text-sm">
            <span className="text-slate-500">From </span>
            <span className="font-bold text-brand-400">KES {price}</span>
          </p>
        </div>

        <div className="mt-4 flex gap-2">
          <Link
            to={`/place/${slug}`}
            onClick={() => onExplore?.(destination)}
            className="btn-gradient flex flex-1 items-center justify-center gap-1.5 rounded-xl py-2.5 text-sm font-semibold text-white"
          >
            View details
          </Link>
          <a
            href={googleMapsUrl(destination.name)}
            target="_blank"
            rel="noreferrer"
            className="flex flex-1 items-center justify-center rounded-xl border border-white/15 bg-white/5 py-2.5 text-sm font-medium text-slate-300 transition hover:border-white/25 hover:bg-white/10 hover:text-white"
          >
            Maps
          </a>
        </div>
      </div>
    </article>
  );
}
