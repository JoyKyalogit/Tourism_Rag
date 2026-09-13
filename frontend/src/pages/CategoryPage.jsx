import { Link, useParams } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import { BackLink } from "../components/BackLink";
import Sidebar from "../components/Sidebar";
import {
  COUNTRY_META,
  getCategoryBySlug,
  getDestinationsByCategory,
  groupDestinationsByCountry,
} from "../data/catalog";

const VIBE_BY_TYPE = {
  Beach: ["romantic", "sunset", "slow travel"],
  Safari: ["wildlife", "bucket-list", "photography"],
  City: ["culture", "food", "weekend"],
  Adventure: ["active", "hiking", "epic views"],
  Nature: ["calm", "eco", "scenic"],
};

const TYPE_BADGE = {
  Beach: "bg-sky-100 text-sky-800",
  Safari: "bg-amber-100 text-amber-900",
  City: "bg-violet-100 text-violet-800",
  Adventure: "bg-orange-100 text-orange-900",
  Nature: "bg-emerald-100 text-emerald-800",
};

function PlaceCard({ place }) {
  return (
    <Link
      to={`/place/${place.slug}`}
      className="group overflow-hidden rounded-3xl border border-emerald-100/80 bg-white/95 shadow-sm backdrop-blur-sm transition duration-300 hover:-translate-y-1 hover:shadow-md"
    >
      <div className="relative h-44 overflow-hidden">
        <img
          src={place.image}
          alt={place.name}
          className="h-full w-full object-cover transition duration-500 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-slate-900/50 via-transparent to-transparent" />
        <span className="absolute bottom-3 left-3 rounded-full bg-white/95 px-2.5 py-1 text-xs text-slate-700">
          {place.location}
        </span>
        <span
          className={`absolute right-3 top-3 rounded-full px-2.5 py-1 text-[11px] font-semibold ${TYPE_BADGE[place.type] || "bg-emerald-100 text-emerald-800"}`}
        >
          {VIBE_BY_TYPE[place.type]?.[0] || "local favorite"}
        </span>
      </div>
      <div className="p-4">
        <h2 className="font-display text-lg font-semibold text-slate-900">{place.name}</h2>
        <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-slate-600">{place.description}</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {(VIBE_BY_TYPE[place.type] || ["curated"]).slice(0, 2).map((tag) => (
            <span
              key={`${place.slug}-${tag}`}
              className="rounded-full border border-emerald-100 bg-emerald-50 px-2 py-0.5 text-[11px] text-emerald-800"
            >
              {tag}
            </span>
          ))}
        </div>
        <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-emerald-700 group-hover:underline">
          View photos &amp; details →
        </p>
      </div>
    </Link>
  );
}

export default function CategoryPage() {
  const { categorySlug } = useParams();
  const category = getCategoryBySlug(categorySlug);
  const places = getDestinationsByCategory(categorySlug);
  const byCountry = groupDestinationsByCountry(places);

  if (!category) {
    return (
      <AppLayout sidebar={<Sidebar />}>
        <main className="flex flex-1 items-center justify-center p-8 text-center">
          <div>
            <p className="text-slate-700">Category not found.</p>
            <Link to="/" className="mt-4 inline-block text-emerald-600 hover:underline">
              Go home
            </Link>
          </div>
        </main>
      </AppLayout>
    );
  }

  return (
    <AppLayout sidebar={<Sidebar />}>
      <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
          <BackLink to="/">Back to home</BackLink>

          <header className="page-content-scrim mb-8 rounded-2xl p-6 sm:p-8">
            <h1 className="editorial-title font-display text-3xl font-semibold text-slate-900 sm:text-5xl">
              {category.label}
            </h1>
            <p className="mt-2 max-w-lg text-sm leading-relaxed text-slate-700 sm:text-base">
              {category.description}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {byCountry.map(({ country }) => {
                const meta = COUNTRY_META[country] || { label: country };
                return (
                  <span
                    key={country}
                    className="rounded-full border border-white/60 bg-white/80 px-3 py-1 text-xs font-semibold text-emerald-900 backdrop-blur-sm"
                  >
                    {meta.label}
                  </span>
                );
              })}
            </div>
            <p className="mt-3 text-xs font-semibold uppercase tracking-[0.14em] text-emerald-900">
              {places.length} destinations · {byCountry.length} countries
            </p>
          </header>

          {byCountry.map(({ country, places: countryPlaces }) => {
            const meta = COUNTRY_META[country] || { label: country };
            return (
              <section key={country} className="mb-10">
                <div className="mb-5 flex items-center gap-3 border-b border-white/40 pb-3">
                  <div>
                    <h2 className="font-display text-xl font-semibold text-slate-900">{meta.label}</h2>
                    <p className="text-xs text-slate-500">
                      {countryPlaces.length} {category.label.toLowerCase()} destination
                      {countryPlaces.length === 1 ? "" : "s"}
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
                  {countryPlaces.map((place) => (
                    <PlaceCard key={place.slug} place={place} />
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </main>
    </AppLayout>
  );
}
