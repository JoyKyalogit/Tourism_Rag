import { Link, useParams } from "react-router-dom";

import AppLayout from "../components/AppLayout";
import { BackLink } from "../components/BackLink";

import Sidebar from "../components/Sidebar";

import HotelCard from "../components/HotelCard";

import { getDestinationBySlug, CATEGORIES } from "../data/catalog";

import { googleMapsUrl } from "../utils/parseResponse";
import { handleImageError, THUMB_SIZES, thumbUrl } from "../utils/imageUrl";



export default function DestinationPage() {

  const { placeSlug } = useParams();

  const place = getDestinationBySlug(placeSlug);



  if (!place) {

    return (

      <AppLayout sidebar={<Sidebar />}>

        <main className="flex flex-1 items-center justify-center p-8">

          <div className="text-center">

            <p className="text-slate-700">Destination not found.</p>

            <Link to="/" className="mt-4 inline-block text-emerald-600 hover:underline">

              Go home

            </Link>

          </div>

        </main>

      </AppLayout>

    );

  }



  const category = CATEGORIES.find((c) => c.type === place.type);

  const categoryLink = category ? `/explore/${category.slug}` : "/";



  return (

    <AppLayout sidebar={<Sidebar />}>

      <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">

        <div className="mx-auto w-full max-w-4xl flex-1 px-4 py-8 sm:px-6">

          <BackLink to={categoryLink}>Back to {place.type}s</BackLink>



          <div className="relative mb-6 overflow-hidden rounded-2xl border border-emerald-100 shadow-sm">

            <img
              src={thumbUrl(place.image, THUMB_SIZES.hero)}
              alt={place.name}
              decoding="async"
              onError={handleImageError(place.image)}
              className="h-56 w-full object-cover sm:h-72"
            />

            <div className="absolute inset-0 bg-gradient-to-t from-slate-900/70 via-transparent to-transparent" />

            <div className="absolute bottom-0 left-0 p-6">

              <span className="rounded-lg bg-emerald-600 px-2.5 py-1 text-xs font-bold text-white">

                {place.type}

              </span>

              <h1 className="mt-3 font-display text-3xl font-bold text-white sm:text-4xl">{place.name}</h1>

              <p className="mt-1 text-sm text-white/90">
                {place.location}
                {place.country ? ` · ${place.country}` : ""}
              </p>

            </div>

          </div>



          <section className="glass-panel mb-6 p-5">

            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">

              <h2 className="font-display text-lg font-bold text-slate-900">Photo gallery</h2>

              <p className="text-xs text-slate-500">

                Photos via{" "}

                <a

                  href="https://commons.wikimedia.org"

                  target="_blank"

                  rel="noreferrer"

                  className="text-emerald-600 hover:underline"

                >

                  Wikimedia Commons

                </a>

              </p>

            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">

              {place.gallery.map((src, i) => (

                <a

                  key={src}

                  href={src}

                  target="_blank"

                  rel="noreferrer"

                  className="group overflow-hidden rounded-xl border border-emerald-100"

                >

                  <img

                    src={thumbUrl(src, THUMB_SIZES.small)}

                    loading="lazy"

                    decoding="async"

                    onError={handleImageError(src)}

                    alt={`${place.name} ${i + 1}`}

                    className="aspect-[4/3] w-full object-cover transition duration-300 group-hover:scale-105"

                  />

                </a>

              ))}

            </div>

          </section>



          <div className="grid gap-6 md:grid-cols-2">

            <section className="glass-panel p-5">

              <h2 className="font-display text-lg font-bold text-slate-900">About</h2>

              <p className="mt-3 text-sm leading-relaxed text-slate-600">{place.description}</p>

              {place.bestTime && (

                <p className="mt-4 text-sm text-slate-500">

                  <span className="font-medium text-slate-800">Best time: </span>

                  {place.bestTime}

                </p>

              )}

              {place.bestMonths?.length > 0 && (

                <p className="mt-2 text-sm text-slate-500">

                  <span className="font-medium text-slate-800">Best months: </span>

                  {place.bestMonths.join(", ")}

                </p>

              )}

            </section>



            <section className="glass-panel p-5">

              <h2 className="font-display text-lg font-bold text-slate-900">Activities</h2>

              <ul className="mt-3 space-y-2">

                {place.activities.map((act) => (

                  <li key={act} className="flex items-center gap-2 text-sm text-slate-600">

                    <span className="text-emerald-600">✓</span> {act}

                  </li>

                ))}

              </ul>

            </section>

          </div>



          {place.hotels.all?.length > 0 && (

            <section className="glass-panel mt-6 p-5">

              <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">

                <div>

                  <h2 className="font-display text-lg font-bold text-slate-900">Where to stay</h2>

                  <p className="mt-1 text-sm text-slate-500">

                    {place.hotels.all.length} hand-picked hotels &amp; lodges — budget to luxury

                  </p>

                </div>

              </div>



              {["luxury", "midrange", "budget"].map((tier) =>

                place.hotels[tier]?.length ? (

                  <div key={tier} className="mb-6 last:mb-0">

                    <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-emerald-700">

                      {tier === "midrange" ? "Mid-range" : tier}

                    </h3>

                    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">

                      {place.hotels[tier].map((hotel) => (

                        <HotelCard key={`${tier}-${hotel.name}`} hotel={hotel} />

                      ))}

                    </div>

                  </div>

                ) : null

              )}

            </section>

          )}



          <div className="mt-8 flex flex-wrap gap-3">

            <a

              href={googleMapsUrl(place.name)}

              target="_blank"

              rel="noreferrer"

              className="btn-gradient rounded-xl px-5 py-2.5 text-sm font-semibold text-white"

            >

              Open in Google Maps

            </a>

          </div>

        </div>

      </main>

    </AppLayout>

  );

}

