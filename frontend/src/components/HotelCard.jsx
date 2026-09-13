const TIER_STYLES = {

  budget: "bg-sky-100 text-sky-800 border-sky-200",

  midrange: "bg-amber-100 text-amber-900 border-amber-200",

  luxury: "bg-violet-100 text-violet-800 border-violet-200",

};



const TIER_LABELS = {

  budget: "Budget",

  midrange: "Mid-range",

  luxury: "Luxury",

};



export default function HotelCard({ hotel }) {

  const href = hotel.sourceUrl || hotel.websiteUrl || hotel.mapsUrl || "#";

  const linkLabel = hotel.sourceUrl ? "View source →" : hotel.websiteUrl ? "Visit website →" : "View on map →";



  return (

    <a

      href={href}

      target="_blank"

      rel="noreferrer"

      className="group flex flex-col overflow-hidden rounded-xl border border-emerald-100 bg-white transition hover:border-emerald-300 hover:shadow-md"

    >

      <div className="relative h-32 overflow-hidden bg-emerald-50">

        {hotel.image ? (

          <img

            src={hotel.image}

            alt={hotel.name}

            loading="lazy"

            className="h-full w-full object-cover transition duration-500 group-hover:scale-105"

          />

        ) : (

          <div className="flex h-full items-center justify-center text-3xl">🏨</div>

        )}

        <span

          className={`absolute left-2 top-2 rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${TIER_STYLES[hotel.tier] || TIER_STYLES.midrange}`}

        >

          {TIER_LABELS[hotel.tier] || hotel.tier}

        </span>

      </div>

      <div className="flex flex-1 flex-col p-3">

        <p className="font-medium text-sm text-slate-900 leading-snug group-hover:text-emerald-700">

          {hotel.name}

        </p>

        {(hotel.location || hotel.country) && (

          <p className="mt-1 text-xs text-slate-500">

            {[hotel.location, hotel.country].filter(Boolean).join(", ")}

          </p>

        )}

        <p className="mt-1 text-xs text-emerald-600">{linkLabel}</p>

      </div>

    </a>

  );

}

