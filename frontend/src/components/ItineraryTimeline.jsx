export default function ItineraryTimeline({ itinerary }) {
  if (!itinerary?.length) return null;

  return (
    <section className="mb-8">
      <h2 className="mb-4 font-display text-base font-semibold text-slate-900">Your itinerary</h2>
      <ol className="space-y-4 border-l-2 border-emerald-200 pl-5">
        {itinerary.map((day, index) => (
          <li key={day.day}>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-700">
              {day.day}
            </p>
            <p className="font-medium text-slate-900">{day.title}</p>
            <p className="mt-1 text-sm leading-relaxed text-slate-600">{day.description}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
