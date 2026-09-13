import DestinationCard from "./DestinationCard";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-3 text-sm text-white shadow-card">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start animate-slide-up">
      <div className="max-w-full space-y-4">
        <div className="max-w-[90%] rounded-2xl rounded-bl-md border border-white/10 bg-surface-800/90 px-4 py-3 text-sm leading-relaxed text-slate-200 shadow-card">
          {message.content}
        </div>

        {message.destinations?.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-2">
            {message.destinations.map((dest) => (
              <DestinationCard key={dest.name} destination={dest} />
            ))}
          </div>
        )}

        {message.itinerary?.length > 0 && (
          <div className="space-y-3">
            {message.itinerary.map((day) => (
              <div
                key={day.day}
                className="rounded-xl border border-white/10 bg-white/5 px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-brand-500/20 px-2.5 py-0.5 text-xs font-semibold text-brand-100">
                    {day.day}
                  </span>
                  <h4 className="font-semibold text-white">{day.title}</h4>
                </div>
                <p className="mt-2 text-sm text-slate-300">{day.description}</p>
              </div>
            ))}
          </div>
        )}

        {message.sources?.length > 0 && (
          <div className="text-xs text-slate-400">
            <span className="font-medium text-slate-300">Sources: </span>
            {message.sources.map((src, i) => (
              <span key={src}>
                <a
                  href={src}
                  target="_blank"
                  rel="noreferrer"
                  className="text-brand-100 hover:underline"
                >
                  link
                </a>
                {i < message.sources.length - 1 ? ", " : ""}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
