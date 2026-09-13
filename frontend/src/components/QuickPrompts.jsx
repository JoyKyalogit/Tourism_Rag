const PROMPTS = [
  "Best beaches in East Africa",
  "Top safari parks in Tanzania",
  "Find hotels in Nairobi",
  "Plan a Zanzibar trip",
  "Luxury lodges in Maasai Mara",
  "Compare Serengeti and Mara",
];

export default function QuickPrompts({ onSelect, disabled, activePrompt }) {
  return (
    <div className="flex flex-wrap gap-2">
      {PROMPTS.map((prompt) => {
        const active = activePrompt === prompt;
        return (
          <button
            key={prompt}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(prompt)}
            className={`rounded-full px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
              active
                ? "bg-emerald-600 text-white shadow-md shadow-emerald-600/20"
                : "border border-emerald-100 bg-white text-slate-600 hover:border-emerald-200 hover:bg-emerald-50 hover:text-emerald-800"
            }`}
          >
            {prompt}
          </button>
        );
      })}
    </div>
  );
}
