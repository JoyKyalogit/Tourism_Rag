const BUDGET_OPTIONS = [
  { value: "", label: "Any budget" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "luxury", label: "Luxury" },
];

const TRAVEL_OPTIONS = [
  { value: "", label: "Any style" },
  { value: "solo", label: "Solo" },
  { value: "family", label: "Family" },
  { value: "adventure", label: "Adventure" },
];

const SEASON_OPTIONS = [
  { value: "", label: "Any season" },
  { value: "January", label: "January" },
  { value: "February", label: "February" },
  { value: "March", label: "March" },
  { value: "April", label: "April" },
  { value: "May", label: "May" },
  { value: "June", label: "June" },
  { value: "July", label: "July" },
  { value: "August", label: "August" },
  { value: "September", label: "September" },
  { value: "October", label: "October" },
  { value: "November", label: "November" },
  { value: "December", label: "December" },
];

function FilterSelect({ icon, label, value, onChange, options }) {
  return (
    <div className="flex min-w-0 flex-1 items-center gap-2 rounded-xl border border-emerald-100 bg-emerald-50/40 px-3 py-2.5">
      <span className="shrink-0 text-emerald-600">{icon}</span>
      <div className="min-w-0 flex-1">
        <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">{label}</p>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full cursor-pointer bg-transparent text-sm font-medium text-slate-800 focus:outline-none"
        >
          {options.map((opt) => (
            <option key={opt.value || "any"} value={opt.value} className="bg-white">
              {opt.label}
            </option>
          ))}
        </select>
      </div>
      <svg className="h-4 w-4 shrink-0 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </div>
  );
}

export default function Filters({ filters, onChange }) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row">
      <FilterSelect
        icon={<WalletIcon />}
        label="Budget"
        value={filters.budget}
        onChange={(v) => onChange({ ...filters, budget: v })}
        options={BUDGET_OPTIONS}
      />
      <FilterSelect
        icon={<UsersIcon />}
        label="Travel type"
        value={filters.travelType}
        onChange={(v) => onChange({ ...filters, travelType: v })}
        options={TRAVEL_OPTIONS}
      />
      <FilterSelect
        icon={<SunIcon />}
        label="Season"
        value={filters.season}
        onChange={(v) => onChange({ ...filters, season: v })}
        options={SEASON_OPTIONS}
      />
    </div>
  );
}

export function buildFilterPayload(filters) {
  return {
    ...(filters.budget ? { budget: filters.budget } : {}),
    ...(filters.travelType ? { travel_type: filters.travelType } : {}),
    ...(filters.season ? { season: filters.season } : {}),
  };
}

function WalletIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path d="M21 12V7H5a2 2 0 010-4h14v4M3 5v14a2 2 0 002 2h16v-5M3 5h18M16 12a2 2 0 100 4 2 2 0 000-4z" />
    </svg>
  );
}

function UsersIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 11a4 4 0 100-8 4 4 0 000 8zM23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );
}
