export default function LoadingSpinner({ label = "Finding the best places for you..." }) {
  return (
    <div className="mb-6 flex items-center gap-3 text-sm text-slate-600">
      <div className="relative h-6 w-6 shrink-0">
        <div className="absolute inset-0 animate-spin rounded-full border-2 border-emerald-200 border-t-emerald-600" />
      </div>
      <span>{label}</span>
    </div>
  );
}
