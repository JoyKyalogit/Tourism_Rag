import DestinationCard from "./DestinationCard";

export default function ResultsGrid({
  destinations,
  budget,
  selectedName,
  hoveredName,
  onSelect,
  onHover,
  onFavorite,
  emptyMessage = "No destinations found yet. Try a search above.",
}) {
  if (!destinations?.length) {
    return (
      <div className="glass-panel rounded-2xl border border-dashed border-white/15 px-6 py-12 text-center">
        <p className="text-4xl">🗺️</p>
        <p className="mt-3 text-sm text-slate-400">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-2">
      {destinations.map((dest) => (
        <DestinationCard
          key={dest.name}
          destination={dest}
          budget={budget}
          isSelected={selectedName === dest.name}
          isHovered={hoveredName === dest.name}
          onExplore={onSelect}
          onHover={onHover}
          onFavorite={onFavorite}
        />
      ))}
    </div>
  );
}
