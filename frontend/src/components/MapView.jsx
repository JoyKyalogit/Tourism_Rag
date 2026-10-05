import { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
import { handleImageError, THUMB_SIZES, thumbUrl } from "../utils/imageUrl";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const KENYA_CENTER = [-0.5, 37.5];
const DEFAULT_ZOOM = 6;

function FlyToDestination({ destination }) {
  const map = useMap();
  const prevRef = useRef(null);

  useEffect(() => {
    if (!destination?.lat || !destination?.lng) return;
    if (prevRef.current === destination.name) return;
    prevRef.current = destination.name;

    let cancelled = false;
    map.whenReady(() => {
      if (cancelled || !map.getContainer()?.isConnected) return;
      try {
        map.flyTo([destination.lat, destination.lng], 9, { duration: 1 });
      } catch {
        // Map was unmounted during animation
      }
    });

    return () => {
      cancelled = true;
    };
  }, [destination, map]);

  return null;
}

function MapMarkers({ destinations, activeName, onSelectDestination }) {
  const markers = destinations.filter((d) => d.lat && d.lng);

  return markers.map((dest) => {
    const isActive = activeName === dest.name;
    return (
      <Marker
        key={dest.name}
        position={[dest.lat, dest.lng]}
        zIndexOffset={isActive ? 1000 : 0}
        opacity={activeName && !isActive ? 0.5 : 1}
      >
        <Popup>
          <div className="text-sm">
            <strong>{dest.name}</strong>
            <p className="mt-1 text-slate-600">{dest.location}</p>
            <button
              type="button"
              className="mt-2 text-xs text-emerald-600 underline"
              onClick={() => onSelectDestination?.(dest)}
            >
              Highlight on map
            </button>
          </div>
        </Popup>
      </Marker>
    );
  });
}

export default function MapView({
  destinations = [],
  selectedDestination,
  hoveredDestination,
  onSelectDestination,
  minHeight = 280,
}) {
  const [mounted, setMounted] = useState(false);
  const markers = destinations.filter((d) => d.lat && d.lng);
  const activeName = hoveredDestination?.name || selectedDestination?.name;
  const flyTarget = selectedDestination || null;

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  if (!mounted) {
    return (
      <div
        className="glass-panel flex items-center justify-center rounded-2xl text-sm text-slate-400"
        style={{ minHeight }}
      >
        Loading map…
      </div>
    );
  }

  return (
    <div className="glass-panel relative flex h-full flex-col overflow-hidden rounded-2xl shadow-xl">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
        <h3 className="text-sm font-semibold text-white">Explore Kenya</h3>
        <span className="rounded-full bg-emerald-500/20 px-2.5 py-0.5 text-xs font-medium text-emerald-300">
          {markers.length} places
        </span>
      </div>

      <div className="relative min-h-0 flex-1" style={{ minHeight }}>
        <MapContainer
          center={KENYA_CENTER}
          zoom={DEFAULT_ZOOM}
          scrollWheelZoom
          className="h-full w-full z-0"
          style={{ minHeight }}
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <FlyToDestination destination={flyTarget} />
          <MapMarkers
            destinations={destinations}
            activeName={activeName}
            onSelectDestination={onSelectDestination}
          />
        </MapContainer>

        {markers.length > 0 && (
          <div className="pointer-events-auto absolute bottom-4 left-4 right-4 z-[500] rounded-xl border border-white/10 bg-[#0b0f19]/90 p-3 shadow-xl backdrop-blur-lg">
            <p className="mb-2 text-xs font-medium text-slate-300">
              {markers.length} place{markers.length !== 1 ? "s" : ""} found
            </p>
            <div className="flex gap-2 overflow-x-auto pb-0.5">
              {markers.slice(0, 6).map((dest) => (
                <button
                  key={dest.name}
                  type="button"
                  onClick={() => onSelectDestination?.(dest)}
                  className={`shrink-0 overflow-hidden rounded-lg border-2 transition duration-200 ${
                    activeName === dest.name
                      ? "border-emerald-400 shadow-lg shadow-emerald-500/30"
                      : "border-transparent opacity-75 hover:opacity-100"
                  }`}
                >
                  <img
                    src={thumbUrl(dest.image, THUMB_SIZES.tiny)}
                    alt={dest.name}
                    loading="lazy"
                    decoding="async"
                    onError={handleImageError(dest.image)}
                    className="h-12 w-16 object-cover"
                  />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
