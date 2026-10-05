import { getCategory } from "./destination";
import { toSlug } from "../data/catalog";
import { lookupDestinationImages } from "../data/destinationImages";
import { lookupHotelImage } from "../data/hotelImages";

const PLACE_PATTERNS = [
  /(?:^|[:\n-]\s*)([A-Z][a-zA-Z\s']+(?:Beach|Mara|Park|Hills|Gate|Forest|Turkana|Nairobi|Mombasa|Lamu|Malindi|Watamu|Diani|Amboseli|Tsavo|Samburu|Naivasha|Nakuru|Kisumu|Nanyuki)[^,\n.]*)/g,
  /([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})/g,
];

export function getDestinationImage(destination) {
  if (destination.image) return destination.image;
  const name = destination.name || destination.destination || "";
  if (name) return lookupDestinationImages(name).image;
  return lookupDestinationImages("nairobi").image;
}

export function googleMapsUrl(name) {
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${name}, Kenya`)}`;
}

export function normalizeDestination(raw) {
  const name = raw.name || raw.destination || "Unknown";
  const normalized = {
    slug: raw.slug || toSlug(name),
    name,
    location: raw.location || raw.region || "Kenya",
    description: raw.description || raw.summary || "",
    activities: raw.activities || "",
    type: raw.type || getCategory({ name, description: raw.description || "", activities: raw.activities || "" }),
    lat: raw.lat ?? null,
    lng: raw.lng ?? null,
    source_url: raw.source_url || "",
    image: getDestinationImage({ ...raw, name }),
  };
  return normalized;
}

export function normalizeDestinations(list) {
  if (!Array.isArray(list)) return [];
  return list.map(normalizeDestination).filter((d) => d.name !== "Unknown");
}

/** Parse plain-text API answers into destination stubs when structured data is missing. */
export function parseDestinationsFromText(answer, existing = []) {
  if (existing.length > 0) return normalizeDestinations(existing);

  const found = new Set();
  const results = [];

  for (const pattern of PLACE_PATTERNS) {
    pattern.lastIndex = 0;
    let match;
    while ((match = pattern.exec(answer)) !== null) {
      const name = match[1]?.trim();
      if (!name || name.length < 4 || found.has(name.toLowerCase())) continue;
      if (/^(Top|Best|Hotel|Budget|Luxury|Midrange|Activities|Why|Recommended)/i.test(name)) continue;
      found.add(name.toLowerCase());
      results.push(
        normalizeDestination({
          name,
          description: `Recommended destination based on your search.`,
          location: "Kenya",
        })
      );
    }
  }

  return results.slice(0, 6);
}

export function processApiResponse(data, fallbackDestinations = []) {
  const structured = normalizeDestinations(data.destinations);
  const parsed =
    structured.length > 0
      ? structured
      : parseDestinationsFromText(data.answer || "", fallbackDestinations);

  const hotels = Array.isArray(data.hotels)
    ? data.hotels.map((h) => {
        const name = h.hotel_name || h.name || "Hotel";
        const location = h.location || "";
        const placePhotos = lookupDestinationImages(location);
        const tier = h.price_range || "midrange";
        const destinationType = getCategory({
          name: location,
          description: h.description || "",
          activities: "",
        });
        return {
          name,
          location,
          country: h.country || "",
          description: h.description || "",
          tier,
          sourceUrl: h.source_url || "",
          websiteUrl: h.website_url || "",
          image:
            h.image_url ||
            lookupHotelImage(name, destinationType, tier, placePhotos.image),
        };
      })
    : [];

  return {
    answer: data.answer || "",
    sources: data.sources || [],
    structuredSources: data.structured_sources || [],
    hotels,
    destinations: parsed.length > 0 ? parsed : normalizeDestinations(fallbackDestinations),
    itinerary: data.itinerary || [],
  };
}
