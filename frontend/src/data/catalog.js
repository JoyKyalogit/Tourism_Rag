import tourismData from "../../../data/processed/kenya_tourism_data.json";
import { EA_DESTINATIONS } from "./eaDestinations";
import { getCategory } from "../utils/destination";
import { lookupDestinationImages } from "./destinationImages";
import { normalizeHotelList } from "./hotelImages";

const COORDS = {
  "maasai mara": { lat: -1.4061, lng: 35.0199 },
  amboseli: { lat: -2.6527, lng: 37.2606 },
  "diani beach": { lat: -4.3224, lng: 39.5796 },
  watamu: { lat: -3.3614, lng: 40.0128 },
  naivasha: { lat: -0.7167, lng: 36.4333 },
  nairobi: { lat: -1.2921, lng: 36.8219 },
  tsavo: { lat: -3.3704, lng: 38.2018 },
  "nanyuki and mount kenya": { lat: -0.02, lng: 37.074 },
  mombasa: { lat: -4.0435, lng: 39.6682 },
  lamu: { lat: -2.2717, lng: 40.902 },
  malindi: { lat: -3.2175, lng: 40.1191 },
  samburu: { lat: 0.5301, lng: 37.262 },
  "meru national park": { lat: 0.05, lng: 37.85 },
  nakuru: { lat: -0.3031, lng: 36.08 },
  aberdare: { lat: -0.4, lng: 36.7 },
  kisumu: { lat: -0.0917, lng: 34.768 },
  "kakamega forest": { lat: 0.2827, lng: 34.7519 },
  "lake turkana": { lat: 3.5, lng: 36.0 },
  marsabit: { lat: 2.3284, lng: 37.9899 },
  "isiolo and shaba": { lat: 0.3545, lng: 37.5822 },
  "chyulu hills": { lat: -2.6833, lng: 37.8833 },
  "taita hills": { lat: -3.4, lng: 38.3 },
  "nairobi national park": { lat: -1.3733, lng: 36.8589 },
  "ol pejeta conservancy": { lat: 0.0, lng: 37.0 },
  "mount elgon": { lat: 1.1333, lng: 34.55 },
  "hells gate": { lat: -0.9167, lng: 36.3167 },
  eldoret: { lat: -0.5143, lng: 35.2698 },
  nyeri: { lat: -0.4201, lng: 36.9476 },
  machakos: { lat: -1.5177, lng: 37.2634 },
  thika: { lat: -1.0333, lng: 37.0693 },
  embu: { lat: -0.5317, lng: 37.4506 },
  kitale: { lat: 1.0167, lng: 35.0 },
  kericho: { lat: -0.367, lng: 35.2831 },
  narok: { lat: -1.0789, lng: 35.8669 },
  "meru town": { lat: 0.0469, lng: 37.6559 },
};

export const CATEGORIES = [
  {
    slug: "beaches",
    type: "Beach",
    label: "Beaches",
    emoji: "🏖️",
    description: "White sand, coral reefs & Indian Ocean escapes across Kenya, Zanzibar & the East Africa coast",
    cover: "https://upload.wikimedia.org/wikipedia/commons/c/c7/Diani_Beach_Sunrise_Kenya.jpg",
  },
  {
    slug: "safaris",
    type: "Safari",
    label: "Safaris",
    emoji: "🦁",
    description: "Wildlife parks & game drives from the Mara and Amboseli to Serengeti, Ngorongoro & beyond",
    cover: "https://upload.wikimedia.org/wikipedia/commons/5/57/Maasai_Mara_National_Reserve_Kenya.jpg",
  },
  {
    slug: "cities",
    type: "City",
    label: "Cities",
    emoji: "🏙️",
    description: "Urban culture & city breaks in Nairobi, Kampala, Dar es Salaam, Arusha & Kigali",
    cover: "https://upload.wikimedia.org/wikipedia/commons/8/83/Nairobi_skyline_P1000019.jpg",
  },
  {
    slug: "adventure",
    type: "Adventure",
    label: "Adventure",
    emoji: "⛰️",
    description: "Hiking, trekking & outdoor thrills — Kilimanjaro, Mount Kenya, gorilla trails & more",
    cover: "https://upload.wikimedia.org/wikipedia/commons/a/a4/View_of_Mt._Kenya_from_Nanyuki_Municipality.jpg",
  },
  {
    slug: "nature",
    type: "Nature",
    label: "Nature",
    emoji: "🌿",
    description: "Forests, lakes, highlands & eco-tourism across Kenya, Uganda, Tanzania & Rwanda",
    cover: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Kakamega_Forest_%282293443551%29.jpg",
  },
];

export function toSlug(name) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function lookupMeta(name) {
  const key = name.toLowerCase().trim();
  const coords = COORDS[key] || { lat: -0.5, lng: 37.0 };
  const photos = lookupDestinationImages(name);
  return { ...coords, ...photos };
}

function rowToDestination(row) {
  const name = row.destination;
  const meta = lookupMeta(name);
  const type =
    row.type ||
    getCategory({
      name,
      description: row.summary,
      activities: (row.activities || []).join(", "),
    });

  return {
    slug: toSlug(name),
    name,
    location: row.region,
    country: row.country || "Kenya",
    description: row.summary,
    type,
    activities: row.activities || [],
    bestMonths: row.best_months || [],
    bestTime: row.best_time || "",
    hotels: normalizeHotelList(row.hotels || {}, name, type, meta.image),
    sourceUrl: row.source_url || "",
    lat: row.lat ?? meta.lat,
    lng: row.lng ?? meta.lng,
    image: row.image || meta.image,
    gallery: meta.gallery?.length ? meta.gallery : [row.image || meta.image].filter(Boolean),
  };
}

const KENYA_DESTINATIONS = tourismData.map(rowToDestination);
const EA_ONLY = EA_DESTINATIONS.map(rowToDestination);

export const ALL_DESTINATIONS = [...KENYA_DESTINATIONS, ...EA_ONLY];

export const DESTINATIONS_BY_SLUG = Object.fromEntries(
  ALL_DESTINATIONS.map((d) => [d.slug, d])
);

export function getCategoryBySlug(slug) {
  return CATEGORIES.find((c) => c.slug === slug);
}

export function getDestinationsByCategory(slug) {
  const cat = getCategoryBySlug(slug);
  if (!cat) return [];
  return ALL_DESTINATIONS.filter((d) => d.type === cat.type);
}

export const COUNTRY_ORDER = ["Kenya", "Tanzania", "Uganda", "Rwanda", "East Africa"];

export const COUNTRY_META = {
  Kenya: { flag: "🇰🇪", label: "Kenya" },
  Tanzania: { flag: "🇹🇿", label: "Tanzania" },
  Uganda: { flag: "🇺🇬", label: "Uganda" },
  Rwanda: { flag: "🇷🇼", label: "Rwanda" },
  "East Africa": { flag: "🌍", label: "Cross-border" },
};

function normalizeCountry(country) {
  const value = (country || "Kenya").trim();
  if (value === "Kenya") return "Kenya";
  if (value.includes("Tanzania")) return "Tanzania";
  if (value.includes("Uganda")) return "Uganda";
  if (value.includes("Rwanda")) return "Rwanda";
  if (value.includes("/")) return "East Africa";
  return value;
}

/** Group category destinations by country for browse pages. */
export function groupDestinationsByCountry(places) {
  const buckets = new Map();
  for (const place of places) {
    const country = normalizeCountry(place.country);
    if (!buckets.has(country)) buckets.set(country, []);
    buckets.get(country).push(place);
  }

  const ordered = [];
  for (const country of COUNTRY_ORDER) {
    if (buckets.has(country)) {
      ordered.push({ country, places: buckets.get(country) });
      buckets.delete(country);
    }
  }
  for (const [country, countryPlaces] of buckets) {
    ordered.push({ country, places: countryPlaces });
  }
  return ordered;
}

export function getDestinationBySlug(slug) {
  return DESTINATIONS_BY_SLUG[slug] || null;
}

const FEATURED_SLUGS = ["diani-beach", "maasai-mara", "zanzibar", "serengeti"];

export function getFeaturedDestinations(limit = 6) {
  const picked = FEATURED_SLUGS.map((slug) => DESTINATIONS_BY_SLUG[slug]).filter(Boolean);
  const rest = ALL_DESTINATIONS.filter((d) => !FEATURED_SLUGS.includes(d.slug));
  return [...picked, ...rest].slice(0, limit).map((d) => ({
    name: d.name,
    slug: d.slug,
    location: d.location,
    country: d.country,
    description: d.description,
    type: d.type,
    lat: d.lat,
    lng: d.lng,
    image: d.image,
    activities: d.activities.join(", "),
  }));
}
