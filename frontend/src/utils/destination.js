const TYPE_OVERRIDES = {
  mombasa: "City",
  "dar es salaam": "City",
  nairobi: "City",
  kampala: "City",
  kigali: "City",
  arusha: "City",
  "taita hills": "Safari",
  tsavo: "Safari",
  "maasai mara": "Safari",
  amboseli: "Safari",
  samburu: "Safari",
  "nairobi national park": "Safari",
  serengeti: "Safari",
  "ngorongoro crater": "Safari",
  "tarangire national park": "Safari",
  "diani beach": "Beach",
  watamu: "Beach",
  malindi: "Beach",
  lamu: "Beach",
  zanzibar: "Beach",
  "pemba island": "Beach",
};

const BEACH_KEYWORDS = [
  "beach", "beaches", "diani", "watamu", "lamu", "malindi", "marine", "snorkel",
  "zanzibar", "pemba", "nungwi", "indian ocean",
];
const SAFARI_KEYWORDS = [
  "mara", "safari", "amboseli", "tsavo", "samburu", "park", "wildlife", "game",
  "serengeti", "ngorongoro", "tarangire", "queen elizabeth", "migration", "crater",
  "taita", "salt lick",
];
const CITY_KEYWORDS = [
  "nairobi", "kisumu", "eldoret", "nyeri", "machakos", "thika", "embu",
  "kitale", "kericho", "narok", "meru town", "city", "museum", "fort jesus",
  "kampala", "dar es salaam", "arusha", "kigali", "entebbe", "mombasa",
];

export function getCategory(destination) {
  const name = (destination.name || "").toLowerCase().trim();
  if (TYPE_OVERRIDES[name]) return TYPE_OVERRIDES[name];

  const text = `${destination.name} ${destination.description} ${destination.activities || ""}`.toLowerCase();
  // Safari before beach so “access from the coast” safari places don’t become Beach.
  if (SAFARI_KEYWORDS.some((k) => text.includes(k))) return "Safari";
  if (BEACH_KEYWORDS.some((k) => text.includes(k))) return "Beach";
  if (CITY_KEYWORDS.some((k) => text.includes(k))) return "City";
  if (text.includes("hike") || text.includes("mountain") || text.includes("adventure") || text.includes("trek")) {
    return "Adventure";
  }
  return "Nature";
}

export function getRating(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return (4.5 + (Math.abs(hash) % 5) / 10).toFixed(1);
}

export function getPriceKes(destination, budget = "") {
  const base = {
    low: 8500,
    medium: 18500,
    luxury: 45000,
  };
  let hash = 0;
  for (let i = 0; i < destination.name.length; i += 1) {
    hash = destination.name.charCodeAt(i) + ((hash << 5) - hash);
  }
  const tier = budget || "medium";
  const amount = (base[tier] || base.medium) + (Math.abs(hash) % 8000);
  return amount.toLocaleString();
}

export const FEATURED_DESTINATIONS = [
  {
    name: "Diani Beach",
    description: "White sand beaches, warm Indian Ocean waters, and world-class water sports along Kenya's south coast.",
    location: "Kwale County",
    activities: "Beach relaxation, Snorkeling, Kite surfing",
    lat: -4.3224,
    lng: 39.5796,
    image: "https://upload.wikimedia.org/wikipedia/commons/c/c7/Diani_Beach_Sunrise_Kenya.jpg",
  },
  {
    name: "Maasai Mara",
    description: "Kenya's premier safari destination — big cats, the Great Migration, and unforgettable game drives.",
    location: "Narok County",
    activities: "Game drives, Hot air balloon safari",
    lat: -1.4061,
    lng: 35.0199,
    image: "https://upload.wikimedia.org/wikipedia/commons/5/57/Maasai_Mara_National_Reserve_Kenya.jpg",
  },
  {
    name: "Nairobi",
    description: "A vibrant capital blending wildlife parks, culture, museums, and modern city hospitality.",
    location: "Nairobi County",
    activities: "Nairobi National Park, Giraffe Centre",
    lat: -1.2921,
    lng: 36.8219,
    image: "https://upload.wikimedia.org/wikipedia/commons/8/83/Nairobi_skyline_P1000019.jpg",
  },
  {
    name: "Watamu",
    description: "Marine adventures, coral reefs, and laid-back beach holidays on the north coast.",
    location: "Kilifi County",
    activities: "Dolphin watching, Snorkeling, Deep sea fishing",
    lat: -3.3614,
    lng: 40.0128,
    image: "https://upload.wikimedia.org/wikipedia/commons/b/b9/Watamu_Beach%2C_Kenya_01.jpg",
  },
];
