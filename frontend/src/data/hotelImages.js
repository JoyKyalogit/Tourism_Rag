/** Hotel photos from Wikimedia Commons where available; tier/type fallbacks otherwise. */

const TIER_FALLBACKS = {
  Beach: {
    budget: "https://upload.wikimedia.org/wikipedia/commons/9/93/Diani_beach.JPG",
    midrange: "https://upload.wikimedia.org/wikipedia/commons/b/b9/Watamu_Beach%2C_Kenya_01.jpg",
    luxury: "https://upload.wikimedia.org/wikipedia/commons/c/c5/Diani_Beach_overlooking_the_Indian_Ocean.jpg",
  },
  Safari: {
    budget: "https://upload.wikimedia.org/wikipedia/commons/d/d0/Reserve_samburu_faune_2.jpg",
    midrange: "https://upload.wikimedia.org/wikipedia/commons/5/57/Maasai_Mara_National_Reserve_Kenya.jpg",
    luxury: "https://upload.wikimedia.org/wikipedia/commons/f/fa/Elephants_at_Amboseli_national_park_against_Mount_Kilimanjaro.jpg",
  },
  City: {
    budget: "https://upload.wikimedia.org/wikipedia/commons/8/83/Nairobi_skyline_P1000019.jpg",
    midrange: "https://upload.wikimedia.org/wikipedia/commons/d/d3/Nairobi_skyline_P1000020.jpg",
    luxury: "https://upload.wikimedia.org/wikipedia/commons/a/ae/Nairobi_skyline_P1000021.jpg",
  },
  Adventure: {
    budget: "https://upload.wikimedia.org/wikipedia/commons/a/a4/View_of_Mt._Kenya_from_Nanyuki_Municipality.jpg",
    midrange: "https://upload.wikimedia.org/wikipedia/commons/7/7f/Hells_Gate_National_Park%2C_Kenya..JPG",
    luxury: "https://upload.wikimedia.org/wikipedia/commons/2/27/Chyulu_Hills_and_Shetani_Lava_Flow_in_Kenya_01.jpg",
  },
  Nature: {
    budget: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Kakamega_Forest_%282293443551%29.jpg",
    midrange: "https://upload.wikimedia.org/wikipedia/commons/2/2f/Lake_Naivasha_Kenya_04.jpg",
    luxury: "https://upload.wikimedia.org/wikipedia/commons/a/a5/Large_number_of_flamingos_at_Lake_Nakuru.jpg",
  },
};

/** Known hotel/lodge images on Wikimedia Commons. */
export const HOTEL_IMAGES = {
  "mara serena safari lodge":
    "https://upload.wikimedia.org/wikipedia/commons/4/40/Maasai_Mara_National_Reserve%2C_Kenya_%2853336745687%29.jpg",
  "jw marriott masai mara lodge":
    "https://upload.wikimedia.org/wikipedia/commons/5/57/Maasai_Mara_National_Reserve_Kenya.jpg",
  "governor's camp":
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/Maasai_Mara_National_Reserve%2C_Kenya_%2853337843878%29.jpg",
  "ol tukai lodge":
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Elephants_at_Amboseli_national_park_against_Mount_Kilimanjaro.jpg",
  "tortilis camp":
    "https://upload.wikimedia.org/wikipedia/commons/0/04/Elefante_africano_de_sabana_%28Loxodonta_africana%29%2C_parque_nacional_de_Amboseli%2C_Kenia%2C_2024-05-22%2C_DD_06.jpg",
  "amboseli serena safari lodge":
    "https://upload.wikimedia.org/wikipedia/commons/a/a6/Elefantes_africanos_de_sabana_%28Loxodonta_africana%29%2C_parque_nacional_de_Amboseli%2C_Kenia%2C_2024-05-22%2C_DD_86.jpg",
  "swahili beach resort":
    "https://upload.wikimedia.org/wikipedia/commons/c/c7/Diani_Beach_Sunrise_Kenya.jpg",
  "the sands at nomad":
    "https://upload.wikimedia.org/wikipedia/commons/c/c5/Diani_Beach_overlooking_the_Indian_Ocean.jpg",
  "almanara luxury resort":
    "https://upload.wikimedia.org/wikipedia/commons/9/93/Diani_beach.JPG",
  "pinewood beach resort and spa":
    "https://upload.wikimedia.org/wikipedia/commons/c/c5/Diani_Beach_overlooking_the_Indian_Ocean.jpg",
  "waterlovers beach resort":
    "https://upload.wikimedia.org/wikipedia/commons/c/c7/Diani_Beach_Sunrise_Kenya.jpg",
  "baobab beach resort":
    "https://upload.wikimedia.org/wikipedia/commons/9/93/Diani_beach.JPG",
  "papillon lagoon reef":
    "https://upload.wikimedia.org/wikipedia/commons/c/c5/Diani_Beach_overlooking_the_Indian_Ocean.jpg",
  "leopard beach resort":
    "https://upload.wikimedia.org/wikipedia/commons/9/93/Diani_beach.JPG",
  "neptune village beach resort":
    "https://upload.wikimedia.org/wikipedia/commons/c/c7/Diani_Beach_Sunrise_Kenya.jpg",
  "diani backpackers":
    "https://upload.wikimedia.org/wikipedia/commons/8/89/A_bicyle_delivery_guy_in_diani_kenya.jpg",
  "hemingways watamu":
    "https://upload.wikimedia.org/wikipedia/commons/4/4e/Watamu_Beach%2C_Kenya_03.jpg",
  "turtle bay beach club":
    "https://upload.wikimedia.org/wikipedia/commons/e/e9/Watamu_Beach%2C_Kenya_04.jpg",
  "lake naivasha sopa resort":
    "https://upload.wikimedia.org/wikipedia/commons/8/83/Lake_Naivasha_Kenya_05.jpg",
  "enashipai resort and spa":
    "https://upload.wikimedia.org/wikipedia/commons/c/c0/Lake_Naivasha_Kenya_08.jpg",
  "chui lodge":
    "https://upload.wikimedia.org/wikipedia/commons/3/36/Lake_Naivasha_Kenya_pelican.jpg",
  "villa rosa kempinski":
    "https://upload.wikimedia.org/wikipedia/commons/8/83/Nairobi_skyline_P1000019.jpg",
  "hemingways nairobi":
    "https://upload.wikimedia.org/wikipedia/commons/d/d2/Buffalo_Nairobi_Skyline_Savannah_Kenya_May19_R1600769.jpg",
  "fairmont the norfolk":
    "https://upload.wikimedia.org/wikipedia/commons/d/d3/Nairobi_skyline_P1000020.jpg",
  "sarova stanley":
    "https://upload.wikimedia.org/wikipedia/commons/a/ae/Nairobi_skyline_P1000021.jpg",
  "kilaguni serena safari lodge":
    "https://upload.wikimedia.org/wikipedia/commons/e/e5/Quelea_quelea_%28flock%29%2C_Tsavo_West_National_Park%2C_Kenya.jpg",
  "finch hattons luxury camp":
    "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tsavo_east_panorama.jpg",
  "fairmont mount kenya safari club":
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/Mount_Kenya_Sarfari_Club.jpg",
  "serena mountain lodge":
    "https://upload.wikimedia.org/wikipedia/commons/a/a9/Media_Capture_Mount_Kenya%27s_Scenery.jpg",
  "serena beach resort and spa":
    "https://upload.wikimedia.org/wikipedia/commons/e/e0/Nyali_Beach.JPG",
  "sarova whitesands beach resort":
    "https://upload.wikimedia.org/wikipedia/commons/d/da/Nyali_Beach_from_the_Reef_Hotel_during_high_tide_in_Mombasa%2C_Kenya_42.jpg",
  "voyager beach resort":
    "https://upload.wikimedia.org/wikipedia/commons/1/1d/Voyager_Beach_Resort_from_Nyali_Beach%2C_Mombasa%2C_Kenya_3.jpg",
  "peponi hotel":
    "https://upload.wikimedia.org/wikipedia/commons/8/80/Lamu_old_town_street.jpg",
  "the majlis resort":
    "https://upload.wikimedia.org/wikipedia/commons/9/9d/Usita_wa_Mui_street_Lamu_Old_Town.jpg",
  "lamu house hotel":
    "https://upload.wikimedia.org/wikipedia/commons/1/15/Lamu_old_street.jpg",
  "ocean beach resort and spa":
    "https://upload.wikimedia.org/wikipedia/commons/2/2d/Malindi_beach_Kenya.jpg",
  "samburu sopa lodge":
    "https://upload.wikimedia.org/wikipedia/commons/d/d0/Reserve_samburu_faune_2.jpg",
  "saruni samburu":
    "https://upload.wikimedia.org/wikipedia/commons/6/6a/Samburu_National_Reserve%2C_Kenya-26December2012.jpg",
  "elephant bedroom camp":
    "https://upload.wikimedia.org/wikipedia/commons/3/3e/Struthio_molybdophanes%2C_male%2C_Samburu_National_Reserve%2C_Kenya.jpg",
  "elsa's kopje":
    "https://upload.wikimedia.org/wikipedia/commons/c/c4/Meru_National_Park%2C_Kenya.jpg",
  "lake nakuru sopa lodge":
    "https://upload.wikimedia.org/wikipedia/commons/a/a5/Large_number_of_flamingos_at_Lake_Nakuru.jpg",
  "sarova lion hill game lodge":
    "https://upload.wikimedia.org/wikipedia/commons/b/b7/Flamingos_in_Lake_Nakuru_1.jpg",
  "the cliff nakuru":
    "https://upload.wikimedia.org/wikipedia/commons/7/70/Lake_Nakuru_National_Park_02_-_Lesser_Flamingo_%28Phoeniconaias_minor%29.jpg",
  "lake nakuru lodge":
    "https://upload.wikimedia.org/wikipedia/commons/9/93/Sarova_Lion_Hill_Lodge.jpg",
  "camp carnelley's":
    "https://upload.wikimedia.org/wikipedia/commons/9/91/Camp_Carnelley%27s_%28181090713%29.jpg",
  "treetops lodge":
    "https://upload.wikimedia.org/wikipedia/commons/1/1e/Treetops.jpg",
  "the ark lodge":
    "https://upload.wikimedia.org/wikipedia/commons/d/d6/Aberdare_National_Park%2C_Kenya_%2830714805264%29.jpg",
  "aberdare country club":
    "https://upload.wikimedia.org/wikipedia/commons/4/48/Aberdare_National_Park%2C_Kenya_%2830747173493%29.jpg",
  "ciala resort":
    "https://upload.wikimedia.org/wikipedia/commons/0/0a/Lake_Victoria_as_visible_from_Kisumu_City.jpg",
  "rondo retreat centre":
    "https://upload.wikimedia.org/wikipedia/commons/e/e2/Rondo_Retreat%2C_Kakamega_Rain_Forest%2C_Kenya_-_panoramio.jpg",
  "kakamega forest lodge":
    "https://upload.wikimedia.org/wikipedia/commons/a/a8/Kakamega_Forest_%282293443551%29.jpg",
  "desert rose lodge":
    "https://upload.wikimedia.org/wikipedia/commons/9/9f/Lake_Turkana_in_Kenya_02.jpg",
  "sarova shaba game lodge":
    "https://upload.wikimedia.org/wikipedia/commons/4/4b/Shaba_reserve_Kenya_mountains.jpg",
  "shaba game lodge":
    "https://upload.wikimedia.org/wikipedia/commons/4/42/Shaba_Kenya_river.jpg",
  "campi ya kanzi":
    "https://upload.wikimedia.org/wikipedia/commons/2/27/Chyulu_Hills_and_Shetani_Lava_Flow_in_Kenya_01.jpg",
  "ol donyo lodge":
    "https://upload.wikimedia.org/wikipedia/commons/5/5d/Cattle_%40_Chyulu_Hills_%2820513653483%29.jpg",
  "salt lick safari lodge":
    "https://upload.wikimedia.org/wikipedia/commons/0/03/Salt_Lick_Game_Lodge_from_the_south_in_the_Taita_Hills_Wildlife_Sanctuary%2C_Kenya.jpg",
  "taita hills safari resort and spa":
    "https://upload.wikimedia.org/wikipedia/commons/b/ba/Group_of_Loxodonta_africana_and_a_game_viewer_vehicle_in_the_evening_south_of_Taita_Hills_Game_Lodge_within_the_Taita_Hills_Wildlife_Sanctuary_in_Kenya.jpg",
  "ole sereni hotel":
    "https://upload.wikimedia.org/wikipedia/commons/0/01/Zebras_at_Nairobi_National_Park%2C_Kenya_2.jpg",
  "sweetwaters serena camp":
    "https://upload.wikimedia.org/wikipedia/commons/e/e2/Ceratotherium_simum_cottoni_-Ol_Pejeta_Conservancy%2C_Kenya.jpg",
  "mount elgon lodge":
    "https://upload.wikimedia.org/wikipedia/commons/5/57/Mount_Elgon-2.jpg",
  "hells gate national park":
    "https://upload.wikimedia.org/wikipedia/commons/7/7f/Hells_Gate_National_Park%2C_Kenya..JPG",
};

const CHAIN_HINTS = [
  ["serena", "https://upload.wikimedia.org/wikipedia/commons/5/57/Maasai_Mara_National_Reserve_Kenya.jpg"],
  ["sarova", "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tsavo_east_panorama.jpg"],
  ["sopa", "https://upload.wikimedia.org/wikipedia/commons/c/c4/Meru_National_Park%2C_Kenya.jpg"],
  ["fairmont", "https://upload.wikimedia.org/wikipedia/commons/e/e8/Mount_Kenya_Sarfari_Club.jpg"],
  ["hemingways", "https://upload.wikimedia.org/wikipedia/commons/4/4e/Watamu_Beach%2C_Kenya_03.jpg"],
  ["beach resort", "https://upload.wikimedia.org/wikipedia/commons/c/c7/Diani_Beach_Sunrise_Kenya.jpg"],
  ["beach club", "https://upload.wikimedia.org/wikipedia/commons/b/b9/Watamu_Beach%2C_Kenya_01.jpg"],
  ["safari lodge", "https://upload.wikimedia.org/wikipedia/commons/5/57/Maasai_Mara_National_Reserve_Kenya.jpg"],
  ["tented camp", "https://upload.wikimedia.org/wikipedia/commons/d/d0/Reserve_samburu_faune_2.jpg"],
  ["camp", "https://upload.wikimedia.org/wikipedia/commons/6/6a/Samburu_National_Reserve%2C_Kenya-26December2012.jpg"],
];

function normalizeKey(name) {
  return name.toLowerCase().trim();
}

export function lookupHotelImage(name, destinationType = "Safari", tier = "midrange", destinationImage) {
  const key = normalizeKey(name);
  if (HOTEL_IMAGES[key]) return HOTEL_IMAGES[key];

  for (const [hint, url] of CHAIN_HINTS) {
    if (key.includes(hint)) return url;
  }

  const typeFallbacks = TIER_FALLBACKS[destinationType] || TIER_FALLBACKS.Safari;
  if (typeFallbacks[tier]) return typeFallbacks[tier];

  return destinationImage || TIER_FALLBACKS.Safari.midrange;
}

export function hotelMapsUrl(hotelName, destinationName) {
  const q = `${hotelName}, ${destinationName}, Kenya`;
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(q)}`;
}

export function normalizeHotelList(hotels, destinationName, destinationType, destinationImage) {
  const tiers = ["budget", "midrange", "luxury"];
  const result = { budget: [], midrange: [], luxury: [], all: [] };

  for (const tier of tiers) {
    for (const name of hotels?.[tier] || []) {
      const entry = {
        name,
        tier,
        image: lookupHotelImage(name, destinationType, tier, destinationImage),
        mapsUrl: hotelMapsUrl(name, destinationName),
      };
      result[tier].push(entry);
      result.all.push(entry);
    }
  }
  return result;
}
