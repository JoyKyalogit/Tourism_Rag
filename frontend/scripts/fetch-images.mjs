const places = [
  ["maasai mara", "Maasai Mara National Reserve Kenya"],
  ["amboseli", "Amboseli National Park Kenya elephants"],
  ["diani beach", "Diani Beach Kenya"],
  ["watamu", "Watamu Kenya beach"],
  ["naivasha", "Lake Naivasha Kenya"],
  ["nairobi", "Nairobi Kenya skyline"],
  ["tsavo", "Tsavo National Park Kenya"],
  ["nanyuki and mount kenya", "Mount Kenya Nanyuki"],
  ["mombasa", "Mombasa Kenya Fort Jesus"],
  ["lamu", "Lamu Old Town Kenya"],
  ["malindi", "Malindi Kenya beach"],
  ["samburu", "Samburu National Reserve Kenya"],
  ["meru national park", "Meru National Park Kenya"],
  ["nakuru", "Lake Nakuru flamingos Kenya"],
  ["aberdare", "Aberdare National Park Kenya"],
  ["kisumu", "Kisumu Lake Victoria Kenya"],
  ["kakamega forest", "Kakamega Forest Kenya"],
  ["lake turkana", "Lake Turkana Kenya"],
  ["marsabit", "Marsabit National Park Kenya"],
  ["isiolo and shaba", "Shaba National Reserve Kenya"],
  ["chyulu hills", "Chyulu Hills Kenya"],
  ["taita hills", "Taita Hills Kenya safari"],
  ["nairobi national park", "Nairobi National Park Kenya"],
  ["ol pejeta conservancy", "Ol Pejeta Conservancy Kenya"],
  ["mount elgon", "Mount Elgon Kenya"],
  ["hells gate", "Hells Gate National Park Kenya"],
];

async function searchImages(query, limit = 4) {
  const url =
    "https://commons.wikimedia.org/w/api.php?" +
    new URLSearchParams({
      action: "query",
      generator: "search",
      gsrsearch: query,
      gsrnamespace: "6",
      gsrlimit: String(limit),
      prop: "imageinfo",
      iiprop: "url",
      iiurlwidth: "1200",
      format: "json",
      origin: "*",
    });
  const res = await fetch(url);
  const data = await res.json();
  const pages = data.query?.pages || {};
  return Object.values(pages)
    .map((p) => p.imageinfo?.[0]?.url)
    .filter(Boolean);
}

const out = {};
for (const [key, query] of places) {
  try {
    const urls = await searchImages(query, 4);
    if (urls.length) {
      out[key] = { image: urls[0], gallery: urls.slice(0, 4) };
      console.log("OK", key, urls.length);
    } else {
      console.log("EMPTY", key);
    }
  } catch (e) {
    console.log("ERR", key, e.message);
  }
  await new Promise((r) => setTimeout(r, 350));
}

console.log("---JSON---");
console.log(JSON.stringify(out, null, 2));
