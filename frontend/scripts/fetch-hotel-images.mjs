import { readFileSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const dataPath = join(__dirname, "../../data/processed/kenya_tourism_data.json");
const rows = JSON.parse(readFileSync(dataPath, "utf8"));

const names = new Set();
for (const row of rows) {
  for (const tier of ["budget", "midrange", "luxury"]) {
    for (const h of row.hotels?.[tier] || []) names.add(h);
  }
}

async function searchImage(query) {
  const url =
    "https://commons.wikimedia.org/w/api.php?" +
    new URLSearchParams({
      action: "query",
      generator: "search",
      gsrsearch: query,
      gsrnamespace: "6",
      gsrlimit: "5",
      prop: "imageinfo",
      iiprop: "url",
      iiurlwidth: "800",
      format: "json",
      origin: "*",
    });
  const res = await fetch(url);
  const text = await res.text();
  if (!text.startsWith("{")) throw new Error(text.slice(0, 80));
  const data = JSON.parse(text);
  const pages = Object.values(data.query?.pages || {});
  for (const p of pages) {
    const u = p.imageinfo?.[0]?.url;
    if (u && /\.(jpe?g|png)$/i.test(u.split("?")[0])) return u;
  }
  return null;
}

const out = {};
const list = [...names].sort();
for (const name of list) {
  try {
    const url = await searchImage(`${name} Kenya hotel lodge`);
    if (url) {
      out[name.toLowerCase()] = url;
      console.log("OK", name);
    } else {
      console.log("MISS", name);
    }
  } catch (e) {
    console.log("ERR", name, e.message);
  }
  await new Promise((r) => setTimeout(r, 2500));
}

const outPath = join(__dirname, "../src/data/hotelImages.raw.json");
writeFileSync(outPath, JSON.stringify(out, null, 2));
console.log("Wrote", outPath, Object.keys(out).length, "images");
