const COMMONS_ORIGINAL =
  /^(https:\/\/upload\.wikimedia\.org\/wikipedia\/commons)\/([0-9a-f])\/([0-9a-f]{2})\/([^/?#]+\.(?:jpe?g|png|webp|gif))$/i;

// Wikimedia caches these widths; arbitrary widths are slower and get rate-limited.
export const THUMB_SIZES = { tiny: 250, small: 500, card: 960, hero: 1280 };

/** Resized Wikimedia Commons URL (originals are often several MB). Other URLs pass through. */
export function thumbUrl(url, width = THUMB_SIZES.card) {
  if (!url) return url;
  const match = url.match(COMMONS_ORIGINAL);
  if (!match) return url;
  const [, base, a, ab, file] = match;
  return `${base}/thumb/${a}/${ab}/${file}/${width}px-${file}`;
}

/** onError handler: retry the full-size original (thumbs fail for images narrower than `width`), then an optional fallback. */
export function handleImageError(original, finalFallback) {
  return (e) => {
    const img = e.currentTarget;
    if (original && img.dataset.retry !== "original" && img.dataset.retry !== "final") {
      img.dataset.retry = "original";
      img.src = original;
      return;
    }
    if (finalFallback && img.dataset.retry !== "final") {
      img.dataset.retry = "final";
      img.src = finalFallback;
    }
  };
}
