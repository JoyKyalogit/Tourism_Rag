export function getFriendlyError(error) {
  if (!error) return "Oops, something went wrong. Try again.";

  if (error.code === "ECONNABORTED") {
    return "The request took too long. Please try again.";
  }

  if (!error.response) {
    return "Cannot reach the server. Restart the backend on port 8000, then restart the frontend dev server (`npm run dev`) and try again.";
  }

  const status = error.response.status;
  const detail = error.response.data?.detail;

  if (status === 404) {
    return "Service not found. Check that the backend API is running.";
  }

  if (status === 400 && typeof detail === "string") {
    if (detail.includes("Vector index")) {
      return "The knowledge base is not ready yet. Run `python scripts/ingest.py` in the backend folder.";
    }
    return detail;
  }

  if (status >= 500) {
    return "Our servers are having trouble. Please try again in a moment.";
  }

  if (typeof detail === "string") {
    if (detail.startsWith("{") || detail.includes('"detail"')) {
      return "Oops, something went wrong. Try again.";
    }
    return detail;
  }
  return "Oops, something went wrong. Try again.";
}
