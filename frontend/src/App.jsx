import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import CategoryPage from "./pages/CategoryPage";
import DestinationPage from "./pages/DestinationPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/explore/:categorySlug" element={<CategoryPage />} />
        <Route path="/place/:placeSlug" element={<DestinationPage />} />
      </Routes>
    </BrowserRouter>
  );
}
