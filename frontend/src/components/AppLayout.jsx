import TopNav from "./TopNav";
import { PAGE_BACKDROP } from "../data/themeImages";

export default function AppLayout({ children, sidebar }) {
  return (
    <div className="relative flex h-screen flex-col overflow-hidden">
      <div
        className="travel-page-bg pointer-events-none fixed inset-0 z-0"
        style={{ "--travel-backdrop-url": `url('${PAGE_BACKDROP}')` }}
        aria-hidden
      />

      <div className="relative z-10 flex min-h-0 flex-1 flex-col">
        <TopNav />
        <div className="flex min-h-0 flex-1">{sidebar}{children}</div>
      </div>
    </div>
  );
}

export { BackLink } from "./BackLink";
