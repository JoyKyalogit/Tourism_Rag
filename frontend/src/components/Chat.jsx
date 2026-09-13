import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import LoadingSpinner from "./LoadingSpinner";

export default function Chat({ messages, loading, error }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, error]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto px-1 py-2 pr-2">
        {messages.length === 0 && !loading && (
          <div className="flex h-full min-h-[200px] flex-col items-center justify-center text-center">
            <div className="mb-3 text-4xl">🦁</div>
            <h2 className="font-display text-xl font-semibold text-white">
              Welcome to Travel Africa
            </h2>
            <p className="mt-2 max-w-sm text-sm text-slate-400">
              Your AI travel assistant for beaches, safaris, hotels, and itineraries across Kenya.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {loading && <LoadingSpinner />}

        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200 animate-fade-in">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
