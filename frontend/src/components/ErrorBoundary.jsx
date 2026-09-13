import { Component } from "react";

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-[#0B0F19] p-6 text-center">
          <div className="max-w-md rounded-2xl border border-red-500/30 bg-red-500/10 p-6">
            <p className="text-lg font-semibold text-white">Something went wrong</p>
            <p className="mt-2 text-sm text-red-200">{this.state.error.message}</p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="mt-4 rounded-xl bg-white/10 px-4 py-2 text-sm text-white hover:bg-white/20"
            >
              Reload page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
