"use client"

interface NavbarProps {
  activeTab: "linguistics" | "psycho" | "news-search"
  onTabChange: (tab: "linguistics" | "psycho" | "news-search") => void
}

export function Navbar({ activeTab, onTabChange }: NavbarProps) {
  return (
    <nav className="bg-card border-b border-border mb-8">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <h1 className="text-xl font-bold text-foreground">News Analysis Dashboard</h1>
          <div className="flex space-x-1">
            <button
              onClick={() => onTabChange("linguistics")}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === "linguistics"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
            >
              Linguistics Analysis
            </button>
            <button
              onClick={() => onTabChange("psycho")}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === "psycho"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
            >
              Psycho-Linguistics
            </button>
            <button
              onClick={() => onTabChange("news-search")}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === "news-search"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
            >
              News Search
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
}

