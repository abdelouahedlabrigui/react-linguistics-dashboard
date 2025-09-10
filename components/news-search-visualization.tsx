"use client"

import type React from "react"

import { useState } from "react"

interface NewsSearchData {
  actions: {
    content: string[]
    plot: string
  }
  decisions: {
    content: string[]
    plot: string
  }
  facts: {
    content: string[]
    plot: string
  }
}

export function NewsSearchVisualization() {
  const [query, setQuery] = useState("")
  const [data, setData] = useState<NewsSearchData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(
        `http://10.42.0.1:5000/news_search_retrieve_actions_by_query?query=${encodeURIComponent(query)}`,
      )

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  const renderSection = (title: string, sectionData: { content: string[]; plot: string }) => (
    <div className="bg-card rounded-lg border border-border p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4 capitalize">{title}</h3>

      {/* Content List */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-muted-foreground mb-2">Content:</h4>
        {sectionData.content.length > 0 ? (
          <ul className="space-y-1">
            {sectionData.content.map((item, index) => (
              <li key={index} className="text-sm text-foreground bg-muted px-3 py-2 rounded">
                {item || <span className="text-muted-foreground italic">Empty</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground italic">No content available</p>
        )}
      </div>

      {/* Plot Visualization */}
      <div>
        <h4 className="text-sm font-medium text-muted-foreground mb-2">Visualization:</h4>
        {sectionData.plot ? (
          <div className="bg-muted rounded-lg p-4 overflow-hidden">
            <img
              src={`data:image/png;base64,${sectionData.plot}`}
              alt={`${title} visualization`}
              className="max-w-full h-auto rounded"
              style={{ maxHeight: "400px", objectFit: "contain" }}
            />
          </div>
        ) : (
          <div className="bg-muted rounded-lg p-8 text-center">
            <p className="text-muted-foreground italic">No visualization available</p>
          </div>
        )}
      </div>
    </div>
  )

  return (
    <div className="max-w-7xl mx-auto px-6">
      {/* Search Section */}
      <div className="bg-card rounded-lg border border-border p-6 mb-8">
        <h2 className="text-xl font-semibold text-foreground mb-4">News Search & Visualization</h2>
        <div className="flex gap-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter your search query..."
            className="flex-1 px-4 py-2 border border-border rounded-lg bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          />
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 mb-8">
          <p className="text-destructive font-medium">Error: {error}</p>
        </div>
      )}

      {/* Results Display */}
      {data && (
        <div className="space-y-8">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-foreground mb-2">Search Results</h2>
            <p className="text-muted-foreground">Query: "{query}"</p>
          </div>

          <div className="grid gap-8 lg:grid-cols-1 xl:grid-cols-1">
            {renderSection("Actions", data.actions)}
            {renderSection("Decisions", data.decisions)}
            {renderSection("Facts", data.facts)}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!data && !loading && !error && (
        <div className="text-center py-12">
          <div className="bg-muted rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
            <svg className="w-8 h-8 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-foreground mb-2">Start Your Search</h3>
          <p className="text-muted-foreground">
            Enter a query to retrieve and visualize news actions, decisions, and facts.
          </p>
        </div>
      )}
    </div>
  )
}
