"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface NewsArticle {
  author: string
  content: string
  description: string
  id: number
  publishedAt: string
  source: string
  title: string
  url: string
  urlToImage: string
}

interface PsycholinguisticInterpretation {
  attention_allocation: string[]
  cognitive_load: {
    contributing_elements: string[]
    score: number
  }
  emotional_valence: {
    contributing_words_phrases: string[]
    velance: string
  }
  priming_effects: string[]
  processing_fluency: {
    contributing_factors: string[]
    level: string
  }
  reader_model: string
  working_memory_demands: string[]
}

interface AnalysisResult {
  _id: string
  _index: string
  _score: number
  _source: {
    document_id: string
    news_article: NewsArticle
    psycholinguistic_interpretation: PsycholinguisticInterpretation
    timestamp: string
  }
}

export function PsychoLinguisticsAnalysis() {
  const [query, setQuery] = useState("")
  const [data, setData] = useState<AnalysisResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const fetchData = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError("")

    try {
      const response = await fetch(
        `http://10.42.0.1:5001/news/psycho_linguistics/v1/search?query=${encodeURIComponent(query)}`,
      )
      if (!response.ok) throw new Error("Failed to fetch psycho-linguistics data")
      const result = await response.json()
      setData(Array.isArray(result) ? result : [result])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  const getCognitiveLoadColor = (score: number) => {
    if (score >= 0.8) return "#ef4444"
    if (score >= 0.6) return "#f59e0b"
    if (score >= 0.4) return "#eab308"
    return "#22c55e"
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Search Section */}
      <Card className="p-6 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Psycho-Linguistics Analysis Search</h2>
        <div className="space-y-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className="w-full px-4 py-3 border border-border rounded-lg bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-ring text-lg"
            onKeyPress={(e) => e.key === "Enter" && fetchData()}
          />
          <Button onClick={fetchData} disabled={loading || !query.trim()} className="w-full py-3 text-lg">
            {loading ? "Searching..." : "Search Psycho-Linguistics Data"}
          </Button>
          {error && <p className="text-destructive text-sm">{error}</p>}
        </div>
      </Card>

      {/* Results Section */}
      <div className="space-y-6">
        <h3 className="text-xl font-semibold">Results ({data.length})</h3>
        {data.map((result, index) => (
          <Card key={result._id || index} className="p-6">
            <div className="space-y-6">
              {/* Article Header */}
              <div className="border-b border-border pb-4">
                <h4 className="text-xl font-semibold text-primary mb-2">{result._source.news_article.title}</h4>
                <div className="flex items-center space-x-4 text-sm text-muted-foreground mb-3">
                  <span>By {result._source.news_article.author}</span>
                  <span>•</span>
                  <span>{new Date(result._source.news_article.publishedAt).toLocaleDateString()}</span>
                  <span>•</span>
                  <span>Score: {result._score.toFixed(2)}</span>
                </div>
                <p className="text-foreground">{result._source.news_article.description}</p>
              </div>

              {/* Cognitive Load with detailed breakdown */}
              <div className="space-y-4">
                <h5 className="font-semibold text-lg">Cognitive Load Analysis</h5>
                <div className="flex items-center space-x-3 mb-4">
                  <div className="flex-1 bg-muted rounded-full h-6 overflow-hidden">
                    <div
                      className="h-full transition-all duration-500"
                      style={{
                        width: `${result._source.psycholinguistic_interpretation.cognitive_load.score * 100}%`,
                        backgroundColor: getCognitiveLoadColor(
                          result._source.psycholinguistic_interpretation.cognitive_load.score,
                        ),
                      }}
                    />
                  </div>
                  <span className="text-xl font-bold min-w-[4rem]">
                    {(result._source.psycholinguistic_interpretation.cognitive_load.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <p className="font-medium mb-2 text-sm">Contributing Elements:</p>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    {result._source.psycholinguistic_interpretation.cognitive_load.contributing_elements.map(
                      (element, idx) => (
                        <li key={idx} className="flex items-start">
                          <span className="text-primary mr-2">•</span>
                          <span>{element}</span>
                        </li>
                      ),
                    )}
                  </ul>
                </div>
              </div>

              {/* Priming Effects - Enhanced Display */}
              {result._source.psycholinguistic_interpretation.priming_effects.length > 0 && (
                <div className="space-y-4">
                  <h5 className="font-semibold text-lg">Priming Effects</h5>
                  <div className="bg-accent/10 p-4 rounded-lg">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {result._source.psycholinguistic_interpretation.priming_effects
                        .slice(0, 12)
                        .map((effect, idx) => (
                          <div key={idx} className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-accent rounded-full flex-shrink-0" />
                            <span className="text-sm text-foreground">
                              {effect.length > 50 ? `${effect}` : effect}
                            </span>
                          </div>
                        ))}
                    </div>
                    {result._source.psycholinguistic_interpretation.priming_effects.length > 12 && (
                      <div className="mt-3 pt-3 border-t border-border">
                        <span className="text-sm text-muted-foreground">
                          +{result._source.psycholinguistic_interpretation.priming_effects.length - 12} more priming
                          effects
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Working Memory Demands */}
              {result._source.psycholinguistic_interpretation.working_memory_demands.length > 0 && (
                <div className="space-y-3">
                  <h5 className="font-semibold text-lg">Working Memory Demands</h5>
                  <div className="bg-secondary/20 p-4 rounded-lg space-y-3">
                    {result._source.psycholinguistic_interpretation.working_memory_demands.map((demand, idx) => (
                      <div key={idx} className="flex items-start space-x-3">
                        <div className="w-6 h-6 bg-secondary rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-bold text-secondary-foreground">{idx + 1}</span>
                        </div>
                        <p className="text-sm text-foreground">{demand}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Reader Model - Enhanced */}
              <div className="space-y-3">
                <h5 className="font-semibold text-lg">Reader Model Profile</h5>
                <div className="bg-primary/5 border border-primary/20 p-4 rounded-lg">
                  <p className="text-foreground font-medium">
                    {result._source.psycholinguistic_interpretation.reader_model}
                  </p>
                </div>
              </div>

              {/* Processing Fluency */}
              <div className="space-y-3">
                <h5 className="font-semibold text-lg">Processing Fluency</h5>
                <div className="flex items-center space-x-4 mb-3">
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-primary rounded-full" />
                    <span className="font-medium capitalize">
                      {result._source.psycholinguistic_interpretation.processing_fluency.level} Fluency
                    </span>
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  <p className="font-medium mb-2">Factors affecting fluency:</p>
                  <ul className="space-y-1">
                    {result._source.psycholinguistic_interpretation.processing_fluency.contributing_factors.map(
                      (factor, idx) => (
                        <li key={idx}>• {factor}</li>
                      ),
                    )}
                  </ul>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Empty State */}
      {data.length === 0 && !loading && (
        <div className="text-center py-12">
          <p className="text-muted-foreground text-lg">Enter a search query above to analyze psycho-linguistic data</p>
        </div>
      )}
    </div>
  )
}
