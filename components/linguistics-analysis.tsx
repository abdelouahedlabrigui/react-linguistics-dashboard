"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface LexicalDiversity {
  average_word_length: number
  hapax_legomena: number
  lexical_density: number
  most_frequent_words: Array<{
    frequency: number
    word: string
  }>
}

interface MorphologicalComponents {
  inflectional_variants: Array<{
    original_word: string
    root_word: string
    variations: string[]
  }>
  prefix_analysis: Record<string, number>
  suffix_analysis: Record<string, number>
}

interface PhonologicalFeatures {
  alliteration_instances: any[]
  estimated_syllable_count: number
  phonetic_density: number
  rhyme_patterns: Array<{
    rhyme_ending: string
    word1: string
    word2: string
  }>
  total_consonants: number
  total_vowels: number
  vowel_consonant_ratio: number
}

interface RegisterAnalysis {
  average_sentence_complexity: number
  contraction_usage: number
  formal_language_indicators: number
  formality_score: number
  informal_language_indicators: number
  passive_voice_indicators: number
  register_classification: string
  register_features: {
    academic: boolean
    conversational: boolean
    literary: boolean
    technical: boolean
  }
  technical_vocabulary_usage: number
}

interface SemanticElements {
  concept_density: {
    adjective_ratio: number
    noun_ratio: number
    verb_ratio: number
  }
  content_words: string[]
  named_entities: Array<{
    entity: string
    entity_type: string
  }>
  semantic_fields: Record<string, string[]>
  semantic_richness: number
}

interface SyntacticStructures {
  average_sentence_length: number
  clause_analysis: Array<{
    clause_id: number
    clause_pos: string[]
  }>
}

interface LinguisticInterpretation {
  lexical_diversity: LexicalDiversity
  morphological_components: MorphologicalComponents
  phonological_features: PhonologicalFeatures
  register_analysis: RegisterAnalysis
  semantic_elements: SemanticElements
  syntactic_structures: SyntacticStructures
}

interface LinguisticsResult {
  _id: string
  _index: string
  _score: number
  _source: {
    document_id: string
    linguistic_interpretation: LinguisticInterpretation
  }
}

export function LinguisticsAnalysis() {
  const [query, setQuery] = useState("")
  const [data, setData] = useState<LinguisticsResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const fetchData = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError("")

    try {
      const response = await fetch(
        `http://10.42.0.1:5001/news/linguistics/v1/search?query=${encodeURIComponent(query)}`,
      )
      if (!response.ok) throw new Error("Failed to fetch linguistics data")
      const result = await response.json()
      setData(Array.isArray(result) ? result : [result])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  const getLexicalDensityColor = (density: number) => {
    if (density >= 0.8) return "#22c55e"
    if (density >= 0.6) return "#eab308"
    if (density >= 0.4) return "#f59e0b"
    return "#ef4444"
  }

  const getFormalityColor = (score: number) => {
    if (score >= 0.7) return "#3b82f6"
    if (score >= 0.4) return "#6b7280"
    return "#f59e0b"
  }

  const getComplexityColor = (complexity: number) => {
    if (complexity >= 25) return "#ef4444"
    if (complexity >= 20) return "#f59e0b"
    if (complexity >= 15) return "#eab308"
    return "#22c55e"
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Search Section */}
      <Card className="p-6 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Linguistics Analysis Search</h2>
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
            {loading ? "Searching..." : "Search Linguistics Data"}
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
              {/* Document Header */}
              <div className="border-b border-border pb-4">
                <h4 className="text-xl font-semibold text-primary mb-2">Document Analysis</h4>
                <div className="flex items-center space-x-4 text-sm text-muted-foreground mb-3">
                  <span>ID: {result._source.document_id.slice(0, 8)}...</span>
                  <span>•</span>
                  <span>Score: {result._score.toFixed(2)}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Lexical Diversity */}
                <div className="space-y-3">
                  <h5 className="font-semibold text-lg">Lexical Diversity</h5>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Density</span>
                      <span className="font-medium">
                        {(result._source.linguistic_interpretation.lexical_diversity.lexical_density * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex-1 bg-muted rounded-full h-3 overflow-hidden">
                      <div
                        className="h-full transition-all duration-500"
                        style={{
                          width: `${result._source.linguistic_interpretation.lexical_diversity.lexical_density * 100}%`,
                          backgroundColor: getLexicalDensityColor(
                            result._source.linguistic_interpretation.lexical_diversity.lexical_density,
                          ),
                        }}
                      />
                    </div>
                    <div className="text-xs text-muted-foreground space-y-1">
                      <div>
                        Avg Word Length:{" "}
                        {result._source.linguistic_interpretation.lexical_diversity.average_word_length.toFixed(1)}
                      </div>
                      <div>
                        Unique Words: {result._source.linguistic_interpretation.lexical_diversity.hapax_legomena}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Register Analysis */}
                <div className="space-y-3">
                  <h5 className="font-semibold text-lg">Register Analysis</h5>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Formality</span>
                      <span className="font-medium">
                        {(result._source.linguistic_interpretation.register_analysis.formality_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{
                          backgroundColor: getFormalityColor(
                            result._source.linguistic_interpretation.register_analysis.formality_score,
                          ),
                        }}
                      />
                      <span className="text-sm capitalize">
                        {result._source.linguistic_interpretation.register_analysis.register_classification}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      <div>
                        Passive Voice:{" "}
                        {result._source.linguistic_interpretation.register_analysis.passive_voice_indicators}
                      </div>
                      <div>
                        Technical Terms:{" "}
                        {result._source.linguistic_interpretation.register_analysis.technical_vocabulary_usage}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Syntactic Complexity */}
                <div className="space-y-3">
                  <h5 className="font-semibold text-lg">Syntactic Complexity</h5>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Avg Sentence Length</span>
                      <span className="font-medium">
                        {result._source.linguistic_interpretation.syntactic_structures.average_sentence_length.toFixed(
                          1,
                        )}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{
                          backgroundColor: getComplexityColor(
                            result._source.linguistic_interpretation.syntactic_structures.average_sentence_length,
                          ),
                        }}
                      />
                      <span className="text-sm">
                        {result._source.linguistic_interpretation.syntactic_structures.average_sentence_length >= 25
                          ? "High"
                          : result._source.linguistic_interpretation.syntactic_structures.average_sentence_length >= 20
                            ? "Medium-High"
                            : result._source.linguistic_interpretation.syntactic_structures.average_sentence_length >=
                                15
                              ? "Medium"
                              : "Low"}{" "}
                        Complexity
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      <div>
                        Clauses: {result._source.linguistic_interpretation.syntactic_structures.clause_analysis.length}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Phonological Features */}
              <div className="space-y-3">
                <h5 className="font-semibold text-lg">Phonological Features</h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="bg-muted p-3 rounded-lg text-center">
                    <div className="font-semibold text-lg">
                      {result._source.linguistic_interpretation.phonological_features.estimated_syllable_count}
                    </div>
                    <div className="text-muted-foreground">Syllables</div>
                  </div>
                  <div className="bg-muted p-3 rounded-lg text-center">
                    <div className="font-semibold text-lg">
                      {result._source.linguistic_interpretation.phonological_features.phonetic_density.toFixed(2)}
                    </div>
                    <div className="text-muted-foreground">Phonetic Density</div>
                  </div>
                  <div className="bg-muted p-3 rounded-lg text-center">
                    <div className="font-semibold text-lg">
                      {result._source.linguistic_interpretation.phonological_features.vowel_consonant_ratio.toFixed(2)}
                    </div>
                    <div className="text-muted-foreground">V/C Ratio</div>
                  </div>
                  <div className="bg-muted p-3 rounded-lg text-center">
                    <div className="font-semibold text-lg">
                      {result._source.linguistic_interpretation.phonological_features.rhyme_patterns.length}
                    </div>
                    <div className="text-muted-foreground">Rhyme Patterns</div>
                  </div>
                </div>
              </div>

              {/* Semantic Elements */}
              <div className="space-y-3">
                <h5 className="font-semibold text-lg">Semantic Analysis</h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h6 className="font-medium mb-2">Concept Density</h6>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span>Nouns:</span>
                        <span>
                          {(
                            result._source.linguistic_interpretation.semantic_elements.concept_density.noun_ratio * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Verbs:</span>
                        <span>
                          {(
                            result._source.linguistic_interpretation.semantic_elements.concept_density.verb_ratio * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Adjectives:</span>
                        <span>
                          {(
                            result._source.linguistic_interpretation.semantic_elements.concept_density.adjective_ratio *
                            100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h6 className="font-medium mb-2">Semantic Richness</h6>
                    <div className="flex items-center space-x-3">
                      <div className="flex-1 bg-muted rounded-full h-3 overflow-hidden">
                        <div
                          className="h-full bg-blue-500 transition-all duration-500"
                          style={{
                            width: `${result._source.linguistic_interpretation.semantic_elements.semantic_richness * 100}%`,
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium">
                        {(result._source.linguistic_interpretation.semantic_elements.semantic_richness * 100).toFixed(
                          1,
                        )}
                        %
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Most Frequent Words */}
              <div className="space-y-3">
                <h5 className="font-semibold text-lg">Most Frequent Words</h5>
                <div className="flex flex-wrap gap-2">
                  {result._source.linguistic_interpretation.lexical_diversity.most_frequent_words
                    .slice(0, 10)
                    .map((wordData, idx) => (
                      <span key={idx} className="px-3 py-1 bg-secondary text-secondary-foreground rounded-full text-sm">
                        {wordData.word} ({wordData.frequency})
                      </span>
                    ))}
                </div>
              </div>

              {/* Named Entities */}
              {result._source.linguistic_interpretation.semantic_elements.named_entities.length > 0 && (
                <div className="space-y-3">
                  <h5 className="font-semibold text-lg">Named Entities</h5>
                  <div className="flex flex-wrap gap-2">
                    {result._source.linguistic_interpretation.semantic_elements.named_entities.map((entity, idx) => (
                      <span key={idx} className="px-3 py-1 bg-accent text-accent-foreground rounded-lg text-sm">
                        {entity.entity} <span className="text-xs opacity-75">({entity.entity_type})</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>
        ))}
      </div>

      {/* Empty State */}
      {data.length === 0 && !loading && (
        <div className="text-center py-12">
          <p className="text-muted-foreground text-lg">Enter a search query above to analyze linguistic data</p>
        </div>
      )}
    </div>
  )
}
