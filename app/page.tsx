"use client"

import { useState } from "react"
import { Navbar } from "@/components/navbar"
import { LinguisticsAnalysis } from "@/components/linguistics-analysis"
import { PsychoLinguisticsAnalysis } from "@/components/psycho-linguistics-analysis"
import { NewsSearchVisualization } from "@/components/news-search-visualization"

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<"linguistics" | "psycho" | "news-search">("linguistics")

  return (
    <div className="min-h-screen bg-background">
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="px-6 pb-8">
        {activeTab === "linguistics" ? (
          <LinguisticsAnalysis />
        ) : activeTab === "psycho" ? (
          <PsychoLinguisticsAnalysis />
        ) : (
          <NewsSearchVisualization />
        )}
      </div>
    </div>
  )
}

