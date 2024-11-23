import React, { useState } from 'react';
import { Search, Activity, AlertCircle, Loader2 } from 'lucide-react';
import { AuditResult } from './components/AuditResult';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';

interface AuditResponse {
  message: string;
  data_received: any;  // Peut être ajusté selon les données spécifiques
}

interface AuditResult {
  title: string;
  description: string;
  status: 'error' | 'warning' | 'success';
  details: string;
  score: number;
}

interface SEOAnalysis {
  url: string;
  summary: {
    score: number;
    total_issues: number;
    status: 'good' | 'fair' | 'poor';
  };
  detailed_analysis: {
    critical_issues: SEOIssue[];
    warnings: SEOIssue[];
    passed_checks: SEOIssue[];
  };
  recommendations: {
    priority_fixes: string[];
    impact_score: number;
  };
  timestamp: string;
}

interface SEOIssue {
  title: string;
  description: string;
  status: string;
  details: string;
}

function App() {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<AuditResult[]>([]);
  const [score, setScore] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [analysis, setAnalysis] = useState<SEOAnalysis | null>(null);
  const [apiData, setApiData] = useState<any>(null);
  const [pagesAnalyzed, setPagesAnalyzed] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setResults([]);
    setScore(null);

    try {
      const requestData = {
        url: url.trim()
      };

      console.log('Sending data:', requestData); // Debug log

      const response = await fetch('http://127.0.0.1:8000/api/app', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ url })
      });

      console.log('Response status:', response.status); // Debug log

      // Lire le texte brut de la réponse pour le debug
      const responseText = await response.text();
      console.log('Response text:', responseText);

      // Parser la réponse en JSON
      const data = JSON.parse(responseText);
      console.log('Parsed data:', data);
      setApiData(data);

      if (!response.ok) {
        throw new Error(data.message || `HTTP error! status: ${response.status}`);
      }

      // Créer un résultat plus détaillé
      setResults([
        {
          title: "Résultat de l'analyse SEO",
          description: `Score moyen: ${data.summary.average_score}%`,
          status: 'success',
          details: `
            Pages analysées: ${data.summary.total_pages}
            Pages avec bon score: ${data.summary.good_pages}
            Pages avec score moyen: ${data.summary.fair_pages}
            Pages avec mauvais score: ${data.summary.poor_pages}

            Détails par page:
            ${data.pages.map((page: { url: string; score: number; error?: string }) => `
            URL: ${page.url}
            Score: ${page.score}
            ${page.error ? `Erreur: ${page.error}` : ''}
            `).join('\n')}

            Analyse effectuée le: ${new Date(data.timestamp).toLocaleString()}
                      `,
          score: data.summary.average_score
        }
      ]);

    } catch (err) {
      console.error('Error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  // Fonction pour transformer les données des mots-clés pour le treemap
  const transformKeywordsForTreemap = (keywords: [string, number][]) => {
    return [{
      name: "Keywords",
      children: keywords.map(([word, count]) => ({
        name: word,
        size: count,
      }))
    }];
  };

  return (
    // VIdeo d'arrière plan
    <div className="relative min-h-screen overflow-hidden">
      {/* Video Background */}
      <div className="absolute inset-0 w-full h-full">
        <div className="absolute inset-0 bg-black/20 backdrop-blur-sm z-10"></div>
        <video
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        >
          <source
            src="/background.mp4"
            type="video/mp4"
          />
        </video>
      </div>

      {/* Content */}
      <div className="relative z-20">
        <div className="max-w-6xl mx-auto px-4 py-12">
          <div className="text-center mb-12">
            <div className="flex items-center justify-center mb-4">
              <Activity className="w-16 h-16 text-[#EF233C]" />
            </div>
            <h1 className="text-5xl font-bold text-white mb-4 drop-shadow-lg">
              SEO Audit Tool
            </h1>
            <p className="text-lg text-gray-100 max-w-2xl mx-auto drop-shadow">
              Entre l'url de ton Siteweb pour avoir une analyse SEO et des recommendations par IA, dans le but d'être encore mieux référencé.
            </p>
          </div>

          {/* Champ pour rentrer le lien avec boutton*/}
          <form onSubmit={handleSubmit} className="max-w-2xl mx-auto mb-12">
            <div className="flex gap-4">
              <div className="relative flex-1">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com"
                  required
                  className="block w-full pl-10 pr-3 py-4 border border-gray-200 rounded-lg focus:ring-2 focus:ring-[#EF233C] focus:border-[#EF233C] bg-white/90 backdrop-blur-sm shadow-sm transition-all duration-200"
                />
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="px-8 py-4 bg-[#EF233C] text-white font-medium rounded-lg hover:bg-[#D90429] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#EF233C] disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all duration-200 shadow-sm"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyse en cours...
                  </>
                ) : (
                  'Analyze'
                )}
              </button>
            </div>
          </form>

          {error && (
            <div className="max-w-2xl mx-auto mb-8 p-4 bg-red-50/95 backdrop-blur-sm border border-red-200 rounded-lg flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700">{error}</p>
            </div>
          )}
          {/* Colloration en fonction du score SEO */}
          {score !== null && (
            <div className="max-w-2xl mx-auto mb-8">
              <div className={`text-center p-6 rounded-lg backdrop-blur-sm ${score >= 80 ? 'bg-green-50/95 text-green-700' :
                score >= 60 ? 'bg-yellow-50/95 text-yellow-700' :
                  'bg-red-50/95 text-red-700'
                }`}>
                <h2 className="text-3xl font-bold mb-2">SEO Score: {score}/100</h2>
                <p className="text-lg">
                  {score >= 80 ? 'Great job! Your website has strong SEO fundamentals.' :
                    score >= 60 ? 'Your website needs some improvements.' :
                      'Your website requires significant SEO optimization.'}
                </p>
              </div>
            </div>
          )}

          {results.length > 0 && (
            <div className="max-w-4xl mx-auto space-y-6">
              {results.map((result, index) => (
                <div key={index} className="bg-white/95 backdrop-blur-sm rounded-lg p-6 shadow">
                  <h2 className="text-2xl font-bold mb-4">{result.title}</h2>
                  <p className="text-lg">{result.description}</p>
                  <p className="mt-2"><strong>Status:</strong> {result.status}</p>
                  <pre className="whitespace-pre-wrap">{result.details}</pre>
                  <p className="mt-2"><strong>Score:</strong> {result.score}</p>
                </div>
              ))}


              <div className="max-w-4xl mx-auto space-y-6">
                {/* Treemap des mots-clés */}
                <div className="bg-white/95 backdrop-blur-sm rounded-lg p-6 shadow">
                  <h2 className="text-2xl font-bold mb-4">Mots-clés</h2>
                  <div style={{ width: '100%', height: 400 }}>
                    <ResponsiveContainer>
                      <Treemap
                        data={transformKeywordsForTreemap(apiData?.keywords || [])}
                        dataKey="size"
                        aspectRatio={4 / 3}
                        stroke="#fff"
                        fill="black"
                      >
                        <Tooltip content={<CustomTooltip />} />
                      </Treemap>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Résumé des statistiques */}
                <div className="bg-white/95 backdrop-blur-sm rounded-lg p-6 shadow">
                  <h2 className="text-2xl font-bold mb-4">Quelques Statistiques</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard
                      title="Score Moyen"
                      value={`${apiData?.summary.average_score.toFixed(1)}%`}
                    />
                    <StatCard
                      title="Nb total Pages"
                      value={apiData?.summary.total_pages}
                    />
                    <StatCard
                      title="Liens valides"
                      value={apiData?.summary.valid_links}
                    />
                    <StatCard
                      title="Liens invalides"
                      value={apiData?.summary.invalid_links}
                    />
                    <StatCard
                      title="Liens vérifiés"
                      value={apiData?.summary.verified_links}
                    />
                    <StatCard
                      title="Total des liens vérifiés"
                      value={apiData?.summary.total_verified_links}
                    />
                    <StatCard
                      title="Liens internes valides"
                      value={apiData?.summary.valid_internal_links}
                    />
                    <StatCard
                      title="Liens externes valides"
                      value={apiData?.summary.valid_external_links}
                    />
                    <StatCard
                      title="Liens internes invalides"
                      value={apiData?.summary.invalid_internal_links}
                    />
                    <StatCard
                      title="Liens externes invalides"
                      value={apiData?.summary.invalid_external_links}
                    />
                    <StatCard
                      title="Total d'images"
                      value={apiData?.summary.total_images}
                    />
                    <StatCard
                      title="Images avec texte alt"
                      value={apiData?.summary.images_with_alt_text}
                    />
                    <StatCard
                      title="Images sans texte alt"
                      value={apiData?.summary.images_without_alt_text}
                    />
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-white/95 backdrop-blur-sm rounded-lg p-6 shadow">
                  <h2 className="text-2xl font-bold mb-4">Analyse Générale</h2>
                  <div className="prose max-w-none">
                    {apiData?.overall_gpt_recommendations.split('\n').map((line: string, index: number) => (
                      <p key={index}>{line}</p>
                    ))}
                  </div>
                </div>

                {/* Analyse par page */}
                {isLoading && (
                  <div className="w-full bg-gray-200 rounded-full h-4 mb-4">
                    <div className="bg-blue-500 h-4 rounded-full" style={{ width: `${(pagesAnalyzed / totalPages) * 100}%` }}></div>
                  </div>
                )}

                {apiData?.pages.map((page: {
                  url: string;
                  seo_score: number;
                  links: {
                    total_links: number;
                    valid_links: number;
                    invalid_links: number;
                    verified_links: number;
                  };
                  gpt_recommendation: string;
                }, index: number) => (
                  <div key={index} className="bg-white/95 backdrop-blur-sm rounded-lg p-6 shadow">
                    <h2 className="text-2xl font-bold mb-4">Analyse de page</h2>
                    <div className="space-y-4">
                      <p><strong>URL:</strong> {page.url}</p>

                      {/* Aperçu du score avec couleur */}
                      <div className={`p-2 rounded-lg text-white font-bold ${page.seo_score >= 80 ? 'bg-green-500' :
                        page.seo_score >= 60 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}>
                        Aperçu du score : {page.seo_score.toFixed(1)}%
                      </div>

                      <div className="border-t pt-4">
                        <h3 className="text-xl font-bold mb-2">Analyse des liens</h3>
                        <div className="grid grid-cols-2 gap-4">
                          <p><strong>Total Links:</strong> {page.links.total_links}</p>
                          <p><strong>Valid Links:</strong> {page.links.valid_links}</p>
                          <p><strong>Invalid Links:</strong> {page.links.invalid_links}</p>
                          <p><strong>Verified Links:</strong> {page.links.verified_links}</p>
                        </div>
                      </div>
                      <div className="border-t pt-4">
                        <h3 className="text-xl font-bold mb-2">Recommandations</h3>
                        <div className="prose max-w-none">
                          {page.gpt_recommendation.split('\n').map((line: string, index: number) => (
                            <p key={index}>{line}</p>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {isLoading && (
                <div className="text-center py-4">
                  <div className="animate-spin h-8 w-8 border-4 border-blue-500 rounded-full border-t-transparent mx-auto"></div>
                </div>
              )}

              {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded my-4">
                  {error}
                </div>
              )}

              {analysis && (
                <div className="space-y-6 mt-8">
                  {/* Score Summary */}
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center justify-between">
                      <h2 className="text-2xl font-bold">SEO Score: {analysis.summary.score}/100</h2>
                      <span className={`px-4 py-2 rounded-full ${analysis.summary.status === 'good' ? 'bg-green-100 text-green-800' :
                        analysis.summary.status === 'fair' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                        {analysis.summary.status.toUpperCase()}
                      </span>
                    </div>
                    <p className="mt-2 text-gray-600">
                      Found {analysis.summary.total_issues} issues to address
                    </p>
                  </div>

                  {/* Critical Issues */}
                  {analysis.detailed_analysis.critical_issues.length > 0 && (
                    <div className="bg-white rounded-lg shadow p-6">
                      <h3 className="text-xl font-bold text-red-600 mb-4">Critical Issues</h3>
                      <div className="space-y-4">
                        {analysis.detailed_analysis.critical_issues.map((issue, index) => (
                          <div key={index} className="border-l-4 border-red-500 pl-4">
                            <h4 className="font-semibold">{issue.title}</h4>
                            <p className="text-gray-600">{issue.description}</p>
                            <p className="text-sm text-gray-500 mt-1">{issue.details}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Warnings */}
                  {analysis.detailed_analysis.warnings.length > 0 && (
                    <div className="bg-white rounded-lg shadow p-6">
                      <h3 className="text-xl font-bold text-yellow-600 mb-4">Warnings</h3>
                      <div className="space-y-4">
                        {analysis.detailed_analysis.warnings.map((issue, index) => (
                          <div key={index} className="border-l-4 border-yellow-500 pl-4">
                            <h4 className="font-semibold">{issue.title}</h4>
                            <p className="text-gray-600">{issue.description}</p>
                            <p className="text-sm text-gray-500 mt-1">{issue.details}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-xl font-bold text-blue-600 mb-4">Priority Fixes</h3>
                    <div className="space-y-2">
                      {analysis.recommendations.priority_fixes.map((fix, index) => (
                        <div key={index} className="flex items-center">
                          <span className="mr-2">•</span>
                          <p>{fix}</p>
                        </div>
                      ))}
                    </div>
                    <div className="mt-4 pt-4 border-t">
                      <p className="text-gray-600">
                        Potential impact score improvement: +{analysis.recommendations.impact_score} points
                      </p>
                    </div>
                  </div>

                  {/* Passed Checks */}
                  {analysis.detailed_analysis.passed_checks.length > 0 && (
                    <div className="bg-white rounded-lg shadow p-6">
                      <h3 className="text-xl font-bold text-green-600 mb-4">Passed Checks</h3>
                      <div className="space-y-4">
                        {analysis.detailed_analysis.passed_checks.map((check, index) => (
                          <div key={index} className="border-l-4 border-green-500 pl-4">
                            <h4 className="font-semibold">{check.title}</h4>
                            <p className="text-gray-600">{check.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="text-right text-sm text-gray-500">
                    Analysis performed: {new Date(analysis.timestamp).toLocaleString()}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Composant pour la carte de statistiques
const StatCard = ({ title, value }) => (
  <div className="bg-gray-50 p-4 rounded-lg">
    <h3 className="text-sm font-medium text-gray-500">{title}</h3>
    <p className="mt-1 text-2xl font-semibold">{value}</p>
  </div>
);

// Composant pour le tooltip du treemap
const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: any[] }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-2 border rounded shadow">
        <p>{`${payload[0].payload.name}: ${payload[0].value}`}</p>
      </div>
    );
  }
  return null;
};

export default App;
