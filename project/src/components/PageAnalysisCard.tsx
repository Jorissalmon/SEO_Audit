import React from 'react';
import { CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';

interface PageAnalysisCardProps {
  pageData: PageAnalysis;
}

export const PageAnalysisCard: React.FC<PageAnalysisCardProps> = ({ pageData }) => {
  const getStatusIcon = (status: 'success' | 'warning' | 'error') => {
    switch (status) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-500" />;
    }
  };

  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-lg shadow-lg overflow-hidden">
      {/* En-tête de la carte */}
      <div className="p-4 border-b bg-gray-50">
        <h3 className="text-lg font-semibold text-gray-800 truncate">{pageData.url}</h3>
        <div className="mt-2 flex items-center gap-2">
          <div className={`text-2xl font-bold ${
            pageData.score >= 80 ? 'text-green-600' :
            pageData.score >= 60 ? 'text-yellow-600' : 'text-red-600'
          }`}>
            {pageData.score}/100
          </div>
          <span className="text-sm text-gray-500">Score global</span>
        </div>
      </div>

      {/* Résumé */}
      <div className="p-4 bg-blue-50 border-b">
        <p className="text-blue-800">{pageData.summary}</p>
      </div>

      {/* Critères */}
      <div className="p-4 space-y-6">
        {Object.entries(pageData.criteria).map(([key, section]) => (
          <div key={key} className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-gray-700">{section.title}</h4>
              <span className={`px-2 py-1 rounded-full text-sm ${
                section.score >= 80 ? 'bg-green-100 text-green-800' :
                section.score >= 60 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {section.score}/100
              </span>
            </div>
            <div className="space-y-2">
              {section.items.map((item, index) => (
                <div key={index} className="flex items-start gap-2 text-sm">
                  {getStatusIcon(item.status)}
                  <div>
                    <span className="font-medium">{item.name}:</span>
                    <span className="ml-1 text-gray-600">{item.message}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}; 