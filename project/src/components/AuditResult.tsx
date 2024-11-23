import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle } from 'lucide-react';

interface AuditResultProps {
  title: string;
  description: string;
  status: 'error' | 'warning' | 'success';
  details: string;
}

export const AuditResult: React.FC<AuditResultProps> = ({
  title,
  description,
  status,
  details
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      default:
        return <CheckCircle className="w-5 h-5 text-green-500" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'error':
        return 'bg-red-50/95 border-red-200';
      case 'warning':
        return 'bg-yellow-50/95 border-yellow-200';
      case 'success':
        return 'bg-green-50/95 border-green-200';
      default:
        return 'bg-green-50/95 border-green-200';
    }
  };

  return (
    <div className={`p-4 rounded-lg backdrop-blur-sm border ${getStatusColor()}`}>
      <div className="flex items-start gap-3">
        {getStatusIcon()}
        <div className="flex-1">
          <h3 className="font-medium mb-1">{title}</h3>
          {description && <p className="text-gray-600 mb-2">{description}</p>}
          {details && (
            <div className="text-sm text-gray-500 whitespace-pre-wrap">
              {details}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};