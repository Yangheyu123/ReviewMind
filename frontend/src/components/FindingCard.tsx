import { cn } from '@/lib/utils';
import { RiskBadge } from './RiskBadge';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ChevronDown, ChevronUp, Locate } from 'lucide-react';
import type { Finding } from '../types';

interface FindingCardProps {
  finding: Finding;
  expanded?: boolean;
  onToggle?: () => void;
  onJumpToDiff?: () => void;
}

const levelBorderColors: Record<string, string> = {
  CRITICAL: 'border-l-red-500/40',
  HIGH: 'border-l-orange-500/40',
  MEDIUM: 'border-l-yellow-500/40',
  LOW: 'border-l-blue-500/40',
  WARNING: 'border-l-amber-500/40',
  INFO: 'border-l-zinc-500/40',
};

export function FindingCard({ finding, expanded = false, onToggle, onJumpToDiff }: FindingCardProps) {
  return (
    <Card
      className={cn(
        'bg-zinc-900/40 border-l-2 border-zinc-800/60 transition-all hover:bg-zinc-800/30 py-0 gap-0',
        levelBorderColors[finding.level] || 'border-l-zinc-500/40',
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-2 flex-wrap min-w-0">
            <RiskBadge level={finding.level} />
            <Badge variant="outline" className="text-xs font-mono text-zinc-400 bg-zinc-900/60 border-zinc-800/60">
              {finding.type}
            </Badge>
            <span className="text-xs text-zinc-600">{finding.agent}</span>
          </div>
          {onToggle && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onToggle}
              className="text-xs text-zinc-500 hover:text-zinc-200 h-6 px-2"
            >
              {expanded ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
              {expanded ? '收起' : '展开'}
            </Button>
          )}
        </div>

        <div className="flex items-center gap-2 text-sm font-mono text-zinc-600 mb-2">
          <span>{finding.file}</span>
          <span className="text-zinc-500">L{finding.line}</span>
          {finding.symbol && (
            <span className="text-purple-400/70 truncate">→ {finding.symbol}</span>
          )}
          {onJumpToDiff && (
            <Button
              variant="outline"
              size="sm"
              onClick={(e) => { e.stopPropagation(); onJumpToDiff(); }}
              className="ml-auto shrink-0 text-xs text-zinc-500 hover:text-zinc-200 border-zinc-800/60 h-6 px-2"
            >
              <Locate className="size-3" /> 定位 Diff
            </Button>
          )}
        </div>

        <p className="text-sm text-zinc-300 leading-relaxed mb-3">{finding.description}</p>

        <div className="bg-emerald-950/15 border border-emerald-500/15 rounded-lg p-3">
          <span className="text-xs font-semibold text-emerald-400/80">建议: </span>
          <span className="text-sm text-emerald-300/70">{finding.suggestion}</span>
        </div>

        {expanded && finding.code_snippet && (
          <pre className="mt-3 rounded-lg bg-zinc-950/70 border border-zinc-800/40 p-3 text-xs font-mono text-zinc-300 overflow-auto whitespace-pre-wrap">
            {finding.code_snippet}
          </pre>
        )}

        {finding.confidence > 0 && (
          <div className="mt-3 flex items-center gap-2">
            <span className="text-xs text-zinc-600">置信度</span>
            <Progress value={Math.round(finding.confidence * 100)} className="h-1.5 w-20" />
            <span className="text-xs text-zinc-600">{Math.round(finding.confidence * 100)}%</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}