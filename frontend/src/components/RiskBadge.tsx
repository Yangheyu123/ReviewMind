import { Badge } from '@/components/ui/badge';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import type { RiskLevel } from '../types';

const riskBadgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold',
  {
    variants: {
      level: {
        CRITICAL: 'bg-red-950/40 text-red-400 border-red-500/20',
        HIGH: 'bg-orange-950/40 text-orange-400 border-orange-500/20',
        MEDIUM: 'bg-yellow-950/40 text-yellow-400 border-yellow-500/20',
        LOW: 'bg-blue-950/40 text-blue-400 border-blue-500/20',
        WARNING: 'bg-amber-950/40 text-amber-400 border-amber-500/20',
        INFO: 'bg-zinc-800/60 text-zinc-400 border-zinc-700/30',
      },
    },
    defaultVariants: {
      level: 'INFO',
    },
  },
);

interface RiskBadgeProps extends VariantProps<typeof riskBadgeVariants> {
  level: RiskLevel;
}

export function RiskBadge({ level }: RiskBadgeProps) {
  return (
    <span className={cn(riskBadgeVariants({ level }))}>
      {level}
    </span>
  );
}