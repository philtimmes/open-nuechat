/**
 * AgentStatus - Shows the current agent state machine status
 */

import type { AgentTask } from '../../pages/VibeCode';

type AgentState = 'idle' | 'analyzing' | 'planning' | 'generating' | 'reviewing' | 'applying' | 'testing' | 'complete' | 'error';

interface AgentStatusProps {
  state: AgentState;
  task: AgentTask | null;
}

const stateConfig: Record<AgentState, { icon: string; label: string; color: string }> = {
  idle: { icon: '💤', label: 'Idle', color: 'text-gray-400' },
  analyzing: { icon: '🔍', label: 'Analyzing request...', color: 'text-blue-400' },
  planning: { icon: '📋', label: 'Creating execution plan...', color: 'text-purple-400' },
  generating: { icon: '⚡', label: 'Generating code...', color: 'text-yellow-400' },
  reviewing: { icon: '👀', label: 'Reviewing changes...', color: 'text-orange-400' },
  applying: { icon: '✏️', label: 'Applying changes...', color: 'text-cyan-400' },
  testing: { icon: '🧪', label: 'Running tests & linting...', color: 'text-pink-400' },
  complete: { icon: '✅', label: 'Complete!', color: 'text-green-400' },
  error: { icon: '❌', label: 'Error occurred', color: 'text-red-400' },
};

const stateOrder: AgentState[] = ['analyzing', 'planning', 'generating', 'reviewing', 'applying', 'testing', 'complete'];

export default function AgentStatus({ state, task }: AgentStatusProps) {
  const config = stateConfig[state];
  const currentIndex = stateOrder.indexOf(state);
  
  return (
    <div className="bg-[var(--color-surface)] border-b border-[var(--color-border)] px-4 py-2">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg animate-bounce">{config.icon}</span>
          <span className={`text-sm font-medium ${config.color}`}>{config.label}</span>
        </div>
        {task && (
          <span className="text-xs text-[var(--color-text-secondary)] truncate max-w-xs">
            {task.description}
          </span>
        )}
      </div>
      
      {/* Progress bar */}
      <div className="flex gap-1">
        {stateOrder.slice(0, -1).map((s, i) => (
          <div
            key={s}
            className={`flex-1 h-1 rounded-full transition-colors ${
              i < currentIndex
                ? 'bg-green-500'
                : i === currentIndex
                ? 'bg-[var(--color-primary)] animate-pulse'
                : 'bg-[var(--color-border)]'
            }`}
          />
        ))}
      </div>
      
      {/* State steps */}
      <div className="flex justify-between mt-1">
        {stateOrder.slice(0, -1).map((s, i) => (
          <span
            key={s}
            className={`text-[9px] ${
              i <= currentIndex ? 'text-[var(--color-text)]' : 'text-[var(--color-text-secondary)]'
            }`}
          >
            {stateConfig[s].icon}
          </span>
        ))}
      </div>
    </div>
  );
}
