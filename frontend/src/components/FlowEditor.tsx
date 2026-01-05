/**
 * FlowEditor - Visual node-based filter chain editor
 * Similar to n8n workflow builder with side panel configuration
 */
import { useCallback, useState, useMemo, useRef, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Connection,
  Edge,
  Node,
  MarkerType,
  Panel,
  useReactFlow,
  ReactFlowProvider,
  Handle,
  Position,
  NodeProps,
  NodeChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// ============================================================================
// TYPES
// ============================================================================

interface StepConfig {
  [key: string]: unknown;
}

interface StepPosition {
  x: number;
  y: number;
}

interface FilterChainStep {
  id: string;
  type: string;
  name?: string;
  enabled?: boolean;
  config: StepConfig;
  position?: StepPosition;
  on_error?: string;
  jump_to_step?: string;
  add_to_context?: boolean;  // Add step output to context for main LLM
  context_label?: string;    // Label for context entry
  conditional?: {
    enabled: boolean;
    logic?: string;
    comparisons: Array<{ left: string; operator: string; right: string }>;
    on_true?: FilterChainStep[];
    on_false?: FilterChainStep[];
  };
}

interface FilterChainDefinition {
  steps: FilterChainStep[];
}

interface FlowEditorProps {
  definition: FilterChainDefinition;
  onChange: (definition: FilterChainDefinition) => void;
  availableTools?: Array<{ value: string; label: string; category: string }>;
  filterChains?: Array<{ id: string; name: string }>;
}

interface StepNodeData extends Record<string, unknown> {
  step: FilterChainStep;
  stepIndex: number;
  selected: boolean;
  onSelect: (stepId: string) => void;
}

// ============================================================================
// STEP TYPE DEFINITIONS
// ============================================================================

const STEP_TYPES: Record<string, { icon: string; label: string; color: string; category: string }> = {
  // AI Actions
  to_llm: { icon: '🤖', label: 'Ask AI', color: '#8B5CF6', category: 'AI' },
  query: { icon: '🔍', label: 'Generate Query', color: '#8B5CF6', category: 'AI' },
  
  // Tool Actions
  to_tool: { icon: '🔧', label: 'Run Tool', color: '#F59E0B', category: 'Tools' },
  
  // Flow Control
  keyword_check: { icon: '🔑', label: 'Keyword Check', color: '#10B981', category: 'Flow' },
  go_to_llm: { icon: '➡️', label: 'Send to Main AI', color: '#10B981', category: 'Flow' },
  filter_complete: { icon: '✅', label: 'Complete', color: '#10B981', category: 'Flow' },
  stop: { icon: '⛔', label: 'Stop', color: '#EF4444', category: 'Flow' },
  block: { icon: '🚫', label: 'Block', color: '#EF4444', category: 'Flow' },
  
  // Data
  set_var: { icon: '📝', label: 'Set Variable', color: '#3B82F6', category: 'Data' },
  set_array: { icon: '📋', label: 'Set Array', color: '#3B82F6', category: 'Data' },
  context_insert: { icon: '📎', label: 'Add Context', color: '#3B82F6', category: 'Data' },
  modify: { icon: '✏️', label: 'Modify', color: '#3B82F6', category: 'Data' },
  
  // Logic
  compare: { icon: '⚖️', label: 'Compare', color: '#EC4899', category: 'Logic' },
  call_chain: { icon: '🔗', label: 'Call Chain', color: '#EC4899', category: 'Logic' },
  branch: { icon: '🔀', label: 'Branch', color: '#06B6D4', category: 'Logic' },
};

const COMPARISON_OPERATORS = [
  { value: 'eq', label: '= equals' },
  { value: 'ne', label: '≠ not equals' },
  { value: 'contains', label: 'contains' },
  { value: 'not_contains', label: 'not contains' },
  { value: 'starts_with', label: 'starts with' },
  { value: 'ends_with', label: 'ends with' },
  { value: 'gt', label: '> greater than' },
  { value: 'lt', label: '< less than' },
  { value: 'gte', label: '>= greater or equal' },
  { value: 'lte', label: '<= less or equal' },
  { value: 'is_empty', label: 'is empty' },
  { value: 'is_not_empty', label: 'is not empty' },
  { value: 'matches', label: 'matches regex' },
];

// ============================================================================
// STEP NODE COMPONENT (Simplified - config in side panel)
// ============================================================================

// Branch output colors for visual distinction
const BRANCH_OUTPUT_COLORS = ['#22C55E', '#EF4444', '#F59E0B', '#8B5CF6', '#06B6D4', '#EC4899'];

function StepNode({ data }: NodeProps<Node<StepNodeData, string>>) {
  const { step, stepIndex, selected, onSelect } = data;
  const stepType = STEP_TYPES[step.type] || { icon: '❓', label: step.type, color: '#6B7280', category: 'Other' };
  
  // Get output variable if defined
  const outputVar = step.config?.output_var || step.config?.var_name;
  
  // Get branch outputs if this is a branch node
  const branchOutputs = step.type === 'branch' && Array.isArray(step.config?.outputs) 
    ? step.config.outputs as Array<{ id: string; label: string; condition?: { var: string; operator: string; value: string } }>
    : null;

  return (
    <div
      className={`relative min-w-[200px] rounded-lg shadow-lg border-2 transition-all cursor-pointer ${
        selected ? 'border-blue-500 ring-2 ring-blue-500/30' : 'border-transparent hover:border-gray-500'
      } ${step.enabled === false ? 'opacity-50' : ''}`}
      style={{ backgroundColor: 'var(--color-surface)' }}
      onClick={() => onSelect(step.id)}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-gray-400 !border-2 !border-gray-600"
      />
      
      {/* Step number badge */}
      <div 
        className="absolute -top-2 -left-2 w-5 h-5 rounded-full bg-gray-700 text-white text-[10px] font-bold flex items-center justify-center border border-gray-500"
        title={`Execution order: ${stepIndex + 1}`}
      >
        {stepIndex + 1}
      </div>
      
      {/* Header */}
      <div
        className="px-3 py-2 rounded-t-lg flex items-center gap-2"
        style={{ backgroundColor: stepType.color }}
      >
        <span className="text-lg">{stepType.icon}</span>
        <div className="flex-1 min-w-0">
          <span className="font-medium text-white text-sm">{stepType.label}</span>
        </div>
        {step.enabled === false && (
          <span className="text-white/70 text-xs">OFF</span>
        )}
      </div>
      
      {/* Summary */}
      <div className="px-3 py-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        {step.name && step.name !== 'Ask LLM' && step.name !== stepType.label && (
          <div className="font-medium truncate" style={{ color: 'var(--color-text)' }}>{step.name}</div>
        )}
        {typeof outputVar === 'string' && outputVar ? (
          <div className="truncate">→ ${String(outputVar)}</div>
        ) : null}
        {step.type === 'to_tool' && typeof step.config?.tool_name === 'string' && step.config.tool_name ? (
          <div className="truncate">{String(step.config.tool_name)}</div>
        ) : null}
        {(step.type === 'to_llm' || step.type === 'query') && typeof step.config?.prompt === 'string' && step.config.prompt ? (
          <div className="truncate">{String(step.config.prompt).slice(0, 40)}...</div>
        ) : null}
        
        {/* Branch outputs preview */}
        {branchOutputs && branchOutputs.length > 0 && (
          <div className="mt-1 space-y-0.5">
            {branchOutputs.map((output, idx) => (
              <div key={output.id} className="flex items-center gap-1">
                <span 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: BRANCH_OUTPUT_COLORS[idx % BRANCH_OUTPUT_COLORS.length] }}
                />
                <span className="truncate">{output.label || `Output ${idx + 1}`}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Output handles - single for normal nodes, multiple for branch nodes */}
      {branchOutputs && branchOutputs.length > 0 ? (
        <div className="relative h-4">
          {branchOutputs.map((output, idx) => (
            <Handle
              key={output.id}
              type="source"
              position={Position.Bottom}
              id={output.id}
              className="!w-3 !h-3 !border-2"
              style={{ 
                left: `${((idx + 1) / (branchOutputs.length + 1)) * 100}%`,
                backgroundColor: BRANCH_OUTPUT_COLORS[idx % BRANCH_OUTPUT_COLORS.length],
                borderColor: '#374151',
              }}
            />
          ))}
        </div>
      ) : (
        <Handle
          type="source"
          position={Position.Bottom}
          className="!w-3 !h-3 !bg-gray-400 !border-2 !border-gray-600"
        />
      )}
    </div>
  );
}

// ============================================================================
// START NODE
// ============================================================================

function StartNode() {
  return (
    <div className="px-4 py-2 rounded-full bg-green-600 text-white font-medium shadow-lg">
      <span className="mr-2">▶</span> Start
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-green-400 !border-2 !border-green-700"
      />
    </div>
  );
}

// ============================================================================
// NODE PALETTE
// ============================================================================

interface NodePaletteProps {
  onAddNode: (type: string) => void;
}

function NodePalette({ onAddNode }: NodePaletteProps) {
  const categories = useMemo(() => {
    const cats: Record<string, Array<{ type: string; icon: string; label: string; color: string }>> = {};
    Object.entries(STEP_TYPES).forEach(([type, info]) => {
      if (!cats[info.category]) cats[info.category] = [];
      cats[info.category].push({ type, ...info });
    });
    return cats;
  }, []);

  return (
    <div className="bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] p-2 w-44">
      <div className="text-xs font-medium text-[var(--color-text)] mb-2">Add Node</div>
      {Object.entries(categories).map(([category, items]) => (
        <div key={category} className="mb-2">
          <div className="text-[10px] text-[var(--color-text-secondary)] mb-1 uppercase">{category}</div>
          <div className="space-y-0.5">
            {items.map(item => (
              <button
                key={item.type}
                onClick={() => onAddNode(item.type)}
                className="w-full px-2 py-1 rounded text-left text-xs flex items-center gap-1.5 hover:bg-[var(--color-background)] transition-colors"
                style={{ color: 'var(--color-text)' }}
              >
                <span
                  className="w-5 h-5 rounded flex items-center justify-center text-white text-[10px]"
                  style={{ backgroundColor: item.color }}
                >
                  {item.icon}
                </span>
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className="border-t border-[var(--color-border)] mt-2 pt-2 text-[10px] text-[var(--color-text-secondary)]">
        <div className="font-medium mb-1">💡 Tips:</div>
        <ul className="space-y-0.5 list-disc list-inside">
          <li>Drag handles to connect nodes</li>
          <li>Click wire to delete it</li>
          <li>Numbers show exec order</li>
        </ul>
      </div>
    </div>
  );
}

// ============================================================================
// VARIABLE INSPECTOR
// ============================================================================

interface VariableInspectorProps {
  steps: FilterChainStep[];
  currentStepIndex: number;
}

function VariableInspector({ steps, currentStepIndex }: VariableInspectorProps) {
  const [expanded, setExpanded] = useState(true);
  
  const availableVars = useMemo(() => {
    const vars: Array<{ name: string; source: string; stepIndex: number; type: string }> = [
      { name: '$Query', source: 'Input', stepIndex: -1, type: 'string' },
      { name: '$PreviousResult', source: 'Previous Step', stepIndex: -1, type: 'any' },
    ];
    
    steps.slice(0, currentStepIndex).forEach((s, i) => {
      const outputVar = s.config?.output_var || s.config?.var_name;
      if (outputVar && typeof outputVar === 'string') {
        const stepType = STEP_TYPES[s.type];
        let varType = 'any';
        if (s.type === 'to_llm' || s.type === 'query') varType = 'string';
        if (s.type === 'to_tool') varType = 'object';
        if (s.type === 'set_array') varType = 'array';
        if (s.type === 'compare') varType = 'boolean';
        
        vars.push({
          name: `$Var[${outputVar}]`,
          source: stepType?.label || s.type,
          stepIndex: i,
          type: varType,
        });
      }
    });
    
    return vars;
  }, [steps, currentStepIndex]);

  return (
    <div className="border-t border-[var(--color-border)] mt-4 pt-4">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-sm font-medium text-[var(--color-text)] w-full"
      >
        <span>{expanded ? '▼' : '▶'}</span>
        <span>📊 Available Variables</span>
        <span className="text-xs text-[var(--color-text-secondary)]">({availableVars.length})</span>
      </button>
      
      {expanded && (
        <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
          {availableVars.map((v, i) => (
            <div
              key={i}
              className="flex items-center justify-between px-2 py-1.5 rounded bg-[var(--color-background)] text-xs"
            >
              <code className="text-blue-400 font-mono">{v.name}</code>
              <span className="text-[var(--color-text-secondary)]">
                <span className="px-1 py-0.5 rounded bg-[var(--color-surface)] mr-1">{v.type}</span>
                {v.source}
              </span>
            </div>
          ))}
          {currentStepIndex < 0 && (
            <div className="text-xs text-[var(--color-text-secondary)] italic p-2">
              Select a step to see available variables
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// CONFIGURATION PANEL
// ============================================================================

interface ConfigPanelProps {
  step: FilterChainStep | null;
  stepIndex: number;
  allSteps: FilterChainStep[];
  onUpdate: (updates: Partial<FilterChainStep>) => void;
  onDelete: () => void;
  availableTools?: Array<{ value: string; label: string; category: string }>;
  filterChains?: Array<{ id: string; name: string }>;
}

function ConfigPanel({ step, stepIndex, allSteps, onUpdate, onDelete, availableTools, filterChains }: ConfigPanelProps) {
  if (!step) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 flex items-center justify-center text-[var(--color-text-secondary)] text-sm">
          <div className="text-center">
            <div className="text-4xl mb-2">👆</div>
            <div>Select a node to configure</div>
            <div className="text-xs mt-2 opacity-70">or drag from palette to add</div>
          </div>
        </div>
        <div className="p-3 border-t border-[var(--color-border)]">
          <VariableInspector steps={allSteps} currentStepIndex={-1} />
        </div>
      </div>
    );
  }

  const stepType = STEP_TYPES[step.type] || { icon: '❓', label: step.type, color: '#6B7280', category: 'Other' };
  
  const getConfigStr = (key: string, defaultVal = ''): string => {
    const legacyKeyMap: Record<string, string[]> = {
      'tool': ['tool', 'tool_name'],
      'system': ['system', 'system_prompt'],
    };
    const keysToCheck = legacyKeyMap[key] || [key];
    for (const k of keysToCheck) {
      const val = step.config?.[k];
      if (typeof val === 'string' && val) return val;
    }
    return defaultVal;
  };

  const updateConfig = (key: string, value: unknown) => {
    onUpdate({ config: { ...step.config, [key]: value } });
  };

  // Available variables for insertion
  const availableVars = useMemo(() => {
    const vars = [
      { value: '$Query', label: 'User Message' },
      { value: '$PreviousResult', label: 'Previous Result' },
    ];
    allSteps.slice(0, stepIndex).forEach((s, i) => {
      const outputVar = s.config?.output_var || s.config?.var_name;
      if (outputVar && typeof outputVar === 'string') {
        vars.push({ value: `$Var[${outputVar}]`, label: `Step ${i + 1}: ${outputVar}` });
      }
    });
    return vars;
  }, [allSteps, stepIndex]);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-3 border-b border-[var(--color-border)] flex items-center gap-2" style={{ backgroundColor: stepType.color }}>
        <span className="text-xl">{stepType.icon}</span>
        <span className="font-medium text-white flex-1">{stepType.label}</span>
        <span className="text-white/70 text-xs">Step {stepIndex + 1}</span>
      </div>
      
      {/* Config Form */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Basic */}
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Custom Label</label>
            <input
              type="text"
              value={step.name === 'Ask LLM' ? '' : (step.name || '')}
              onChange={(e) => onUpdate({ name: e.target.value || undefined })}
              placeholder={stepType.label}
              className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
            />
          </div>
          
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={step.enabled !== false}
              onChange={(e) => onUpdate({ enabled: e.target.checked })}
              className="rounded"
            />
            <span className="text-sm text-[var(--color-text)]">Enabled</span>
          </label>
        </div>

        {/* Type-specific config */}
        {(step.type === 'to_llm' || step.type === 'query') && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">
                {step.type === 'query' ? 'Query Prompt' : 'AI Prompt'}
              </label>
              <div className="flex flex-wrap gap-1 mb-1">
                {availableVars.map(v => (
                  <button
                    key={v.value}
                    onClick={() => updateConfig('prompt', getConfigStr('prompt') + `{${v.value}}`)}
                    className="px-1.5 py-0.5 text-[10px] rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
                  >
                    {v.label}
                  </button>
                ))}
              </div>
              <textarea
                value={getConfigStr('prompt')}
                onChange={(e) => updateConfig('prompt', e.target.value)}
                rows={4}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] resize-none"
                placeholder="Enter prompt..."
              />
            </div>
            {step.type === 'to_llm' && (
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">System Prompt</label>
                <textarea
                  value={getConfigStr('system')}
                  onChange={(e) => updateConfig('system', e.target.value)}
                  rows={2}
                  className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] resize-none"
                  placeholder="Optional system prompt..."
                />
              </div>
            )}
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Save Result As</label>
              <input
                type="text"
                value={getConfigStr('output_var')}
                onChange={(e) => updateConfig('output_var', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="e.g., AIResponse"
              />
            </div>
          </div>
        )}

        {step.type === 'to_tool' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Tool</label>
              <select
                value={getConfigStr('tool') || getConfigStr('tool_name')}
                onChange={(e) => updateConfig('tool_name', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
              >
                <option value="">Select a tool...</option>
                {availableTools?.map(tool => (
                  <option key={tool.value} value={tool.value}>{tool.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Input Variable</label>
              <input
                type="text"
                value={getConfigStr('input_var', '$Query')}
                onChange={(e) => updateConfig('input_var', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Save Result As</label>
              <input
                type="text"
                value={getConfigStr('output_var')}
                onChange={(e) => updateConfig('output_var', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="e.g., ToolResult"
              />
            </div>
          </div>
        )}

        {step.type === 'set_var' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Variable Name</label>
              <input
                type="text"
                value={getConfigStr('var_name')}
                onChange={(e) => updateConfig('var_name', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="e.g., MyVar"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Value</label>
              <input
                type="text"
                value={getConfigStr('value')}
                onChange={(e) => updateConfig('value', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="Value or {$Variable}"
              />
            </div>
          </div>
        )}

        {step.type === 'set_array' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Array Name</label>
              <input
                type="text"
                value={getConfigStr('var_name') || getConfigStr('output_var')}
                onChange={(e) => { updateConfig('var_name', e.target.value); updateConfig('output_var', e.target.value); }}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="e.g., urls"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Values (JSON array or expressions)</label>
              <textarea
                value={Array.isArray(step.config?.values) ? JSON.stringify(step.config.values, null, 2) : '[]'}
                onChange={(e) => {
                  try {
                    updateConfig('values', JSON.parse(e.target.value));
                  } catch {
                    // Invalid JSON, keep as-is
                  }
                }}
                rows={4}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] font-mono resize-none"
                placeholder='["value1", "value2"]'
              />
            </div>
          </div>
        )}

        {step.type === 'compare' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Left Value</label>
              <input
                type="text"
                value={getConfigStr('left')}
                onChange={(e) => updateConfig('left', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="$Query or value"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Operator</label>
              <select
                value={getConfigStr('operator', 'eq')}
                onChange={(e) => updateConfig('operator', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
              >
                {COMPARISON_OPERATORS.map(op => (
                  <option key={op.value} value={op.value}>{op.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Right Value</label>
              <input
                type="text"
                value={getConfigStr('right')}
                onChange={(e) => updateConfig('right', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="Value to compare"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Save Result As</label>
              <input
                type="text"
                value={getConfigStr('output_var')}
                onChange={(e) => updateConfig('output_var', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="e.g., CompareResult"
              />
            </div>
          </div>
        )}

        {step.type === 'modify' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Target</label>
              <select
                value={getConfigStr('target', 'query')}
                onChange={(e) => updateConfig('target', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
              >
                <option value="query">User Message</option>
                <option value="response">AI Response</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">New Value</label>
              <input
                type="text"
                value={getConfigStr('transform')}
                onChange={(e) => updateConfig('transform', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="New value (can use {$Variables})"
              />
            </div>
          </div>
        )}

        {step.type === 'context_insert' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Label</label>
              <input
                type="text"
                value={getConfigStr('label') || getConfigStr('context_label')}
                onChange={(e) => updateConfig('context_label', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="e.g., Search Results"
              />
            </div>
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Content Variable</label>
              <input
                type="text"
                value={getConfigStr('content_var') || getConfigStr('source_var', '$PreviousResult')}
                onChange={(e) => updateConfig('source_var', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
              />
            </div>
          </div>
        )}

        {step.type === 'call_chain' && (
          <div>
            <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Chain to Call</label>
            <select
              value={getConfigStr('chain_name')}
              onChange={(e) => updateConfig('chain_name', e.target.value)}
              className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
            >
              <option value="">Select a chain...</option>
              {filterChains?.map(chain => (
                <option key={chain.id} value={chain.name}>{chain.name}</option>
              ))}
            </select>
          </div>
        )}

        {step.type === 'branch' && (
          <div className="space-y-3">
            <div className="text-xs text-[var(--color-text-secondary)] p-2 bg-[var(--color-background)] rounded">
              Configure conditional outputs. First matching condition wins. Connect each output to its target step.
            </div>
            
            {/* Branch outputs */}
            {(Array.isArray(step.config?.outputs) ? step.config.outputs : []).map((output: { id: string; label: string; condition?: { var: string; operator: string; value: string }; jump_to?: string }, idx: number) => (
              <div key={output.id} className="p-2 rounded border border-[var(--color-border)] space-y-2">
                <div className="flex items-center gap-2">
                  <span 
                    className="w-3 h-3 rounded-full flex-shrink-0" 
                    style={{ backgroundColor: BRANCH_OUTPUT_COLORS[idx % BRANCH_OUTPUT_COLORS.length] }}
                  />
                  <input
                    type="text"
                    value={output.label || ''}
                    onChange={(e) => {
                      const outputs = [...(step.config?.outputs as Array<{ id: string; label: string; condition?: { var: string; operator: string; value: string }; jump_to?: string }> || [])];
                      outputs[idx] = { ...outputs[idx], label: e.target.value };
                      updateConfig('outputs', outputs);
                    }}
                    placeholder={idx === (step.config?.outputs as unknown[])?.length - 1 ? 'Else (default)' : `Output ${idx + 1}`}
                    className="flex-1 px-2 py-1 rounded text-xs bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                  />
                  {(step.config?.outputs as unknown[])?.length > 2 && (
                    <button
                      onClick={() => {
                        const outputs = [...(step.config?.outputs as Array<{ id: string; label: string }> || [])];
                        outputs.splice(idx, 1);
                        updateConfig('outputs', outputs);
                      }}
                      className="text-red-400 hover:text-red-300 text-xs"
                    >
                      ✕
                    </button>
                  )}
                </div>
                
                {/* Condition (not for last output - that's the "else") */}
                {idx < (step.config?.outputs as unknown[])?.length - 1 && (
                  <div className="space-y-1 pl-5">
                    <div className="text-[10px] text-[var(--color-text-secondary)]">If:</div>
                    <div className="flex gap-1">
                      <input
                        type="text"
                        value={output.condition?.var || ''}
                        onChange={(e) => {
                          const outputs = [...(step.config?.outputs as Array<{ id: string; label: string; condition?: { var: string; operator: string; value: string } }> || [])];
                          outputs[idx] = { 
                            ...outputs[idx], 
                            condition: { ...outputs[idx].condition!, var: e.target.value, operator: outputs[idx].condition?.operator || 'contains', value: outputs[idx].condition?.value || '' }
                          };
                          updateConfig('outputs', outputs);
                        }}
                        placeholder="$Var[name]"
                        className="flex-1 px-1.5 py-0.5 rounded text-[10px] bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      />
                      <select
                        value={output.condition?.operator || 'contains'}
                        onChange={(e) => {
                          const outputs = [...(step.config?.outputs as Array<{ id: string; label: string; condition?: { var: string; operator: string; value: string } }> || [])];
                          outputs[idx] = { 
                            ...outputs[idx], 
                            condition: { ...outputs[idx].condition!, operator: e.target.value, var: outputs[idx].condition?.var || '', value: outputs[idx].condition?.value || '' }
                          };
                          updateConfig('outputs', outputs);
                        }}
                        className="w-24 px-1 py-0.5 rounded text-[10px] bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      >
                        {COMPARISON_OPERATORS.map(op => (
                          <option key={op.value} value={op.value}>{op.label}</option>
                        ))}
                      </select>
                      <input
                        type="text"
                        value={output.condition?.value || ''}
                        onChange={(e) => {
                          const outputs = [...(step.config?.outputs as Array<{ id: string; label: string; condition?: { var: string; operator: string; value: string } }> || [])];
                          outputs[idx] = { 
                            ...outputs[idx], 
                            condition: { ...outputs[idx].condition!, value: e.target.value, var: outputs[idx].condition?.var || '', operator: outputs[idx].condition?.operator || 'contains' }
                          };
                          updateConfig('outputs', outputs);
                        }}
                        placeholder="value"
                        className="flex-1 px-1.5 py-0.5 rounded text-[10px] bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      />
                    </div>
                  </div>
                )}
                
                {/* Jump target for this output */}
                <div className="pl-5">
                  <select
                    value={output.jump_to || ''}
                    onChange={(e) => {
                      const outputs = [...(step.config?.outputs as Array<{ id: string; label: string; jump_to?: string }> || [])];
                      outputs[idx] = { ...outputs[idx], jump_to: e.target.value || undefined };
                      updateConfig('outputs', outputs);
                    }}
                    className="w-full px-1.5 py-0.5 rounded text-[10px] bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                  >
                    <option value="">Then: Continue to next step</option>
                    {allSteps.filter(s => s.id !== step.id).map((s) => {
                      const sType = STEP_TYPES[s.type];
                      return (
                        <option key={s.id} value={s.id}>
                          Then: {sType?.label || s.type} (Step {allSteps.indexOf(s) + 1})
                        </option>
                      );
                    })}
                  </select>
                </div>
              </div>
            ))}
            
            {/* Add output button */}
            <button
              onClick={() => {
                const outputs = [...(step.config?.outputs as Array<{ id: string; label: string }> || [])];
                // Insert before the last (else) output
                const newOutput = { id: `output_${Date.now()}`, label: '', condition: { var: '', operator: 'contains', value: '' } };
                outputs.splice(outputs.length - 1, 0, newOutput);
                updateConfig('outputs', outputs);
              }}
              className="w-full px-2 py-1.5 rounded text-xs bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 flex items-center justify-center gap-1"
            >
              <span>+</span> Add Output
            </button>
          </div>
        )}

        {step.type === 'keyword_check' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">
                Keywords (one per line)
              </label>
              <textarea
                value={Array.isArray(step.config?.keywords) ? (step.config.keywords as string[]).join('\n') : ''}
                onChange={(e) => updateConfig('keywords', e.target.value.split('\n').filter(k => k.trim()))}
                rows={4}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] font-mono"
                placeholder="Enter keywords, one per line"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Match Mode</label>
                <select
                  value={getConfigStr('mode', 'any')}
                  onChange={(e) => updateConfig('mode', e.target.value)}
                  className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                >
                  <option value="any">Any keyword</option>
                  <option value="all">All keywords</option>
                </select>
              </div>
              
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Match Type</label>
                <select
                  value={getConfigStr('match_type', 'contains')}
                  onChange={(e) => updateConfig('match_type', e.target.value)}
                  className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                >
                  <option value="contains">Contains</option>
                  <option value="word">Whole word</option>
                  <option value="exact">Exact match</option>
                  <option value="starts_with">Starts with</option>
                  <option value="ends_with">Ends with</option>
                  <option value="regex">Regex</option>
                </select>
              </div>
            </div>
            
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">On No Match</label>
              <select
                value={getConfigStr('on_no_match', 'skip_chain')}
                onChange={(e) => updateConfig('on_no_match', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
              >
                <option value="skip_chain">Skip entire chain (quick exit)</option>
                <option value="go_to_llm">Go to main AI</option>
                <option value="continue">Continue to next step</option>
              </select>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                "Skip chain" is fastest - exits immediately if keywords don't match
              </p>
            </div>
            
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id={`case-sensitive-${step.id}`}
                checked={Boolean(step.config?.case_sensitive)}
                onChange={(e) => updateConfig('case_sensitive', e.target.checked)}
                className="rounded"
              />
              <label htmlFor={`case-sensitive-${step.id}`} className="text-xs text-[var(--color-text)]">
                Case sensitive
              </label>
            </div>
          </div>
        )}

        {step.type === 'go_to_llm' && (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">Input Variable</label>
              <input
                type="text"
                value={getConfigStr('input_var', '$Query')}
                onChange={(e) => updateConfig('input_var', e.target.value)}
                className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                placeholder="Variable to send to main AI"
              />
            </div>
            
            {/* Include Variables section */}
            <div>
              <label className="block text-xs text-[var(--color-text-secondary)] mb-1">
                Include Variables
                <span className="ml-1 text-[var(--color-text-secondary)] font-normal">(pass to LLM)</span>
              </label>
              <div className="space-y-1">
                {(Array.isArray(step.config?.include_vars) ? step.config.include_vars : []).map((varName: string, idx: number) => (
                  <div key={idx} className="flex gap-1">
                    <input
                      type="text"
                      value={varName}
                      onChange={(e) => {
                        const vars = [...(step.config?.include_vars as string[] || [])];
                        vars[idx] = e.target.value;
                        updateConfig('include_vars', vars);
                      }}
                      className="flex-1 px-2 py-1 rounded text-xs bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
                      placeholder="$Var[name] or name"
                    />
                    <button
                      onClick={() => {
                        const vars = [...(step.config?.include_vars as string[] || [])];
                        vars.splice(idx, 1);
                        updateConfig('include_vars', vars);
                      }}
                      className="px-2 text-red-400 hover:text-red-300 text-xs"
                    >
                      ✕
                    </button>
                  </div>
                ))}
                <button
                  onClick={() => {
                    const vars = [...(step.config?.include_vars as string[] || []), ''];
                    updateConfig('include_vars', vars);
                  }}
                  className="text-xs text-[var(--color-primary)] hover:text-[var(--color-primary-hover)]"
                >
                  + Add Variable
                </button>
              </div>
            </div>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={step.config?.include_context !== false}
                onChange={(e) => updateConfig('include_context', e.target.checked)}
                className="w-4 h-4 rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
              />
              <span className="text-xs text-[var(--color-text)]">Include accumulated context</span>
            </label>
          </div>
        )}

        {(step.type === 'filter_complete' || step.type === 'stop' || step.type === 'block') && (
          <div className="text-xs text-[var(--color-text-secondary)] italic p-2 bg-[var(--color-background)] rounded">
            {step.type === 'filter_complete' && 'Ends the filter chain and proceeds to main AI'}
            {step.type === 'stop' && 'Stops all processing, no AI response'}
            {step.type === 'block' && 'Blocks the message with an error response'}
          </div>
        )}

        {/* Add to Context - for steps that produce output */}
        {!['go_to_llm', 'filter_complete', 'stop', 'block', 'branch', 'context_insert'].includes(step.type) && (
          <div className="p-2 bg-[var(--color-background)] rounded border border-[var(--color-border)] space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={step.add_to_context || false}
                onChange={(e) => onUpdate({ add_to_context: e.target.checked })}
                className="w-4 h-4 rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
              />
              <span className="text-xs text-[var(--color-text)]">Add output to context</span>
            </label>
            {step.add_to_context && (
              <div>
                <label className="block text-[10px] text-[var(--color-text-secondary)] mb-1">Context Label</label>
                <input
                  type="text"
                  value={step.context_label || ''}
                  onChange={(e) => onUpdate({ context_label: e.target.value || undefined })}
                  className="w-full px-2 py-1 rounded text-xs bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text)]"
                  placeholder={step.name || STEP_TYPES[step.type]?.label || step.type}
                />
                <p className="text-[9px] text-[var(--color-text-secondary)] mt-1">
                  Output will be passed to main AI as context
                </p>
              </div>
            )}
          </div>
        )}

        {/* Connect to step - hidden for branch nodes which have their own outputs */}
        {step.type !== 'branch' && (
          <div>
            <label className="block text-xs text-[var(--color-text-secondary)] mb-1">
              Connect To
              {!step.jump_to_step && (
                <span className="ml-1 text-yellow-500">(no connection)</span>
              )}
            </label>
            <select
              value={step.jump_to_step || ''}
              onChange={(e) => onUpdate({ jump_to_step: e.target.value || undefined })}
              className="w-full px-2 py-1.5 rounded text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)]"
            >
              <option value="">— No connection —</option>
              {allSteps.filter(s => s.id !== step.id).map((s) => {
                const sType = STEP_TYPES[s.type];
                return (
                  <option key={s.id} value={s.id}>
                    → {sType?.label || s.type} (Step {allSteps.indexOf(s) + 1})
                  </option>
                );
              })}
            </select>
            <p className="text-[10px] text-[var(--color-text-secondary)] mt-1">
              Drag from output handle or select target step
            </p>
          </div>
        )}

        {/* Variable Inspector */}
        <VariableInspector steps={allSteps} currentStepIndex={stepIndex} />
      </div>
      
      {/* Footer */}
      <div className="p-3 border-t border-[var(--color-border)]">
        <button
          onClick={onDelete}
          className="w-full px-3 py-2 bg-red-500/20 text-red-400 rounded text-sm hover:bg-red-500/30"
        >
          Delete Step
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN FLOW EDITOR
// ============================================================================

const nodeTypes = {
  stepNode: StepNode,
  startNode: StartNode,
};

function FlowEditorInner({ definition, onChange, availableTools, filterChains }: FlowEditorProps) {
  const { fitView } = useReactFlow();
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null);
  const initializedRef = useRef(false);
  
  // Normalize steps
  const normalizedSteps = useMemo(() => {
    if (!definition?.steps || !Array.isArray(definition.steps)) {
      return [];
    }
    return definition.steps.map((step, index) => ({
      ...step,
      id: step.id || `step_${index}_${Date.now()}`,
      type: step.type || 'to_llm',
      enabled: step.enabled !== false,
      config: step.config || {},
    }));
  }, [definition?.steps]);

  // Notify parent of normalization
  useEffect(() => {
    if (!initializedRef.current && normalizedSteps.length > 0) {
      const needsNormalization = definition?.steps?.some((s, i) => !s.id || s.id !== normalizedSteps[i].id);
      if (needsNormalization) {
        onChange({ steps: normalizedSteps });
      }
      initializedRef.current = true;
    }
  }, [normalizedSteps, definition?.steps, onChange]);

  // Build nodes from steps
  const buildNodes = useCallback((steps: FilterChainStep[], selectedId: string | null) => {
    const nodes: Node[] = [
      {
        id: 'start',
        type: 'startNode',
        position: { x: 250, y: 0 },
        data: {},
        draggable: false,
      },
    ];
    
    steps.forEach((step, index) => {
      const defaultPos = { x: 250 - 100, y: 80 + index * 140 };
      nodes.push({
        id: step.id,
        type: 'stepNode',
        position: step.position || defaultPos,
        data: {
          step,
          stepIndex: index,
          selected: step.id === selectedId,
          onSelect: setSelectedStepId,
        },
      });
    });
    
    return nodes;
  }, []);

  // Build edges from steps
  const buildEdges = useCallback((steps: FilterChainStep[]) => {
    const edges: Edge[] = [];
    
    if (steps.length > 0) {
      edges.push({
        id: 'start-to-first',
        source: 'start',
        target: steps[0].id,
        markerEnd: { type: MarkerType.ArrowClosed },
        style: { stroke: '#6B7280' },
      });
    }
    
    steps.forEach((step, index) => {
      // Handle branch nodes with multiple outputs
      if (step.type === 'branch' && Array.isArray(step.config?.outputs)) {
        const outputs = step.config.outputs as Array<{ id: string; label?: string; jump_to?: string }>;
        outputs.forEach((output, outputIdx) => {
          if (output.jump_to) {
            edges.push({
              id: `branch-${step.id}-${output.id}-to-${output.jump_to}`,
              source: step.id,
              sourceHandle: output.id,
              target: output.jump_to,
              markerEnd: { type: MarkerType.ArrowClosed },
              style: { stroke: BRANCH_OUTPUT_COLORS[outputIdx % BRANCH_OUTPUT_COLORS.length], strokeWidth: 2 },
              animated: true,
              label: output.label || `Output ${outputIdx + 1}`,
              data: { deletable: true, branchOutputId: output.id },
            });
          }
        });
      } else if (step.jump_to_step) {
        // Explicit jump/connection
        edges.push({
          id: `jump-${step.id}-to-${step.jump_to_step}`,
          source: step.id,
          target: step.jump_to_step,
          markerEnd: { type: MarkerType.ArrowClosed },
          style: { stroke: '#F59E0B', strokeWidth: 2 },
          animated: true,
          data: { deletable: true },
        });
      }
      // No more implicit sequential edges - all connections must be explicit
    });
    
    return edges;
  }, []);

  // State for edge context menu
  const [edgeContextMenu, setEdgeContextMenu] = useState<{ x: number; y: number; edgeId: string } | null>(null);

  const [nodes, setNodes, onNodesChange] = useNodesState(buildNodes(normalizedSteps, selectedStepId));
  const [edges, setEdges, onEdgesChange] = useEdgesState(buildEdges(normalizedSteps));

  // Update nodes when steps or selection changes
  useEffect(() => {
    setNodes(buildNodes(normalizedSteps, selectedStepId));
    setEdges(buildEdges(normalizedSteps));
  }, [normalizedSteps, selectedStepId, buildNodes, buildEdges, setNodes, setEdges]);

  // Handle edge click - show delete menu
  const onEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    event.preventDefault();
    event.stopPropagation();
    setEdgeContextMenu({
      x: event.clientX,
      y: event.clientY,
      edgeId: edge.id,
    });
  }, []);

  // Delete an edge
  const deleteEdge = useCallback((edgeId: string) => {
    const edge = edges.find(e => e.id === edgeId);
    if (!edge) return;

    // Find the source step and clear its connection
    const newSteps = normalizedSteps.map(step => {
      if (step.id !== edge.source) return step;

      // Check if this is a branch output
      if (step.type === 'branch' && edge.sourceHandle) {
        const outputs = [...(step.config?.outputs as Array<{ id: string; jump_to?: string }> || [])];
        const outputIdx = outputs.findIndex(o => o.id === edge.sourceHandle);
        if (outputIdx >= 0) {
          outputs[outputIdx] = { ...outputs[outputIdx], jump_to: undefined };
        }
        return { ...step, config: { ...step.config, outputs } };
      }

      // Regular step - clear jump_to_step
      return { ...step, jump_to_step: undefined };
    });

    onChange({ steps: newSteps });
    setEdgeContextMenu(null);
  }, [edges, normalizedSteps, onChange]);

  // Handle node position changes - save to step when drag ends
  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    onNodesChange(changes);
  }, [onNodesChange]);

  // Save positions when drag ends (more reliable than filtering changes)
  const handleNodeDragStop = useCallback((_event: React.MouseEvent, node: Node) => {
    if (node.id === 'start') return; // Don't save start node position
    
    const newSteps = normalizedSteps.map(step => {
      if (step.id === node.id) {
        return { ...step, position: { x: node.position.x, y: node.position.y } };
      }
      return step;
    });
    
    onChange({ steps: newSteps });
  }, [normalizedSteps, onChange]);

  // Get selected step
  const selectedStep = normalizedSteps.find(s => s.id === selectedStepId) || null;
  const selectedStepIndex = selectedStep ? normalizedSteps.indexOf(selectedStep) : -1;

  // Update selected step
  const updateSelectedStep = useCallback((updates: Partial<FilterChainStep>) => {
    if (!selectedStepId) return;
    const newSteps = normalizedSteps.map(s =>
      s.id === selectedStepId ? { ...s, ...updates } : s
    );
    onChange({ steps: newSteps });
  }, [selectedStepId, normalizedSteps, onChange]);

  // Delete selected step
  const deleteSelectedStep = useCallback(() => {
    if (!selectedStepId) return;
    
    // Remove the step and clean up any connections pointing to it
    const newSteps = normalizedSteps
      .filter(s => s.id !== selectedStepId)
      .map(s => {
        // Clear jump_to_step if it pointed to deleted step
        if (s.jump_to_step === selectedStepId) {
          return { ...s, jump_to_step: undefined };
        }
        // Clear branch outputs that pointed to deleted step
        if (s.type === 'branch' && Array.isArray(s.config?.outputs)) {
          const outputs = (s.config.outputs as Array<{ id: string; jump_to?: string }>).map(o => 
            o.jump_to === selectedStepId ? { ...o, jump_to: undefined } : o
          );
          return { ...s, config: { ...s.config, outputs } };
        }
        return s;
      });
    
    onChange({ steps: newSteps });
    setSelectedStepId(null);
  }, [selectedStepId, normalizedSteps, onChange]);

  // Add new node
  const addNode = useCallback((type: string) => {
    const newId = `step_${Date.now()}`;
    const lastNode = nodes.filter(n => n.type === 'stepNode').pop();
    const yPos = lastNode ? lastNode.position.y + 140 : 80;
    
    // Initialize config based on type
    let config: StepConfig = {};
    if (type === 'branch') {
      // Branch nodes start with two outputs: If and Else
      config = {
        outputs: [
          { id: `output_${Date.now()}_1`, label: 'If True', condition: { var: '', operator: 'contains', value: '' } },
          { id: `output_${Date.now()}_2`, label: 'Else' },
        ]
      };
    }
    
    const newStep: FilterChainStep = {
      id: newId,
      type,
      enabled: true,
      config,
      position: { x: 150, y: yPos },
    };
    
    const newSteps = [...normalizedSteps, newStep];
    onChange({ steps: newSteps });
    setSelectedStepId(newId);
    
    setTimeout(() => fitView({ padding: 0.2 }), 100);
  }, [normalizedSteps, nodes, onChange, fitView]);

  // Handle edge connections
  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target) return;
    if (connection.source === 'start') return;
    
    const sourceStep = normalizedSteps.find(s => s.id === connection.source);
    
    // Handle branch node connections
    if (sourceStep?.type === 'branch' && connection.sourceHandle) {
      const newSteps = normalizedSteps.map(s => {
        if (s.id !== connection.source) return s;
        const outputs = [...(s.config?.outputs as Array<{ id: string; jump_to?: string }> || [])];
        const outputIdx = outputs.findIndex(o => o.id === connection.sourceHandle);
        if (outputIdx >= 0) {
          outputs[outputIdx] = { ...outputs[outputIdx], jump_to: connection.target! };
        }
        return { ...s, config: { ...s.config, outputs } };
      });
      onChange({ steps: newSteps });
    } else {
      // Regular step connection
      const newSteps = normalizedSteps.map(s =>
        s.id === connection.source ? { ...s, jump_to_step: connection.target } : s
      );
      onChange({ steps: newSteps });
    }
  }, [normalizedSteps, onChange]);

  return (
    <div className="w-full h-full flex">
      {/* Canvas */}
      <div className="flex-1 h-full relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={handleNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onEdgeClick={onEdgeClick}
          onNodeDragStop={handleNodeDragStop}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          defaultEdgeOptions={{
            markerEnd: { type: MarkerType.ArrowClosed },
          }}
          style={{ backgroundColor: 'var(--color-background)' }}
          onPaneClick={() => { setSelectedStepId(null); setEdgeContextMenu(null); }}
        >
          <Background color="var(--color-border)" gap={20} />
          <Controls
            style={{
              backgroundColor: 'var(--color-surface)',
              borderColor: 'var(--color-border)',
            }}
          />
          <Panel position="top-left">
            <NodePalette onAddNode={addNode} />
          </Panel>
        </ReactFlow>
        
        {/* Edge Context Menu */}
        {edgeContextMenu && (
          <div
            className="fixed z-50 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-lg py-1"
            style={{ left: edgeContextMenu.x, top: edgeContextMenu.y }}
          >
            <button
              onClick={() => deleteEdge(edgeContextMenu.edgeId)}
              className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/20 flex items-center gap-2"
            >
              <span>🗑️</span> Delete Connection
            </button>
            <button
              onClick={() => setEdgeContextMenu(null)}
              className="w-full px-4 py-2 text-left text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-background)]"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
      
      {/* Config Panel */}
      <div className="w-80 h-full border-l border-[var(--color-border)] bg-[var(--color-surface)]">
        <ConfigPanel
          step={selectedStep}
          stepIndex={selectedStepIndex}
          allSteps={normalizedSteps}
          onUpdate={updateSelectedStep}
          onDelete={deleteSelectedStep}
          availableTools={availableTools}
          filterChains={filterChains}
        />
      </div>
    </div>
  );
}

// ============================================================================
// EXPORTED COMPONENT
// ============================================================================

export default function FlowEditor(props: FlowEditorProps) {
  return (
    <ReactFlowProvider>
      <FlowEditorInner {...props} />
    </ReactFlowProvider>
  );
}
