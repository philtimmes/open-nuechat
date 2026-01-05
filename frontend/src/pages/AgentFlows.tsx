import { useState, useRef, useCallback, useEffect } from 'react';
import type { FlowNode, FlowConnection, FlowNodeType, AgentFlow, LogicConfig, FilterConfig, DocumentsConfig, WebConfig, ModelRequestConfig, OutputConfig } from '../types';

// Node type definitions with metadata
const NODE_TYPES: Record<FlowNodeType, { 
  label: string; 
  icon: string; 
  color: string; 
  hasInput: boolean; 
  hasOutput: boolean;
  description: string;
}> = {
  start: { 
    label: 'Start', 
    icon: '▶', 
    color: '#22C55E', 
    hasInput: false, 
    hasOutput: true,
    description: 'Entry point - receives prompt from user or chat'
  },
  return: { 
    label: 'Return', 
    icon: '↩', 
    color: '#EF4444', 
    hasInput: true, 
    hasOutput: false,
    description: 'Returns result to calling LLM or Agent UI'
  },
  logic: { 
    label: 'Logic', 
    icon: '⚙', 
    color: '#F59E0B', 
    hasInput: true, 
    hasOutput: true,
    description: 'Boolean logic, comparisons, conditionals'
  },
  filter: { 
    label: 'Filter', 
    icon: '⊘', 
    color: '#8B5CF6', 
    hasInput: true, 
    hasOutput: true,
    description: 'Transform or reject based on patterns'
  },
  documents: { 
    label: 'Documents', 
    icon: '◧', 
    color: '#06B6D4', 
    hasInput: true, 
    hasOutput: true,
    description: 'Query Knowledge Base, links, or Google Drive'
  },
  web: { 
    label: 'Web', 
    icon: '◎', 
    color: '#3B82F6', 
    hasInput: true, 
    hasOutput: true,
    description: 'Fetch content from URLs'
  },
  model_request: { 
    label: 'Model Request', 
    icon: '◈', 
    color: '#EC4899', 
    hasInput: true, 
    hasOutput: true,
    description: 'Send to LLM with prompt template'
  },
  output: { 
    label: 'Output', 
    icon: '▷', 
    color: '#10B981', 
    hasInput: true, 
    hasOutput: true,
    description: 'Format and output data'
  },
};

// Generate unique ID
const generateId = () => `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

// Node dimensions
const NODE_WIDTH = 180;
const NODE_HEIGHT = 80;
const PIN_SIZE = 14;

export default function AgentFlows() {
  // Flow state
  const [flows, setFlows] = useState<AgentFlow[]>([]);
  const [currentFlow, setCurrentFlow] = useState<AgentFlow | null>(null);
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const [connections, setConnections] = useState<FlowConnection[]>([]);
  
  // Editing state
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [connecting, setConnecting] = useState<{ fromNodeId: string; startX: number; startY: number } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [flowName, setFlowName] = useState('New Agent Flow');
  const [showNodeConfig, setShowNodeConfig] = useState(false);
  const [showFlowList, setShowFlowList] = useState(true);
  
  // Canvas state
  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  
  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Create new flow
  const createNewFlow = () => {
    const newFlow: AgentFlow = {
      id: generateId(),
      name: 'New Agent Flow',
      nodes: [],
      connections: [],
      is_active: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    setFlows([...flows, newFlow]);
    setCurrentFlow(newFlow);
    setNodes([]);
    setConnections([]);
    setFlowName(newFlow.name);
    setSelectedNode(null);
  };

  // Load a flow
  const loadFlow = (flow: AgentFlow) => {
    setCurrentFlow(flow);
    setNodes(flow.nodes);
    setConnections(flow.connections);
    setFlowName(flow.name);
    setSelectedNode(null);
  };

  // Save current flow
  const saveFlow = () => {
    if (!currentFlow) return;
    
    const updatedFlow: AgentFlow = {
      ...currentFlow,
      name: flowName,
      nodes,
      connections,
      updated_at: new Date().toISOString(),
    };
    
    setFlows(flows.map(f => f.id === updatedFlow.id ? updatedFlow : f));
    setCurrentFlow(updatedFlow);
  };

  // Delete flow
  const deleteFlow = (flowId: string) => {
    if (!confirm('Delete this flow?')) return;
    setFlows(flows.filter(f => f.id !== flowId));
    if (currentFlow?.id === flowId) {
      setCurrentFlow(null);
      setNodes([]);
      setConnections([]);
    }
  };

  // Add node from palette
  const handleAddNode = (type: FlowNodeType) => {
    const canvasRect = canvasRef.current?.getBoundingClientRect();
    if (!canvasRect) return;
    
    const newNode: FlowNode = {
      id: generateId(),
      type,
      label: NODE_TYPES[type].label,
      position: {
        x: 300 - canvasOffset.x + Math.random() * 100,
        y: 200 - canvasOffset.y + Math.random() * 100,
      },
    };
    
    setNodes([...nodes, newNode]);
    setSelectedNode(newNode.id);
    setShowNodeConfig(true);
  };

  // Delete selected node
  const deleteSelectedNode = () => {
    if (!selectedNode) return;
    setNodes(nodes.filter(n => n.id !== selectedNode));
    setConnections(connections.filter(c => c.fromNodeId !== selectedNode && c.toNodeId !== selectedNode));
    setSelectedNode(null);
    setShowNodeConfig(false);
  };

  // Start dragging a node
  const handleNodeMouseDown = (e: React.MouseEvent, nodeId: string) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;
    
    setDraggingNode(nodeId);
    setSelectedNode(nodeId);
    setDragOffset({
      x: e.clientX - node.position.x - canvasOffset.x,
      y: e.clientY - node.position.y - canvasOffset.y,
    });
  };

  // Handle mouse move
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvasRect = canvasRef.current?.getBoundingClientRect();
    if (!canvasRect) return;
    
    setMousePos({
      x: e.clientX - canvasRect.left,
      y: e.clientY - canvasRect.top,
    });
    
    if (draggingNode) {
      setNodes(nodes.map(n => 
        n.id === draggingNode 
          ? { ...n, position: { 
              x: e.clientX - dragOffset.x - canvasOffset.x, 
              y: e.clientY - dragOffset.y - canvasOffset.y 
            }} 
          : n
      ));
    }
    
    if (isPanning) {
      setCanvasOffset({
        x: canvasOffset.x + (e.clientX - panStart.x),
        y: canvasOffset.y + (e.clientY - panStart.y),
      });
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  }, [draggingNode, dragOffset, canvasOffset, nodes, isPanning, panStart]);

  // Handle mouse up
  const handleMouseUp = () => {
    setDraggingNode(null);
    setConnecting(null);
    setIsPanning(false);
  };

  // Start connection from output pin
  const handleOutputPinMouseDown = (e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;
    
    const canvasRect = canvasRef.current?.getBoundingClientRect();
    if (!canvasRect) return;
    
    setConnecting({
      fromNodeId: nodeId,
      startX: node.position.x + canvasOffset.x + NODE_WIDTH,
      startY: node.position.y + canvasOffset.y + NODE_HEIGHT / 2,
    });
  };

  // Complete connection to input pin
  const handleInputPinMouseUp = (e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    if (!connecting) return;
    
    // Don't connect to self
    if (connecting.fromNodeId === nodeId) {
      setConnecting(null);
      return;
    }
    
    // Don't create duplicate connections
    const exists = connections.some(
      c => c.fromNodeId === connecting.fromNodeId && c.toNodeId === nodeId
    );
    if (exists) {
      setConnecting(null);
      return;
    }
    
    setConnections([...connections, {
      id: generateId(),
      fromNodeId: connecting.fromNodeId,
      toNodeId: nodeId,
    }]);
    setConnecting(null);
  };

  // Delete connection
  const handleConnectionClick = (connectionId: string) => {
    setConnections(connections.filter(c => c.id !== connectionId));
  };

  // Canvas panning
  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
    } else if (e.button === 0 && e.target === canvasRef.current) {
      setSelectedNode(null);
      setShowNodeConfig(false);
    }
  };

  // Get connection path (bezier curve)
  const getConnectionPath = (fromNode: FlowNode, toNode: FlowNode) => {
    const startX = fromNode.position.x + canvasOffset.x + NODE_WIDTH;
    const startY = fromNode.position.y + canvasOffset.y + NODE_HEIGHT / 2;
    const endX = toNode.position.x + canvasOffset.x;
    const endY = toNode.position.y + canvasOffset.y + NODE_HEIGHT / 2;
    
    const midX = (startX + endX) / 2;
    const cpOffset = Math.min(Math.abs(endX - startX) / 2, 100);
    
    return `M ${startX} ${startY} C ${startX + cpOffset} ${startY}, ${endX - cpOffset} ${endY}, ${endX} ${endY}`;
  };

  // Update node config
  const updateNodeConfig = (nodeId: string, config: Partial<FlowNode['config']>) => {
    setNodes(nodes.map(n => 
      n.id === nodeId ? { ...n, config: { ...n.config, ...config } as FlowNode['config'] } : n
    ));
  };

  // Get selected node
  const selectedNodeData = selectedNode ? nodes.find(n => n.id === selectedNode) : null;

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedNode && !showNodeConfig) {
          deleteSelectedNode();
        }
      }
      if (e.key === 'Escape') {
        setConnecting(null);
        setSelectedNode(null);
        setShowNodeConfig(false);
      }
      if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        saveFlow();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNode, showNodeConfig, saveFlow]);

  return (
    <div className="h-full flex bg-[var(--color-background)]">
      {/* Left Panel - Node Palette & Flow List */}
      <div className="w-64 flex-shrink-0 bg-[var(--color-surface)] border-r border-[var(--color-border)] flex flex-col">
        {/* Tabs */}
        <div className="flex border-b border-[var(--color-border)]">
          <button
            onClick={() => setShowFlowList(true)}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              showFlowList 
                ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]' 
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
            }`}
          >
            Flows
          </button>
          <button
            onClick={() => setShowFlowList(false)}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              !showFlowList 
                ? 'text-[var(--color-primary)] border-b-2 border-[var(--color-primary)]' 
                : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
            }`}
          >
            Nodes
          </button>
        </div>

        {showFlowList ? (
          // Flow List
          <div className="flex-1 overflow-y-auto p-3">
            <button
              onClick={createNewFlow}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 mb-3 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 transition-opacity text-sm font-medium"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Flow
            </button>
            
            {flows.length === 0 ? (
              <p className="text-center text-[var(--color-text-secondary)] text-sm py-8">
                No flows yet. Create one to get started!
              </p>
            ) : (
              <div className="space-y-2">
                {flows.map(flow => (
                  <div
                    key={flow.id}
                    onClick={() => loadFlow(flow)}
                    className={`group p-3 rounded-lg cursor-pointer transition-colors ${
                      currentFlow?.id === flow.id
                        ? 'bg-[var(--color-button)]/80 border border-[var(--color-border)]'
                        : 'bg-[var(--color-background)] hover:bg-[var(--color-background)]/80'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium text-[var(--color-text)] truncate">
                          {flow.name}
                        </h4>
                        <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                          {flow.nodes.length} nodes • {flow.connections.length} connections
                        </p>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteFlow(flow.id); }}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/10 text-[var(--color-text-secondary)] hover:text-[var(--color-error)] transition-all"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          // Node Palette
          <div className="flex-1 overflow-y-auto p-3">
            <p className="text-xs text-[var(--color-text-secondary)] mb-3">
              Click a node to add it to the canvas
            </p>
            <div className="space-y-2">
              {(Object.entries(NODE_TYPES) as [FlowNodeType, typeof NODE_TYPES[FlowNodeType]][]).map(([type, meta]) => (
                <button
                  key={type}
                  onClick={() => handleAddNode(type)}
                  disabled={!currentFlow}
                  className="w-full flex items-center gap-3 p-3 rounded-lg bg-[var(--color-background)] hover:bg-[var(--color-background)]/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed group"
                >
                  <span 
                    className="w-8 h-8 flex items-center justify-center rounded-lg text-white text-sm"
                    style={{ backgroundColor: meta.color }}
                  >
                    {meta.icon}
                  </span>
                  <div className="flex-1 text-left">
                    <span className="text-sm font-medium text-[var(--color-text)] block">
                      {meta.label}
                    </span>
                    <span className="text-xs text-[var(--color-text-secondary)] line-clamp-1">
                      {meta.description}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="h-14 flex items-center justify-between px-4 border-b border-[var(--color-border)] bg-[var(--color-surface)]">
          <div className="flex items-center gap-4">
            {currentFlow ? (
              <input
                type="text"
                value={flowName}
                onChange={(e) => setFlowName(e.target.value)}
                className="text-lg font-semibold bg-transparent border-none outline-none text-[var(--color-text)] focus:ring-0 w-64"
                placeholder="Flow name..."
              />
            ) : (
              <span className="text-lg font-semibold text-[var(--color-text-secondary)]">
                Select or create a flow
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            {currentFlow && (
              <>
                <button
                  onClick={saveFlow}
                  className="flex items-center gap-2 px-3 py-1.5 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 transition-opacity text-sm"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                  </svg>
                  Save
                </button>
                <button
                  onClick={() => {/* TODO: Run flow */}}
                  className="flex items-center gap-2 px-3 py-1.5 bg-[var(--color-success)] text-white rounded-lg hover:opacity-90 transition-opacity text-sm"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Test Run
                </button>
              </>
            )}
          </div>
        </div>

        {/* Canvas */}
        <div 
          ref={canvasRef}
          className="flex-1 relative overflow-hidden cursor-crosshair"
          style={{ 
            backgroundImage: 'radial-gradient(circle, var(--color-border) 1px, transparent 1px)',
            backgroundSize: '20px 20px',
            backgroundPosition: `${canvasOffset.x % 20}px ${canvasOffset.y % 20}px`,
          }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {!currentFlow ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-[var(--color-surface)] flex items-center justify-center">
                  <svg className="w-8 h-8 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-[var(--color-text)] mb-2">No Flow Selected</h3>
                <p className="text-sm text-[var(--color-text-secondary)] mb-4">
                  Create a new flow or select an existing one to start building
                </p>
                <button
                  onClick={createNewFlow}
                  className="px-4 py-2 bg-[var(--color-button)] text-[var(--color-button-text)] rounded-lg hover:opacity-90 transition-opacity text-sm"
                >
                  Create New Flow
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* SVG Layer for Connections */}
              <svg 
                ref={svgRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ zIndex: 1 }}
              >
                {/* Existing connections */}
                {connections.map(conn => {
                  const fromNode = nodes.find(n => n.id === conn.fromNodeId);
                  const toNode = nodes.find(n => n.id === conn.toNodeId);
                  if (!fromNode || !toNode) return null;
                  
                  return (
                    <g key={conn.id} className="pointer-events-auto cursor-pointer" onClick={() => handleConnectionClick(conn.id)}>
                      <path
                        d={getConnectionPath(fromNode, toNode)}
                        fill="none"
                        stroke="var(--color-border)"
                        strokeWidth="10"
                        strokeLinecap="round"
                        className="opacity-0 hover:opacity-50"
                      />
                      <path
                        d={getConnectionPath(fromNode, toNode)}
                        fill="none"
                        stroke="var(--color-primary)"
                        strokeWidth="2"
                        strokeLinecap="round"
                        className="transition-colors"
                      />
                    </g>
                  );
                })}
                
                {/* In-progress connection */}
                {connecting && (
                  <path
                    d={`M ${connecting.startX} ${connecting.startY} C ${connecting.startX + 50} ${connecting.startY}, ${mousePos.x - 50} ${mousePos.y}, ${mousePos.x} ${mousePos.y}`}
                    fill="none"
                    stroke="var(--color-primary)"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeDasharray="5,5"
                    className="opacity-60"
                  />
                )}
              </svg>

              {/* Nodes */}
              {nodes.map(node => {
                const meta = NODE_TYPES[node.type];
                const isSelected = selectedNode === node.id;
                
                return (
                  <div
                    key={node.id}
                    className={`absolute rounded-xl border-2 transition-shadow cursor-move select-none ${
                      isSelected ? 'shadow-lg shadow-[var(--color-primary)]/20' : ''
                    }`}
                    style={{
                      left: node.position.x + canvasOffset.x,
                      top: node.position.y + canvasOffset.y,
                      width: NODE_WIDTH,
                      height: NODE_HEIGHT,
                      backgroundColor: 'var(--color-surface)',
                      borderColor: isSelected ? meta.color : 'var(--color-border)',
                      zIndex: isSelected ? 20 : 10,
                    }}
                    onMouseDown={(e) => handleNodeMouseDown(e, node.id)}
                    onDoubleClick={() => setShowNodeConfig(true)}
                  >
                    {/* Node Header */}
                    <div 
                      className="px-3 py-2 rounded-t-lg flex items-center gap-2"
                      style={{ backgroundColor: meta.color + '20' }}
                    >
                      <span 
                        className="w-6 h-6 flex items-center justify-center rounded text-white text-xs"
                        style={{ backgroundColor: meta.color }}
                      >
                        {meta.icon}
                      </span>
                      <span className="text-sm font-medium text-[var(--color-text)] truncate">
                        {node.label}
                      </span>
                    </div>
                    
                    {/* Node Body */}
                    <div className="px-3 py-2 text-xs text-[var(--color-text-secondary)] truncate">
                      {node.type === 'model_request' && (node.config as ModelRequestConfig)?.prompt 
                        ? (node.config as ModelRequestConfig).prompt.substring(0, 30) + '...'
                        : meta.description.substring(0, 35) + '...'
                      }
                    </div>
                    
                    {/* Input Pin */}
                    {meta.hasInput && (
                      <div
                        className="absolute rounded-full border-2 cursor-pointer hover:scale-125 transition-transform"
                        style={{
                          left: -PIN_SIZE / 2,
                          top: NODE_HEIGHT / 2 - PIN_SIZE / 2,
                          width: PIN_SIZE,
                          height: PIN_SIZE,
                          backgroundColor: 'var(--color-background)',
                          borderColor: meta.color,
                        }}
                        onMouseUp={(e) => handleInputPinMouseUp(e, node.id)}
                      />
                    )}
                    
                    {/* Output Pin */}
                    {meta.hasOutput && (
                      <div
                        className="absolute rounded-full border-2 cursor-pointer hover:scale-125 transition-transform"
                        style={{
                          right: -PIN_SIZE / 2,
                          top: NODE_HEIGHT / 2 - PIN_SIZE / 2,
                          width: PIN_SIZE,
                          height: PIN_SIZE,
                          backgroundColor: meta.color,
                          borderColor: meta.color,
                        }}
                        onMouseDown={(e) => handleOutputPinMouseDown(e, node.id)}
                      />
                    )}
                  </div>
                );
              })}
            </>
          )}
        </div>

        {/* Help text */}
        {currentFlow && (
          <div className="h-8 flex items-center justify-center gap-4 text-xs text-[var(--color-text-secondary)] bg-[var(--color-surface)] border-t border-[var(--color-border)]">
            <span>Drag nodes to move</span>
            <span>•</span>
            <span>Drag from output pin to input pin to connect</span>
            <span>•</span>
            <span>Click connection to delete</span>
            <span>•</span>
            <span>Alt+Drag or Middle-click to pan</span>
          </div>
        )}
      </div>

      {/* Right Panel - Node Configuration */}
      {showNodeConfig && selectedNodeData && (
        <div className="w-80 flex-shrink-0 bg-[var(--color-surface)] border-l border-[var(--color-border)] flex flex-col">
          <div className="p-4 border-b border-[var(--color-border)] flex items-center justify-between">
            <h3 className="font-semibold text-[var(--color-text)]">Configure Node</h3>
            <button
              onClick={() => setShowNodeConfig(false)}
              className="p-1 rounded hover:bg-zinc-700/30 text-[var(--color-text-secondary)]"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Node Label */}
            <div>
              <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
                Label
              </label>
              <input
                type="text"
                value={selectedNodeData.label}
                onChange={(e) => setNodes(nodes.map(n => 
                  n.id === selectedNodeData.id ? { ...n, label: e.target.value } : n
                ))}
                className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
            </div>

            {/* Node Type Specific Config */}
            {selectedNodeData.type === 'logic' && (
              <LogicNodeConfig 
                config={selectedNodeData.config as LogicConfig}
                onChange={(config) => updateNodeConfig(selectedNodeData.id, config)}
              />
            )}
            
            {selectedNodeData.type === 'filter' && (
              <FilterNodeConfig 
                config={selectedNodeData.config as FilterConfig}
                onChange={(config) => updateNodeConfig(selectedNodeData.id, config)}
              />
            )}
            
            {selectedNodeData.type === 'documents' && (
              <DocumentsNodeConfig 
                config={selectedNodeData.config as DocumentsConfig}
                onChange={(config) => updateNodeConfig(selectedNodeData.id, config)}
              />
            )}
            
            {selectedNodeData.type === 'web' && (
              <WebNodeConfig 
                config={selectedNodeData.config as WebConfig}
                onChange={(config) => updateNodeConfig(selectedNodeData.id, config)}
              />
            )}
            
            {selectedNodeData.type === 'model_request' && (
              <ModelRequestNodeConfig 
                config={selectedNodeData.config as ModelRequestConfig}
                onChange={(config) => updateNodeConfig(selectedNodeData.id, config)}
              />
            )}
            
            {selectedNodeData.type === 'output' && (
              <OutputNodeConfig 
                config={selectedNodeData.config as OutputConfig}
                onChange={(config) => updateNodeConfig(selectedNodeData.id, config)}
              />
            )}

            {/* Delete Node Button */}
            <button
              onClick={deleteSelectedNode}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[var(--color-error)]/10 text-[var(--color-error)] rounded-lg hover:bg-[var(--color-error)]/20 transition-colors text-sm"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Delete Node
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Node configuration components
function LogicNodeConfig({ config, onChange }: { config?: LogicConfig; onChange: (c: LogicConfig) => void }) {
  const current = config || { operator: 'equals' as const };
  
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Operator
        </label>
        <select
          value={current.operator}
          onChange={(e) => onChange({ ...current, operator: e.target.value as LogicConfig['operator'] })}
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        >
          <option value="equals">Equals (==)</option>
          <option value="not_equals">Not Equals (!=)</option>
          <option value="contains">Contains</option>
          <option value="greater_than">Greater Than (&gt;)</option>
          <option value="less_than">Less Than (&lt;)</option>
          <option value="and">AND</option>
          <option value="or">OR</option>
          <option value="not">NOT</option>
          <option value="if">IF (conditional)</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Compare Value
        </label>
        <input
          type="text"
          value={current.value || ''}
          onChange={(e) => onChange({ ...current, value: e.target.value })}
          placeholder="Value to compare against..."
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        />
      </div>
    </>
  );
}

function FilterNodeConfig({ config, onChange }: { config?: FilterConfig; onChange: (c: FilterConfig) => void }) {
  const current = config || { type: 'replace' as const, pattern: '' };
  
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Filter Type
        </label>
        <select
          value={current.type}
          onChange={(e) => onChange({ ...current, type: e.target.value as FilterConfig['type'] })}
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        >
          <option value="replace">Replace</option>
          <option value="reject">Reject if matches</option>
          <option value="allow">Allow only if matches</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Pattern (regex)
        </label>
        <input
          type="text"
          value={current.pattern}
          onChange={(e) => onChange({ ...current, pattern: e.target.value })}
          placeholder="Pattern to match..."
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        />
      </div>
      {current.type === 'replace' && (
        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
            Replacement
          </label>
          <input
            type="text"
            value={current.replacement || ''}
            onChange={(e) => onChange({ ...current, replacement: e.target.value })}
            placeholder="Replace with..."
            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
        </div>
      )}
    </>
  );
}

function DocumentsNodeConfig({ config, onChange }: { config?: DocumentsConfig; onChange: (c: DocumentsConfig) => void }) {
  const current = config || { source: 'knowledge_base' as const };
  
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Source
        </label>
        <select
          value={current.source}
          onChange={(e) => onChange({ ...current, source: e.target.value as DocumentsConfig['source'] })}
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        >
          <option value="knowledge_base">Knowledge Base</option>
          <option value="user_link">User-provided Link</option>
          <option value="google_drive">Google Drive</option>
        </select>
      </div>
      {current.source === 'knowledge_base' && (
        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
            Knowledge Store ID
          </label>
          <input
            type="text"
            value={current.knowledgeStoreId || ''}
            onChange={(e) => onChange({ ...current, knowledgeStoreId: e.target.value })}
            placeholder="Select or enter store ID..."
            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
        </div>
      )}
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Query Template
        </label>
        <input
          type="text"
          value={current.query || ''}
          onChange={(e) => onChange({ ...current, query: e.target.value })}
          placeholder="{'{{input}}'} for dynamic query..."
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        />
        <p className="text-xs text-[var(--color-text-secondary)] mt-1">
          Use {'{{input}}'} to reference incoming data
        </p>
      </div>
    </>
  );
}

function WebNodeConfig({ config, onChange }: { config?: WebConfig; onChange: (c: WebConfig) => void }) {
  const current = config || { urlSource: 'static' as const };
  
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          URL Source
        </label>
        <select
          value={current.urlSource}
          onChange={(e) => onChange({ ...current, urlSource: e.target.value as WebConfig['urlSource'] })}
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        >
          <option value="static">Static URL</option>
          <option value="dynamic">From LLM / Input</option>
        </select>
      </div>
      {current.urlSource === 'static' && (
        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
            URL
          </label>
          <input
            type="text"
            value={current.staticUrl || ''}
            onChange={(e) => onChange({ ...current, staticUrl: e.target.value })}
            placeholder="https://example.com/api..."
            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
        </div>
      )}
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="extractContent"
          checked={current.extractContent ?? true}
          onChange={(e) => onChange({ ...current, extractContent: e.target.checked })}
          className="rounded border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-primary)]"
        />
        <label htmlFor="extractContent" className="text-sm text-[var(--color-text)]">
          Extract main content
        </label>
      </div>
    </>
  );
}

function ModelRequestNodeConfig({ config, onChange }: { config?: ModelRequestConfig; onChange: (c: ModelRequestConfig) => void }) {
  const current = config || { prompt: '' };
  
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Prompt Template
        </label>
        <textarea
          value={current.prompt}
          onChange={(e) => onChange({ ...current, prompt: e.target.value })}
          rows={5}
          placeholder="Enter your prompt. Use {{input}} for incoming data..."
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] resize-none"
        />
        <p className="text-xs text-[var(--color-text-secondary)] mt-1">
          Use {'{{input}}'} to include data from previous node
        </p>
      </div>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Model (optional)
        </label>
        <input
          type="text"
          value={current.model || ''}
          onChange={(e) => onChange({ ...current, model: e.target.value })}
          placeholder="Leave empty for default..."
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
            Temperature
          </label>
          <input
            type="number"
            min="0"
            max="2"
            step="0.1"
            value={current.temperature ?? 0.7}
            onChange={(e) => onChange({ ...current, temperature: parseFloat(e.target.value) })}
            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
            Max Tokens
          </label>
          <input
            type="number"
            min="1"
            max="32000"
            value={current.maxTokens ?? 1024}
            onChange={(e) => onChange({ ...current, maxTokens: parseInt(e.target.value) })}
            className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
        </div>
      </div>
    </>
  );
}

function OutputNodeConfig({ config, onChange }: { config?: OutputConfig; onChange: (c: OutputConfig) => void }) {
  const current = config || { format: 'text' as const };
  
  return (
    <>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Output Format
        </label>
        <select
          value={current.format || 'text'}
          onChange={(e) => onChange({ ...current, format: e.target.value as OutputConfig['format'] })}
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        >
          <option value="text">Plain Text</option>
          <option value="json">JSON</option>
          <option value="markdown">Markdown</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-[var(--color-text)] mb-1">
          Template (optional)
        </label>
        <textarea
          value={current.template || ''}
          onChange={(e) => onChange({ ...current, template: e.target.value })}
          rows={3}
          placeholder="Optional output template with {{input}}..."
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] resize-none"
        />
      </div>
    </>
  );
}
