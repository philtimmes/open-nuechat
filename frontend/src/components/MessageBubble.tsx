import React, { useState, memo, useEffect, useRef } from 'react';
import type { Message, Artifact, GeneratedImage } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import GeneratedImageCard from './GeneratedImageCard';
import api from '../lib/api';
import 'katex/dist/katex.min.css';

// Mermaid diagram component with Preview/Code toggle
// Uses iframe with CDN mermaid for consistent rendering (same as ArtifactsPanel)
const MermaidDiagram = memo(({ code }: { code: string }) => {
  const [view, setView] = useState<'preview' | 'code'>('preview');
  const [height, setHeight] = useState(200);
  const iframeId = useRef(`mermaid-iframe-${Math.random().toString(36).substr(2, 9)}`);
  
  // Get theme colors from CSS variables
  const getThemeColors = () => {
    const style = getComputedStyle(document.documentElement);
    return {
      background: style.getPropertyValue('--color-bg-primary').trim() || '#18181b',
      text: style.getPropertyValue('--color-text-primary').trim() || '#e4e4e7',
      primary: style.getPropertyValue('--color-primary').trim() || '#6366f1',
      border: style.getPropertyValue('--color-border').trim() || '#52525b',
    };
  };
  
  const colors = getThemeColors();

  // Generate iframe content with CDN mermaid (same approach as ArtifactsPanel)
  const iframeContent = `<!DOCTYPE html>
<html>
<head>
  <style>
    body { 
      margin: 0; 
      padding: 16px; 
      background: ${colors.background}; 
      display: flex;
      justify-content: center;
      min-height: fit-content;
    }
    .mermaid { 
      background: transparent;
    }
    .mermaid svg {
      max-width: 100%;
      height: auto;
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"><\/script>
</head>
<body>
  <div class="mermaid">${code}</div>
  <script>
    mermaid.initialize({
      startOnLoad: true,
      theme: 'base',
      securityLevel: 'loose',
      themeVariables: {
        background: '${colors.background}',
        primaryColor: '${colors.primary}',
        primaryTextColor: '${colors.text}',
        primaryBorderColor: '${colors.border}',
        lineColor: '${colors.text}',
        secondaryColor: '${colors.primary}22',
        tertiaryColor: '${colors.background}',
        mainBkg: '${colors.primary}33',
        nodeBorder: '${colors.border}',
        clusterBkg: '${colors.background}',
        titleColor: '${colors.text}',
        edgeLabelBackground: '${colors.background}',
        textColor: '${colors.text}',
        nodeTextColor: '${colors.text}',
      }
    });
    // Notify parent of height after render
    setTimeout(function() {
      var height = document.body.scrollHeight;
      window.parent.postMessage({ type: 'mermaid-height', id: '${iframeId.current}', height: height }, '*');
    }, 500);
  <\/script>
</body>
</html>`;

  // Listen for height messages from iframe
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'mermaid-height' && event.data.id === iframeId.current && event.data.height) {
        setHeight(Math.min(event.data.height + 20, 600)); // Cap at 600px
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  return (
    <div className="my-3 rounded-lg bg-zinc-900 overflow-hidden">
      {/* Toggle header */}
      <div className="flex items-center gap-1 px-3 py-2 bg-zinc-800 border-b border-zinc-700">
        <span className="text-xs text-zinc-400 mr-2">Mermaid</span>
        <button
          onClick={() => setView('preview')}
          className={`px-2 py-0.5 text-xs rounded transition-colors ${
            view === 'preview' 
              ? 'bg-[var(--color-primary)] text-white' 
              : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
          }`}
        >
          Preview
        </button>
        <button
          onClick={() => setView('code')}
          className={`px-2 py-0.5 text-xs rounded transition-colors ${
            view === 'code' 
              ? 'bg-[var(--color-primary)] text-white' 
              : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
          }`}
        >
          Code
        </button>
      </div>
      
      {/* Content */}
      {view === 'preview' ? (
        <div className="overflow-auto" style={{ maxHeight: '600px' }}>
          <iframe
            srcDoc={iframeContent}
            className="w-full border-0"
            style={{ height: `${height}px`, background: colors.background }}
            sandbox="allow-scripts"
            title="Mermaid diagram"
          />
        </div>
      ) : (
        <SyntaxHighlighter
          style={oneDark}
          language="mermaid"
          PreTag="div"
          customStyle={{
            margin: 0,
            borderRadius: 0,
            fontSize: '0.8125rem',
            maxHeight: '70vh',
            overflow: 'auto',
          }}
        >
          {code}
        </SyntaxHighlighter>
      )}
    </div>
  );
});

// SVG Preview component with Preview/Code toggle
// Uses iframe for consistent rendering (same as ArtifactsPanel)
const SVGPreview = memo(({ code }: { code: string }) => {
  const [view, setView] = useState<'preview' | 'code'>('preview');
  const [height, setHeight] = useState(200);
  const iframeId = useRef(`svg-iframe-${Math.random().toString(36).substr(2, 9)}`);
  
  // Get theme colors from CSS variables
  const getThemeColors = () => {
    const style = getComputedStyle(document.documentElement);
    return {
      background: style.getPropertyValue('--color-bg-primary').trim() || '#18181b',
    };
  };
  
  const colors = getThemeColors();

  // Generate iframe content for SVG
  const iframeContent = `<!DOCTYPE html>
<html>
<head>
  <style>
    body { 
      margin: 0; 
      padding: 16px; 
      background: ${colors.background}; 
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: fit-content;
    }
    svg {
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
  ${code}
  <script>
    // Notify parent of height after render
    setTimeout(function() {
      var height = document.body.scrollHeight;
      window.parent.postMessage({ type: 'svg-height', id: '${iframeId.current}', height: height }, '*');
    }, 100);
  <\/script>
</body>
</html>`;

  // Listen for height messages from iframe
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'svg-height' && event.data.id === iframeId.current && event.data.height) {
        setHeight(Math.min(event.data.height + 20, 600)); // Cap at 600px
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  return (
    <div className="my-3 rounded-lg bg-zinc-900 overflow-hidden">
      {/* Toggle header */}
      <div className="flex items-center gap-1 px-3 py-2 bg-zinc-800 border-b border-zinc-700">
        <span className="text-xs text-zinc-400 mr-2">SVG</span>
        <button
          onClick={() => setView('preview')}
          className={`px-2 py-0.5 text-xs rounded transition-colors ${
            view === 'preview' 
              ? 'bg-[var(--color-primary)] text-white' 
              : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
          }`}
        >
          Preview
        </button>
        <button
          onClick={() => setView('code')}
          className={`px-2 py-0.5 text-xs rounded transition-colors ${
            view === 'code' 
              ? 'bg-[var(--color-primary)] text-white' 
              : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
          }`}
        >
          Code
        </button>
      </div>
      
      {/* Content */}
      {view === 'preview' ? (
        <div className="overflow-auto" style={{ maxHeight: '600px' }}>
          <iframe
            srcDoc={iframeContent}
            className="w-full border-0"
            style={{ height: `${height}px`, background: colors.background }}
            sandbox="allow-scripts"
            title="SVG preview"
          />
        </div>
      ) : (
        <SyntaxHighlighter
          style={oneDark}
          language="xml"
          PreTag="div"
          customStyle={{
            margin: 0,
            borderRadius: 0,
            fontSize: '0.8125rem',
            maxHeight: '70vh',
            overflow: 'auto',
          }}
        >
          {code}
        </SyntaxHighlighter>
      )}
    </div>
  );
});

// LaTeX block component with Preview/Code toggle
const LaTeXBlock = memo(({ code }: { code: string }) => {
  const [view, setView] = useState<'preview' | 'code'>('preview');

  return (
    <div className="my-3 rounded-lg bg-zinc-900 overflow-hidden">
      {/* Toggle header */}
      <div className="flex items-center gap-1 px-3 py-2 bg-zinc-800 border-b border-zinc-700">
        <span className="text-xs text-zinc-400 mr-2">LaTeX</span>
        <button
          onClick={() => setView('preview')}
          className={`px-2 py-0.5 text-xs rounded transition-colors ${
            view === 'preview' 
              ? 'bg-[var(--color-primary)] text-white' 
              : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
          }`}
        >
          Preview
        </button>
        <button
          onClick={() => setView('code')}
          className={`px-2 py-0.5 text-xs rounded transition-colors ${
            view === 'code' 
              ? 'bg-[var(--color-primary)] text-white' 
              : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
          }`}
        >
          Code
        </button>
      </div>
      
      {/* Content */}
      {view === 'preview' ? (
        <div className="p-4 overflow-x-auto">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {`$$${code}$$`}
          </ReactMarkdown>
        </div>
      ) : (
        <SyntaxHighlighter
          style={oneDark}
          language="latex"
          PreTag="div"
          customStyle={{
            margin: 0,
            borderRadius: 0,
            fontSize: '0.8125rem',
            maxHeight: '70vh',
            overflow: 'auto',
          }}
        >
          {code}
        </SyntaxHighlighter>
      )}
    </div>
  );
});

// Pyodide loader - singleton pattern with installed packages tracking
let pyodidePromise: Promise<any> | null = null;
let installedPackages: Set<string> = new Set(['numpy', 'micropip']);

const loadPyodide = async () => {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      // @ts-ignore - Pyodide loaded from CDN
      const pyodide = await window.loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
      });
      // Pre-load common packages
      await pyodide.loadPackage(['numpy', 'micropip']);
      return pyodide;
    })();
  }
  return pyodidePromise;
};

// Install a package via micropip
const installPackage = async (packageName: string): Promise<{ success: boolean; message: string }> => {
  if (installedPackages.has(packageName.toLowerCase())) {
    return { success: true, message: `${packageName} is already installed` };
  }
  
  try {
    const pyodide = await loadPyodide();
    await pyodide.runPythonAsync(`
import micropip
await micropip.install('${packageName}')
    `);
    installedPackages.add(packageName.toLowerCase());
    return { success: true, message: `Successfully installed ${packageName}` };
  } catch (e: any) {
    return { success: false, message: e.message || `Failed to install ${packageName}` };
  }
};

// Python runner component with Code/Output/Args tabs, pip install, and error reporting
const PythonRunner = memo(({ code: initialCode, onError }: { code: string; onError?: (error: string, code: string) => void }) => {
  const [view, setView] = useState<'code' | 'output' | 'args' | 'packages'>('code');
  const [code, setCode] = useState(initialCode);
  const [args, setArgs] = useState<string>('');
  const [output, setOutput] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [informModel, setInformModel] = useState(false);
  const [hasReportedError, setHasReportedError] = useState(false);
  const lastReportedErrorRef = useRef<string>('');
  
  // Package management
  const [packageInput, setPackageInput] = useState('');
  const [isInstalling, setIsInstalling] = useState(false);
  const [packageStatus, setPackageStatus] = useState<string | null>(null);
  const [sessionPackages, setSessionPackages] = useState<string[]>([...installedPackages]);

  // Check if Pyodide script is loaded
  useEffect(() => {
    const checkPyodide = () => {
      // @ts-ignore
      if (window.loadPyodide) {
        setPyodideReady(true);
      }
    };
    
    checkPyodide();
    
    // @ts-ignore
    if (!window.loadPyodide) {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
      script.onload = checkPyodide;
      document.head.appendChild(script);
    }
  }, []);

  const handleInstallPackage = async () => {
    if (!packageInput.trim() || isInstalling) return;
    
    setIsInstalling(true);
    setPackageStatus(null);
    
    const result = await installPackage(packageInput.trim());
    setPackageStatus(result.message);
    
    if (result.success) {
      setSessionPackages([...installedPackages]);
      setPackageInput('');
    }
    
    setIsInstalling(false);
  };

  const runCode = async () => {
    if (!pyodideReady) {
      setError('Pyodide is still loading...');
      return;
    }
    
    setIsRunning(true);
    setIsLoading(true);
    setError(null);
    setOutput('');
    setView('output');
    setHasReportedError(false);

    try {
      const pyodide = await loadPyodide();
      setIsLoading(false);
      
      // Parse args into sys.argv
      const argsList = args.trim() ? args.trim().split(/\s+/) : [];
      
      // Set up sys.argv
      await pyodide.runPythonAsync(`
import sys
sys.argv = ['script.py'] + ${JSON.stringify(argsList)}
      `);
      
      // Capture stdout
      let stdout = '';
      pyodide.setStdout({
        batched: (text: string) => {
          stdout += text + '\n';
          setOutput(stdout);
        }
      });
      
      // Run the code
      const result = await pyodide.runPythonAsync(code);
      
      // If there's a return value, add it to output
      if (result !== undefined && result !== null) {
        const resultStr = String(result);
        if (resultStr && resultStr !== 'None') {
          stdout += resultStr;
          setOutput(stdout);
        }
      }
      
      if (!stdout.trim()) {
        setOutput('(No output)');
      }
    } catch (e: any) {
      setIsLoading(false);
      const errorMsg = e.message || 'Execution failed';
      setError(errorMsg);
      
      // Report error to LLM if checkbox is checked (only once per unique error)
      const errorKey = `${code}:${errorMsg}`;
      if (informModel && onError && !hasReportedError && lastReportedErrorRef.current !== errorKey) {
        setHasReportedError(true);
        lastReportedErrorRef.current = errorKey;
        onError(errorMsg, code);
      }
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="my-3 rounded-lg bg-zinc-900 overflow-hidden">
      {/* Header with tabs and run button */}
      <div className="flex items-center justify-between px-3 py-2 bg-zinc-800 border-b border-zinc-700 flex-wrap gap-2">
        <div className="flex items-center gap-1">
          <span className="text-xs text-zinc-400 mr-2">Python</span>
          {(['code', 'args', 'packages', 'output'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setView(tab)}
              className={`px-2 py-0.5 text-xs rounded transition-colors ${
                view === tab 
                  ? 'bg-[var(--color-primary)] text-white' 
                  : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          {onError && (
            <label className="flex items-center gap-1 text-xs text-zinc-400 cursor-pointer">
              <input
                type="checkbox"
                checked={informModel}
                onChange={(e) => setInformModel(e.target.checked)}
                className="w-3 h-3 rounded border-zinc-600 bg-zinc-700 text-[var(--color-primary)] focus:ring-[var(--color-primary)]"
              />
              Inform model of errors
            </label>
          )}
          <button
            onClick={runCode}
            disabled={isRunning || !pyodideReady}
            className="flex items-center gap-1 px-2 py-0.5 text-xs rounded bg-green-600 text-white hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isRunning ? (
              <>
                <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Running...
              </>
            ) : (
              <>
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
                Run
              </>
            )}
          </button>
        </div>
      </div>
      
      {/* Content */}
      {view === 'code' ? (
        <div className="relative">
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="w-full p-4 bg-zinc-900 text-zinc-100 font-mono text-sm resize-y min-h-[100px] max-h-[50vh] focus:outline-none"
            spellCheck={false}
            placeholder="# Enter Python code here..."
          />
        </div>
      ) : view === 'args' ? (
        <div className="p-4">
          <label className="block text-xs text-zinc-400 mb-2">
            Command-line arguments (space-separated, available as sys.argv[1:])
          </label>
          <input
            type="text"
            value={args}
            onChange={(e) => setArgs(e.target.value)}
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-zinc-100 font-mono text-sm focus:outline-none focus:border-[var(--color-primary)]"
            placeholder="arg1 arg2 --flag value"
          />
          <div className="mt-3 text-xs text-zinc-500">
            Example: If you enter "hello world --count 5", your code can access:
            <pre className="mt-1 p-2 bg-zinc-800 rounded">
              {`import sys
print(sys.argv)  # ['script.py', 'hello', 'world', '--count', '5']`}
            </pre>
          </div>
        </div>
      ) : view === 'packages' ? (
        <div className="p-4">
          <label className="block text-xs text-zinc-400 mb-2">
            Install packages (via micropip - PyPI packages that support Pyodide)
          </label>
          <div className="flex gap-2 mb-3">
            <input
              type="text"
              value={packageInput}
              onChange={(e) => setPackageInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleInstallPackage()}
              className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-zinc-100 font-mono text-sm focus:outline-none focus:border-[var(--color-primary)]"
              placeholder="package-name"
              disabled={isInstalling}
            />
            <button
              onClick={handleInstallPackage}
              disabled={isInstalling || !packageInput.trim()}
              className="px-3 py-2 text-xs rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isInstalling ? 'Installing...' : 'Install'}
            </button>
          </div>
          {packageStatus && (
            <div className={`text-xs mb-3 ${packageStatus.includes('Success') ? 'text-green-400' : 'text-red-400'}`}>
              {packageStatus}
            </div>
          )}
          <div className="text-xs text-zinc-400 mb-2">Installed packages (this session):</div>
          <div className="flex flex-wrap gap-1">
            {sessionPackages.map(pkg => (
              <span key={pkg} className="px-2 py-0.5 bg-zinc-700 rounded text-xs text-zinc-300">
                {pkg}
              </span>
            ))}
          </div>
          <div className="mt-3 text-xs text-zinc-500">
            Note: Not all PyPI packages work with Pyodide. Pure Python packages and those with Pyodide wheels are supported.
            <a href="https://pyodide.org/en/stable/usage/packages-in-pyodide.html" target="_blank" rel="noopener" className="text-blue-400 ml-1 hover:underline">
              See supported packages →
            </a>
          </div>
        </div>
      ) : (
        <div className="p-4 font-mono text-sm min-h-[100px] max-h-[50vh] overflow-auto">
          {isLoading ? (
            <div className="flex items-center gap-2 text-zinc-400">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Loading Python environment...
            </div>
          ) : error ? (
            <div className="text-red-400 whitespace-pre-wrap">{error}</div>
          ) : output ? (
            <div className="text-green-400 whitespace-pre-wrap">{output}</div>
          ) : (
            <div className="text-zinc-500">Click "Run" to execute the code</div>
          )}
        </div>
      )}
    </div>
  );
});

// Tool citation data
interface ToolCitation {
  id: string;
  tool_name: string;
  operation?: string;
  result_summary?: string;
  result_url?: string;
  success: boolean;
}

// Thinking tokens - fetch fresh on each component mount (with short cache)
interface ThinkingTokensState {
  begin: string;
  end: string;
  loadedAt: number;
}

let thinkingTokensCache: ThinkingTokensState | null = null;
const CACHE_TTL = 30000; // 30 second cache - allows admin changes to take effect quickly

async function fetchThinkingTokens(): Promise<{ begin: string; end: string }> {
  try {
    const res = await api.get('/utils/thinking-tokens');
    const tokens = {
      begin: res.data.think_begin_token || '',
      end: res.data.think_end_token || '',
    };
    // Debug logging
    if (tokens.begin || tokens.end) {
      console.debug('[ThinkingTokens] Loaded:', { begin: tokens.begin, end: tokens.end });
    }
    return tokens;
  } catch (e) {
    console.warn('Failed to load thinking tokens:', e);
    return { begin: '', end: '' };
  }
}

function useThinkingTokens() {
  const [tokens, setTokens] = useState<{ begin: string; end: string } | null>(null);
  
  useEffect(() => {
    let cancelled = false;
    
    async function load() {
      // Check if cache is still valid
      const now = Date.now();
      if (thinkingTokensCache && (now - thinkingTokensCache.loadedAt) < CACHE_TTL) {
        setTokens({ begin: thinkingTokensCache.begin, end: thinkingTokensCache.end });
        return;
      }
      
      // Fetch fresh tokens
      const freshTokens = await fetchThinkingTokens();
      
      if (!cancelled) {
        thinkingTokensCache = { ...freshTokens, loadedAt: now };
        setTokens(freshTokens);
      }
    }
    
    load();
    
    return () => { cancelled = true; };
  }, []);
  
  return tokens;
}

// Extract thinking blocks from content
interface ThinkingBlock {
  thinking: string;
  afterContent: string;
}

function extractThinkingBlocks(content: string, beginToken: string, endToken: string): { blocks: ThinkingBlock[]; finalContent: string } {
  if (!beginToken || !endToken) {
    return { blocks: [], finalContent: content };
  }
  
  const blocks: ThinkingBlock[] = [];
  let remaining = content;
  let result = '';
  
  while (remaining) {
    const beginIdx = remaining.indexOf(beginToken);
    if (beginIdx === -1) {
      result += remaining;
      break;
    }
    
    // Add content before the thinking block
    result += remaining.slice(0, beginIdx);
    
    const afterBegin = remaining.slice(beginIdx + beginToken.length);
    const endIdx = afterBegin.indexOf(endToken);
    
    if (endIdx === -1) {
      // No end token found - treat the rest as thinking (incomplete)
      blocks.push({ thinking: afterBegin, afterContent: '' });
      break;
    }
    
    const thinking = afterBegin.slice(0, endIdx);
    remaining = afterBegin.slice(endIdx + endToken.length);
    
    blocks.push({ thinking, afterContent: '' });
  }
  
  // Debug logging when thinking is found
  if (blocks.length > 0) {
    console.debug(`[ThinkingBlocks] Found ${blocks.length} thinking block(s)`);
  }
  
  return { blocks, finalContent: result };
}

// Collapsible tool result component
function ToolResultPanel({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false);
  
  // Parse tool result content to get summary
  let toolSummary = 'Tool Result';
  try {
    const parsed = JSON.parse(content);
    if (parsed.tool_name) {
      toolSummary = `${parsed.tool_name}`;
      if (parsed.operation) toolSummary += `: ${parsed.operation}`;
    }
  } catch {
    // Not JSON, use first 50 chars as summary
    toolSummary = content.slice(0, 50) + (content.length > 50 ? '...' : '');
  }
  
  return (
    <div className="py-2 pl-4 mx-4 max-w-3xl">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors w-full text-left"
      >
        <span className={`transform transition-transform ${expanded ? 'rotate-90' : ''}`}>
          ▶
        </span>
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        <span className="truncate">{toolSummary}</span>
      </button>
      {expanded && (
        <div className="mt-2 pl-5 border-l-2 border-[var(--color-border)]">
          <pre className="text-xs text-[var(--color-text-secondary)] whitespace-pre-wrap font-mono max-h-64 overflow-y-auto">
            {content}
          </pre>
        </div>
      )}
    </div>
  );
}

// Collapsible thinking block component
function ThinkingBlockPanel({ thinking, isStreaming }: { thinking: string; isStreaming?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div className="mb-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
      >
        <span className={`transform transition-transform ${expanded ? 'rotate-90' : ''}`}>
          ▶
        </span>
        <span className="flex items-center gap-1.5">
          🧠 {isStreaming ? 'Thinking...' : 'Thinking'}
          {isStreaming && (
            <span className="inline-block w-1.5 h-3 bg-[var(--color-primary)] animate-pulse" />
          )}
        </span>
      </button>
      {expanded && (
        <div className="mt-2 pl-5 border-l-2 border-[var(--color-border)]">
          <div className="text-sm text-[var(--color-text-secondary)] whitespace-pre-wrap opacity-75 italic">
            {thinking}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Fix nested code fences for proper markdown rendering.
 * 
 * Problem: When LLM generates a file containing code fences (like README.md),
 * the inner ``` closes the outer block prematurely.
 * 
 * Solution: Detect nested fences and increase outer fence to 4+ backticks.
 * CommonMark spec says closing fence must match opening length.
 */
function fixNestedCodeFences(content: string): string {
  const lines = content.split('\n');
  const result: string[] = [];
  
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // Only match opening fences that HAVE a language specifier (\w+, not \w*)
    // A bare ``` without language should NOT start a new outer code block
    const openMatch = line.match(/^(`{3,})(\w+)$/);
    
    if (openMatch) {
      const [, backticks, lang] = openMatch;
      const openingLength = backticks.length;
      
      // Scan ahead to find content and detect if it has nested fences
      let hasNestedFences = false;
      let depth = 1;
      let endLine = i + 1;
      
      for (let j = i + 1; j < lines.length && depth > 0; j++) {
        const l = lines[j];
        
        // Check for nested opening fence (has language)
        if (l.match(/^`{3,}\w+$/)) {
          hasNestedFences = true;
          depth++;
        }
        // Check for closing fence (no language, just backticks)
        else if (l.match(/^`{3,}\s*$/)) {
          depth--;
          if (depth === 0) {
            endLine = j;
          }
        }
      }
      
      // If we found nested fences and the outer fence is only 3 backticks,
      // increase it to 4 backticks so ReactMarkdown parses correctly
      if (hasNestedFences && openingLength === 3) {
        result.push('````' + lang);
        
        // Copy content lines
        for (let j = i + 1; j < endLine; j++) {
          result.push(lines[j]);
        }
        
        // Add closing fence with same length
        result.push('````');
        i = endLine + 1;
      } else {
        // No nested fences or already using 4+ backticks
        result.push(line);
        i++;
      }
    } else {
      result.push(line);
      i++;
    }
  }
  
  return result.join('\n');
}

// Preprocess content to extract filenames from lines before code fences
// Transforms:
//   `/path/to/file.js`
//   ```javascript
// Into:
//   ```javascript::/path/to/file.js
//
// Also converts XML-style filename tags to markdown code fences:
//   <Menu.cpp>code</Menu.cpp> → ```cpp::Menu.cpp\ncode\n```
//   <artifact=file.py>code</artifact> → ```python::file.py\ncode\n```
function preprocessContent(content: string): string {
  // First fix nested code fences
  let processed = fixNestedCodeFences(content);
  
  // Convert XML-style filename tags to markdown code fences
  // Pattern 4a: Direct filename as tag: <Menu.cpp>...</Menu.cpp>
  processed = processed.replace(
    /<([A-Za-z_][\w\-./]*\.(\w+))>\s*([\s\S]*?)\s*<\/\1>/g,
    (match, filepath, ext, code) => {
      const lang = extToMarkdownLang(ext);
      return `\n\`\`\`${lang}::${filepath}\n${code.trim()}\n\`\`\`\n`;
    }
  );
  
  // Pattern 4b: <artifact=filename> or <file=filename> or <code=filename>
  processed = processed.replace(
    /<(?:artifact|file|code)[=:]([A-Za-z_][\w\-./]*\.(\w+))>\s*([\s\S]*?)\s*<\/(?:artifact|file|code|\1)>/gi,
    (match, filepath, ext, code) => {
      const lang = extToMarkdownLang(ext);
      return `\n\`\`\`${lang}::${filepath}\n${code.trim()}\n\`\`\`\n`;
    }
  );
  
  // Pattern 4c: <artifact name="filename"> attribute style
  processed = processed.replace(
    /<(?:artifact|file|code)\s+(?:name|filename|file|path)=["']([A-Za-z_][\w\-./]*\.(\w+))["'][^>]*>\s*([\s\S]*?)\s*<\/(?:artifact|file|code)>/gi,
    (match, filepath, ext, code) => {
      const lang = extToMarkdownLang(ext);
      return `\n\`\`\`${lang}::${filepath}\n${code.trim()}\n\`\`\`\n`;
    }
  );
  
  // Match: optional backticks around filepath, newline, then code fence with language
  // Captures: 1=filepath (with or without backticks), 2=language
  const pattern = /(?:^|\n)`?([^\n`]+\.[a-zA-Z0-9]+)`?\s*\n```(\w+)/g;
  
  processed = processed.replace(pattern, (match, filepath, language) => {
    // Clean up the filepath - remove backticks if present
    const cleanPath = filepath.trim().replace(/^`|`$/g, '');
    // Only treat as filepath if it looks like one (has / or \ or is just a filename with extension)
    if (cleanPath.includes('/') || cleanPath.includes('\\') || /^\w+\.\w+$/.test(cleanPath)) {
      return `\n\`\`\`${language}::${cleanPath}`;
    }
    return match;
  });
  
  return processed;
}

// Map file extension to markdown language identifier
function extToMarkdownLang(ext: string): string {
  const map: Record<string, string> = {
    'py': 'python',
    'js': 'javascript',
    'ts': 'typescript',
    'tsx': 'tsx',
    'jsx': 'jsx',
    'cpp': 'cpp',
    'cc': 'cpp',
    'cxx': 'cpp',
    'c': 'c',
    'h': 'c',
    'hpp': 'cpp',
    'cs': 'csharp',
    'java': 'java',
    'kt': 'kotlin',
    'go': 'go',
    'rs': 'rust',
    'rb': 'ruby',
    'php': 'php',
    'swift': 'swift',
    'sh': 'bash',
    'bash': 'bash',
    'zsh': 'zsh',
    'ps1': 'powershell',
    'sql': 'sql',
    'html': 'html',
    'htm': 'html',
    'css': 'css',
    'scss': 'scss',
    'less': 'less',
    'json': 'json',
    'xml': 'xml',
    'yaml': 'yaml',
    'yml': 'yaml',
    'toml': 'toml',
    'md': 'markdown',
    'txt': 'text',
  };
  return map[ext.toLowerCase()] || ext.toLowerCase();
}

interface VersionInfo {
  current: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
}

interface MessageProps {
  message: Message;
  isStreaming?: boolean;
  toolCall?: { name: string; input: string } | null;
  toolCitations?: ToolCitation[];
  generatedImage?: GeneratedImage;
  onRetry?: () => void;
  onEdit?: (messageId: string, newContent: string) => void;
  onDelete?: (messageId: string) => void;
  onReadAloud?: (content: string) => void;
  onImageRetry?: (prompt: string, width?: number, height?: number, seed?: number) => void;
  onImageEdit?: (prompt: string) => void;
  onPythonError?: (messageId: string, error: string, code: string) => void;
  isReadingAloud?: boolean;
  isLastAssistant?: boolean;
  onArtifactClick?: (artifact: Artifact) => void;
  onBranchChange?: (branchIndex: number) => void;
  assistantName?: string;
  versionInfo?: VersionInfo;
}

// Memoized message comparison - only re-render if content actually changed
function arePropsEqual(prevProps: MessageProps, nextProps: MessageProps): boolean {
  // Always re-render streaming messages
  if (prevProps.isStreaming || nextProps.isStreaming) return false;
  
  // Check message content
  if (prevProps.message.id !== nextProps.message.id) return false;
  if (prevProps.message.content !== nextProps.message.content) return false;
  if (prevProps.message.current_branch !== nextProps.message.current_branch) return false;
  
  // Check attachments changes
  const prevAttachments = prevProps.message.attachments?.length ?? 0;
  const nextAttachments = nextProps.message.attachments?.length ?? 0;
  if (prevAttachments !== nextAttachments) return false;
  
  // Check metadata changes (especially for image generation)
  const prevMeta = prevProps.message.metadata;
  const nextMeta = nextProps.message.metadata;
  if (prevMeta?.generated_image !== nextMeta?.generated_image) return false;
  if (prevMeta?.image_generation?.status !== nextMeta?.image_generation?.status) return false;
  
  // Check other props
  if (prevProps.isLastAssistant !== nextProps.isLastAssistant) return false;
  if (prevProps.toolCall !== nextProps.toolCall) return false;
  if (prevProps.assistantName !== nextProps.assistantName) return false;
  if (prevProps.onEdit !== nextProps.onEdit) return false;
  if (prevProps.onDelete !== nextProps.onDelete) return false;
  if (prevProps.isReadingAloud !== nextProps.isReadingAloud) return false;
  
  // Check generated image - re-render when image arrives
  if (prevProps.generatedImage !== nextProps.generatedImage) return false;
  
  // Check tool citations
  const prevCitations = prevProps.toolCitations?.length ?? 0;
  const nextCitations = nextProps.toolCitations?.length ?? 0;
  if (prevCitations !== nextCitations) return false;
  
  // Check version info
  if (prevProps.versionInfo?.current !== nextProps.versionInfo?.current) return false;
  if (prevProps.versionInfo?.total !== nextProps.versionInfo?.total) return false;
  
  return true;
}

function MessageBubbleInner({ 
  message, 
  isStreaming, 
  toolCall, 
  toolCitations,
  generatedImage,
  onRetry,
  onEdit,
  onDelete,
  onReadAloud,
  onImageRetry,
  onImageEdit,
  onPythonError,
  isReadingAloud,
  isLastAssistant,
  onArtifactClick,
  onBranchChange,
  assistantName,
  versionInfo,
}: MessageProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const isTool = message.role === 'tool';
  const isAssistant = message.role === 'assistant';
  const isFileContent = message.metadata?.type === 'file_content';
  
  // Branch info
  const hasBranches = message.branches && message.branches.length > 0;
  const totalBranches = hasBranches ? message.branches!.length + 1 : 1;
  const currentBranch = message.current_branch ?? (hasBranches ? message.branches!.length : 0);
  
  const copyToClipboard = async (code: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const copyMessage = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const handlePrevBranch = () => {
    if (onBranchChange && currentBranch > 0) {
      onBranchChange(currentBranch - 1);
    }
  };
  
  const handleNextBranch = () => {
    if (onBranchChange && currentBranch < totalBranches - 1) {
      onBranchChange(currentBranch + 1);
    }
  };
  
  const handleStartEdit = () => {
    setEditContent(message.content);
    setIsEditing(true);
  };
  
  const handleCancelEdit = () => {
    setEditContent(message.content);
    setIsEditing(false);
  };
  
  const handleSaveEdit = () => {
    if (onEdit && editContent.trim() && editContent !== message.content) {
      onEdit(message.id, editContent.trim());
    }
    setIsEditing(false);
  };
  
  const handleDelete = () => {
    if (onDelete) {
      onDelete(message.id);
    }
    setShowDeleteConfirm(false);
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleCancelEdit();
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSaveEdit();
    }
  };
  
  // Find artifact references in content like [📦 Artifact: Title]
  const renderContentWithArtifacts = (content: string) => {
    const artifactPattern = /\[📦 Artifact: ([^\]]+)\]/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;
    let key = 0;
    
    while ((match = artifactPattern.exec(content)) !== null) {
      // Add text before the artifact reference
      if (match.index > lastIndex) {
        parts.push(content.slice(lastIndex, match.index));
      }
      
      // Add artifact button
      const artifactTitle = match[1];
      const artifact = message.artifacts?.find(a => a.title === artifactTitle);
      
      parts.push(
        <button
          key={`artifact-${key++}`}
          onClick={() => artifact && onArtifactClick?.(artifact)}
          className="inline-flex items-center gap-1.5 px-2 py-1 my-1 rounded-lg bg-[var(--color-button)]/80 border border-[var(--color-border)] text-[var(--color-button-text)] text-sm hover:bg-[var(--color-button)] transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
          </svg>
          <span className="font-medium">{artifactTitle}</span>
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </button>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < content.length) {
      parts.push(content.slice(lastIndex));
    }
    
    return parts.length > 0 ? parts : content;
  };
  
  if (isSystem) {
    return (
      <div className="py-2 px-4 text-center">
        <span className="text-xs text-[var(--color-text-secondary)] italic">
          {message.content}
        </span>
      </div>
    );
  }
  
  if (isTool) {
    return <ToolResultPanel content={message.content} />;
  }
  
  // Check if content has artifact references
  const hasArtifactRefs = /\[📦 Artifact: [^\]]+\]/.test(message.content);
  const contentWithoutArtifacts = hasArtifactRefs 
    ? message.content.replace(/\[📦 Artifact: [^\]]+\]/g, '') 
    : message.content;
  
  // Get thinking tokens for extracting thinking blocks
  const thinkingTokens = useThinkingTokens();
  
  // Extract thinking blocks from content
  const { blocks: thinkingBlocks, finalContent: contentAfterThinking } = 
    thinkingTokens && thinkingTokens.begin && thinkingTokens.end
      ? extractThinkingBlocks(contentWithoutArtifacts, thinkingTokens.begin, thinkingTokens.end)
      : { blocks: [], finalContent: contentWithoutArtifacts };
  
  // Preprocess content: fix nested code fences and extract filenames
  const processedContent = preprocessContent(contentAfterThinking);
  
  return (
    <div className={`group py-4 md:py-4 overflow-hidden ${isUser && !isFileContent ? 'bg-[var(--color-surface)]/30' : ''} ${isFileContent ? 'bg-blue-500/5 border-l-2 border-blue-500/50' : ''}`}>
      <div className="max-w-3xl mx-auto px-3 md:px-4 overflow-hidden">
        {/* Role label with branch navigation */}
        <div className="flex items-center gap-2 mb-2">
          <span className={`flex items-center gap-1.5 text-sm md:text-xs font-medium uppercase tracking-wide ${
            isFileContent ? 'text-blue-400' : 
            isUser ? 'text-[var(--color-primary)]' : 'text-[var(--color-secondary)]'
          }`}>
            {isFileContent && (
              <svg className="w-4 h-4 md:w-3.5 md:h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            )}
            {isFileContent ? `File: ${message.metadata?.path || 'Unknown'}` : 
             isUser ? 'You' : (assistantName || 'Assistant')}
          </span>
          {message.input_tokens != null && message.output_tokens != null && (
            <span className="text-sm md:text-xs text-[var(--color-text-secondary)]">
              · {message.input_tokens + message.output_tokens} tokens
              {message.time_to_first_token != null && message.time_to_complete != null && message.output_tokens > 0 && (
                <> · {(message.time_to_first_token / 1000).toFixed(2)}s TTFT · {(message.time_to_complete / 1000).toFixed(2)}s total · {(message.output_tokens / (message.time_to_complete / 1000)).toFixed(1)} t/s</>
              )}
            </span>
          )}
          
          {/* Branch navigation */}
          {hasBranches && totalBranches > 1 && (
            <div className="flex items-center gap-1 ml-2">
              <button
                onClick={handlePrevBranch}
                disabled={currentBranch === 0}
                className="p-0.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <span className="text-xs text-[var(--color-text-secondary)] min-w-[40px] text-center">
                {currentBranch + 1}/{totalBranches}
              </span>
              <button
                onClick={handleNextBranch}
                disabled={currentBranch === totalBranches - 1}
                className="p-0.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}
        </div>
        
        {/* Tool call indicator */}
        {toolCall && (
          <div className="mb-3 py-2 px-3 rounded border border-[var(--color-border)] bg-[var(--color-surface)]">
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Using: {toolCall.name}
            </div>
          </div>
        )}
        
        {/* Artifact references from content */}
        {hasArtifactRefs && message.artifacts && message.artifacts.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {message.artifacts.map((artifact) => (
              <button
                key={artifact.id}
                onClick={() => onArtifactClick?.(artifact)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--color-button)]/80 border border-[var(--color-border)] text-[var(--color-button-text)] text-sm hover:bg-[var(--color-button)] transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>
                <span className="font-medium">{artifact.title}</span>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </button>
            ))}
          </div>
        )}
        
        {/* Edit mode */}
        {isEditing ? (
          <div className="space-y-2">
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full min-h-[100px] p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] resize-y focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              autoFocus
              placeholder="Edit your message..."
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--color-text-secondary)]">
                Ctrl+Enter to save, Esc to cancel
              </span>
              <div className="flex gap-2">
                <button
                  onClick={handleCancelEdit}
                  className="px-3 py-1.5 text-sm rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveEdit}
                  disabled={!editContent.trim() || editContent === message.content}
                  className="px-3 py-1.5 text-sm rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Save & Create Branch
                </button>
              </div>
            </div>
          </div>
        ) : processedContent ? (
          <>
            {/* Render thinking blocks */}
            {thinkingBlocks.length > 0 && (
              <div className="mb-3">
                {thinkingBlocks.map((block, idx) => (
                  <ThinkingBlockPanel 
                    key={idx} 
                    thinking={block.thinking} 
                    isStreaming={isStreaming && idx === thinkingBlocks.length - 1}
                  />
                ))}
              </div>
            )}
            
            {isStreaming ? (
              // During streaming: plain text for performance (skip expensive markdown parsing)
              <div className="message-content-container max-w-none text-[var(--color-text)] whitespace-pre-wrap leading-relaxed">
                {processedContent}
                <span className="inline-block w-1.5 h-4 bg-[var(--color-primary)] animate-pulse ml-0.5 align-middle" />
              </div>
            ) : isUser ? (
              // User messages: render markdown for code blocks, mermaid, svg, latex
              <div className="message-content-container max-w-none text-[var(--color-text)] leading-relaxed">
                <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                  // Preserve paragraph whitespace
                  p: ({ children }) => <p className="whitespace-pre-wrap">{children}</p>,
                  code({ node, inline, className, children, ...props }: any) {
                    const match = /language-(\w+)/.exec(className || '');
                    const code = String(children).replace(/\n$/, '');
                    const language = match ? match[1].toLowerCase() : '';
                    
                    // Render mermaid diagrams
                    if (!inline && language === 'mermaid') {
                      return <MermaidDiagram code={code} />;
                    }
                    
                    // Render SVG with preview
                    if (!inline && language === 'svg') {
                      return <SVGPreview code={code} />;
                    }
                    
                    // Render LaTeX blocks
                    if (!inline && (language === 'latex' || language === 'tex' || language === 'math')) {
                      return <LaTeXBlock code={code} />;
                    }
                    
                    // Regular code blocks with syntax highlighting
                    if (!inline && match) {
                      return (
                        <div className="relative group/code my-3 overflow-hidden">
                          <SyntaxHighlighter
                            style={oneDark}
                            language={match[1]}
                            PreTag="div"
                            customStyle={{
                              margin: 0,
                              borderRadius: '0.5rem',
                              fontSize: '0.8125rem',
                            }}
                            {...props}
                          >
                            {code}
                          </SyntaxHighlighter>
                        </div>
                      );
                    }
                    
                    // Inline code
                    return (
                      <code className="px-1.5 py-0.5 bg-zinc-800 rounded text-sm font-mono" {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
                >
                  {processedContent}
                </ReactMarkdown>
              </div>
            ) : (
              // After streaming complete: full markdown rendering with CSS containment
              <div className="message-content-container max-w-none text-[var(--color-text)] leading-relaxed">
                <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                  code({ node, inline, className, children, ...props }: any) {
                    const match = /language-(\w+)/.exec(className || '');
                    const code = String(children).replace(/\n$/, '');
                    const language = match ? match[1].toLowerCase() : '';
                    
                    // Render mermaid diagrams
                    // Errors are shown to user but NOT auto-reported to LLM
                    // User can manually ask for fixes if needed
                    if (!inline && language === 'mermaid') {
                      return (
                        <MermaidDiagram 
                          code={code} 
                        />
                      );
                    }
                    
                    // Render SVG with preview
                    if (!inline && language === 'svg') {
                      return (
                        <SVGPreview 
                          code={code} 
                        />
                      );
                    }
                    
                    // Render Python with runner
                    // User can check "Inform model of errors" to report errors
                    if (!inline && (language === 'python' || language === 'py')) {
                      return (
                        <PythonRunner 
                          code={code}
                          onError={!isStreaming && onPythonError ? (error, pythonCode) => onPythonError(message.id, error, pythonCode) : undefined}
                        />
                      );
                    }
                    
                    // Render LaTeX blocks
                    if (!inline && (language === 'latex' || language === 'tex' || language === 'math')) {
                      return <LaTeXBlock code={code} />;
                    }
                    
                    // Extract filename from meta or first line comment
                    // Pattern: ```lang::/path/to/file or ```lang filename="/path/to/file"
                    let filename: string | null = null;
                    const metaMatch = className?.match(/language-\w+::(.+)/);
                    if (metaMatch) {
                      filename = metaMatch[1];
                    }
                    
                    if (!inline && match) {
                      return (
                        <div className="relative group/code my-3 overflow-hidden">
                          {/* Filename header */}
                          {filename && (
                            <div className="flex items-center justify-between px-3 py-1.5 bg-zinc-800 border-b border-zinc-700 rounded-t-lg">
                              <span className="text-xs text-zinc-400 font-mono truncate">{filename}</span>
                              <button
                                onClick={() => copyToClipboard(code)}
                                className="px-2 py-0.5 text-xs rounded bg-zinc-700 text-zinc-300 hover:text-white hover:bg-zinc-600 transition-colors"
                              >
                                {copiedCode === code ? 'Copied!' : 'Copy'}
                              </button>
                            </div>
                          )}
                          {/* Copy button for blocks without filename */}
                          {!filename && (
                            <div className="absolute right-2 top-2 opacity-0 group-hover/code:opacity-100 transition-opacity z-10">
                              <button
                                onClick={() => copyToClipboard(code)}
                                className="px-2 py-1 text-xs rounded bg-zinc-700 text-zinc-300 hover:text-white"
                              >
                                {copiedCode === code ? 'Copied!' : 'Copy'}
                              </button>
                            </div>
                          )}
                          <SyntaxHighlighter
                            style={oneDark}
                            language={match[1]}
                            PreTag="div"
                            customStyle={{
                              margin: 0,
                              borderRadius: filename ? '0 0 0.5rem 0.5rem' : '0.5rem',
                              fontSize: '0.8125rem',
                              maxHeight: '70vh',
                              overflow: 'auto',
                            }}
                            {...props}
                          >
                            {code}
                          </SyntaxHighlighter>
                        </div>
                      );
                    }
                    
                    return (
                      <code
                        className={`${className} px-1 py-0.5 rounded bg-zinc-800 text-[var(--color-accent)] text-sm`}
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  // Table components for GFM tables
                  table({ children }) {
                    return (
                      <div className="overflow-x-auto my-4 rounded-lg border border-[var(--color-border)]">
                        <table className="min-w-full divide-y divide-[var(--color-border)]">
                          {children}
                        </table>
                      </div>
                    );
                  },
                  thead({ children }) {
                    return (
                      <thead className="bg-[var(--color-surface)]">
                        {children}
                      </thead>
                    );
                  },
                  tbody({ children }) {
                    return (
                      <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-background)]">
                        {children}
                      </tbody>
                    );
                  },
                  tr({ children }) {
                    return (
                      <tr className="hover:bg-[var(--color-surface)]/50 transition-colors">
                        {children}
                      </tr>
                    );
                  },
                  th({ children }) {
                    return (
                      <th className="px-3 py-2 text-left text-xs font-semibold text-[var(--color-text)] uppercase tracking-wider whitespace-nowrap">
                        {children}
                      </th>
                    );
                  },
                  td({ children }) {
                    return (
                      <td className="px-3 py-2 text-sm text-[var(--color-text-secondary)] whitespace-normal">
                        {children}
                      </td>
                    );
                  },
                  p({ children }) {
                    return <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>;
                  },
                  ul({ children }) {
                    return <ul className="list-disc list-outside ml-4 mb-2 space-y-0.5">{children}</ul>;
                  },
                  ol({ children }) {
                    return <ol className="list-decimal list-outside ml-4 mb-2 space-y-0.5">{children}</ol>;
                  },
                  li({ children }) {
                    return <li className="text-[var(--color-text)] pl-1">{children}</li>;
                  },
                  a({ href, children }) {
                    return (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[var(--color-accent)] hover:underline"
                      >
                        {children}
                      </a>
                    );
                  },
                  blockquote({ children }) {
                    return (
                      <blockquote className="border-l-2 border-[var(--color-border)] pl-4 italic text-[var(--color-text-secondary)]">
                        {children}
                      </blockquote>
                    );
                  },
                  // Headings with proper sizes
                  h1({ children }) {
                    return <h1 className="text-2xl font-bold mt-6 mb-3 text-[var(--color-text)]">{children}</h1>;
                  },
                  h2({ children }) {
                    return <h2 className="text-xl font-bold mt-5 mb-2 text-[var(--color-text)]">{children}</h2>;
                  },
                  h3({ children }) {
                    return <h3 className="text-lg font-semibold mt-4 mb-2 text-[var(--color-text)]">{children}</h3>;
                  },
                  h4({ children }) {
                    return <h4 className="text-base font-semibold mt-3 mb-1 text-[var(--color-text)]">{children}</h4>;
                  },
                  h5({ children }) {
                    return <h5 className="text-sm font-semibold mt-2 mb-1 text-[var(--color-text)]">{children}</h5>;
                  },
                  h6({ children }) {
                    return <h6 className="text-sm font-medium mt-2 mb-1 text-[var(--color-text-secondary)]">{children}</h6>;
                  },
                  // Text formatting
                  strong({ children }) {
                    return <strong className="font-bold text-[var(--color-text)]">{children}</strong>;
                  },
                  em({ children }) {
                    return <em className="italic">{children}</em>;
                  },
                  hr() {
                    return <hr className="my-4 border-t border-[var(--color-border)]" />;
                  },
                }}
              >
                {processedContent}
              </ReactMarkdown>
            </div>
            )}
          </>
        ) : isStreaming ? (
          <span className="inline-block w-1.5 h-4 bg-[var(--color-primary)] animate-pulse" />
        ) : null}
        
        {/* Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.attachments.map((att, idx) => (
              <div key={idx}>
                {att.type === 'image' ? (
                  <img
                    src={att.url || `data:${att.mime_type};base64,${att.data}`}
                    alt={att.name}
                    className="max-w-sm max-h-64 rounded border border-[var(--color-border)]"
                  />
                ) : (
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-sm">
                    <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span className="truncate max-w-[200px] text-[var(--color-text)]">{att.name}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        
        {/* Generated Image */}
        {/* Check both prop and message metadata for generated image */}
        {(() => {
          const img = generatedImage || (message.metadata?.generated_image as GeneratedImage | undefined);
          const imageGenStatus = message.metadata?.image_generation;
          
          // Show generating indicator if status is pending/processing
          if (imageGenStatus && (imageGenStatus.status === 'pending' || imageGenStatus.status === 'processing')) {
            return (
              <div className="mt-3 p-4 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)]">
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-6 w-6 border-2 border-[var(--color-primary)] border-t-transparent"></div>
                  <div>
                    <p className="text-sm font-medium text-[var(--color-text)]">
                      Generating image...
                    </p>
                    {imageGenStatus.queue_position && imageGenStatus.queue_position > 1 && (
                      <p className="text-xs text-[var(--color-text-secondary)]">
                        Queue position: {imageGenStatus.queue_position}
                      </p>
                    )}
                    {imageGenStatus.width && imageGenStatus.height && (
                      <p className="text-xs text-[var(--color-text-secondary)]">
                        {imageGenStatus.width} × {imageGenStatus.height}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          }
          
          // Image must have either base64 or url to render
          if (!img || (!img.base64 && !img.url)) return null;
          return (
            <GeneratedImageCard
              image={img}
              messageId={message.id}
              onRetry={onImageRetry}
              onEdit={onImageEdit}
            />
          );
        })()}
        
        {/* Tool Citations */}
        {toolCitations && toolCitations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
            <p className="text-xs text-[var(--color-text-secondary)] mb-2 flex items-center gap-1">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              Sources
            </p>
            <div className="flex flex-wrap gap-2">
              {toolCitations.map((citation) => (
                <div
                  key={citation.id}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs ${
                    citation.success
                      ? 'bg-[var(--color-button)]/80 text-[var(--color-button-text)]'
                      : 'bg-red-500/10 text-red-400'
                  }`}
                >
                  <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span className="font-medium">{citation.tool_name}</span>
                  {citation.operation && citation.operation !== citation.tool_name && (
                    <span className="text-[var(--color-text-secondary)]">→ {citation.operation}</span>
                  )}
                  {citation.result_url && (
                    <a
                      href={citation.result_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="ml-1 hover:underline text-[var(--color-secondary)]"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Delete confirmation modal */}
        {showDeleteConfirm && (
          <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
            <p className="text-sm text-[var(--color-text)] mb-2">
              Delete this message and all its replies?
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-3 py-1 text-sm rounded bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="px-3 py-1 text-sm rounded bg-red-500 text-white hover:bg-red-600"
              >
                Delete Branch
              </button>
            </div>
          </div>
        )}
        
        {/* Action buttons - show for user and assistant messages */}
        {(isUser || isAssistant) && !isStreaming && !isEditing && (
          <div className="flex items-center justify-between mt-3">
            {/* Action buttons - always visible on mobile, hover-visible on desktop */}
            <div className="flex items-center gap-1 md:gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity">
              {/* Copy button */}
              <button
                onClick={copyMessage}
                className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] transition-colors touch-manipulation"
                title="Copy message"
              >
                {copied ? (
                  <svg className="w-5 h-5 md:w-4 md:h-4 text-[var(--color-success)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                )}
              </button>
              
              {/* Edit button */}
              {onEdit && (
                <button
                  onClick={handleStartEdit}
                  className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] transition-colors touch-manipulation"
                  title="Edit message (creates new branch)"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                  </svg>
                </button>
              )}
              
              {/* Regenerate button (assistant only) */}
              {isAssistant && onRetry && (
                <button
                  onClick={onRetry}
                  className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] transition-colors touch-manipulation"
                  title="Regenerate response"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              )}
              
              {/* Read Aloud button (assistant only) */}
              {isAssistant && onReadAloud && (
                <button
                  onClick={() => onReadAloud(message.content)}
                  className={`p-2 md:p-1.5 rounded transition-colors touch-manipulation ${
                    isReadingAloud 
                      ? 'text-[var(--color-primary)] bg-[var(--color-primary)]/10' 
                      : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)]'
                  }`}
                  title={isReadingAloud ? "Stop reading" : "Read aloud"}
                >
                  {isReadingAloud ? (
                    <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                    </svg>
                  )}
                </button>
              )}
              
              {/* Delete button */}
              {onDelete && (
                <button
                  onClick={() => setShowDeleteConfirm(true)}
                  className="p-2 md:p-1.5 rounded text-[var(--color-text-secondary)] hover:text-red-400 hover:bg-red-500/10 active:text-red-400 active:bg-red-500/10 transition-colors touch-manipulation"
                  title="Delete message and replies"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              )}
            </div>
            
            {/* Version pagination */}
            {versionInfo && versionInfo.total > 1 && (
              <div className="flex items-center gap-1 text-sm text-[var(--color-text-secondary)]">
                <button
                  onClick={versionInfo.onPrev}
                  disabled={versionInfo.current <= 1}
                  className="p-2 md:p-1 rounded hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors touch-manipulation"
                  title="Previous version"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <span className="px-1 min-w-[3rem] text-center">
                  {versionInfo.current}/{versionInfo.total}
                </span>
                <button
                  onClick={versionInfo.onNext}
                  disabled={versionInfo.current >= versionInfo.total}
                  className="p-2 md:p-1 rounded hover:bg-[var(--color-surface)] active:bg-[var(--color-surface)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors touch-manipulation"
                  title="Next version"
                >
                  <svg className="w-5 h-5 md:w-4 md:h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Export memoized version
const MessageBubble = memo(MessageBubbleInner, arePropsEqual);
export default MessageBubble;
export type { ToolCitation };
