import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import type { Artifact } from '../types';
import { groupArtifactsByFilename, type ArtifactGroup } from '../lib/artifacts';
import { formatFileSize, formatMessageTime } from '../lib/formatters';
import api from '../lib/api';

interface ArtifactsPanelProps {
  artifacts: Artifact[];
  selectedArtifact: Artifact | null;
  onSelect: (artifact: Artifact | null) => void;
  onClose: () => void;
}

type View = 'list' | 'versions' | 'detail';
type PreviewMode = 'preview' | 'page'; // 'page' is document-style view for markdown

export default function ArtifactsPanel({ 
  artifacts, 
  selectedArtifact, 
  onSelect, 
  onClose 
}: ArtifactsPanelProps) {
  console.log('[ArtifactsPanel] RENDERING with', artifacts.length, 'artifacts');
  const [view, setView] = useState<View>('list');
  const [selectedGroup, setSelectedGroup] = useState<ArtifactGroup | null>(null);
  const [activeTab, setActiveTab] = useState<'preview' | 'code' | 'signatures'>('preview');
  const [previewMode, setPreviewMode] = useState<PreviewMode>('preview');
  const [copied, setCopied] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [isExportingPng, setIsExportingPng] = useState(false);
  const [currentPath, setCurrentPath] = useState<string[]>([]);  // NC-0.8.0.7: Folder navigation
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const fullscreenIframeRef = useRef<HTMLIFrameElement>(null);
  const mermaidRef = useRef<HTMLDivElement>(null);
  
  // Group artifacts by filename with versions
  const artifactGroups = useMemo(() => 
    groupArtifactsByFilename(artifacts), 
    [artifacts]
  );
  
  // Count of unique files
  const uniqueFileCount = artifactGroups.length;
  
  // NC-0.8.0.7: Compute folder structure for current path
  const { folders, files } = useMemo(() => {
    const currentPathStr = currentPath.join('/');
    const foldersSet = new Set<string>();
    const filesInPath: ArtifactGroup[] = [];
    
    for (const group of artifactGroups) {
      const filepath = group.filename;
      
      // Check if this file is under the current path
      if (currentPathStr && !filepath.startsWith(currentPathStr + '/') && filepath !== currentPathStr) {
        continue;
      }
      
      // Get the remaining path after current path
      const remainingPath = currentPathStr 
        ? filepath.slice(currentPathStr.length + 1) 
        : filepath;
      
      if (!remainingPath) continue;
      
      const parts = remainingPath.split('/');
      
      if (parts.length === 1) {
        // This is a file in the current folder
        filesInPath.push(group);
      } else {
        // This is in a subfolder - add the immediate subfolder
        foldersSet.add(parts[0]);
      }
    }
    
    return {
      folders: Array.from(foldersSet).sort(),
      files: filesInPath,
    };
  }, [artifactGroups, currentPath]);
  
  // Reset view when artifacts change significantly
  useEffect(() => {
    if (artifactGroups.length === 0) {
      setView('list');
      setSelectedGroup(null);
      setCurrentPath([]);
      onSelect(null);
    }
  }, [artifactGroups.length, onSelect]);
  
  // Update activeTab when artifact changes
  useEffect(() => {
    if (selectedArtifact) {
      setActiveTab(canPreview(selectedArtifact) ? 'preview' : 'code');
      setPreviewMode('preview');
    }
  }, [selectedArtifact?.id]);
  
  // Handle escape key for fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFullscreen]);
  
  const canPreview = (art: Artifact): boolean => {
    return ['html', 'react', 'svg', 'markdown', 'mermaid'].includes(art.type);
  };
  
  const copyToClipboard = async () => {
    if (!selectedArtifact) return;
    await navigator.clipboard.writeText(selectedArtifact.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const downloadArtifact = async () => {
    if (!selectedArtifact) return;
    
    const ext = getFileExtension(selectedArtifact);
    const filename = selectedArtifact.filename || `${selectedArtifact.title.replace(/\s+/g, '_')}.${ext}`;
    
    // Handle image downloads differently
    if (selectedArtifact.type === 'image') {
      const imgSrc = selectedArtifact.imageData?.url || selectedArtifact.imageData?.base64 || selectedArtifact.content;
      
      if (imgSrc.startsWith('http') || imgSrc.startsWith('/')) {
        // URL - fetch and download
        try {
          const response = await fetch(imgSrc);
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = filename.includes('/') ? filename.split('/').pop()! : filename;
          a.click();
          URL.revokeObjectURL(url);
        } catch (err) {
          console.error('Failed to download image:', err);
        }
      } else {
        // Base64 - convert and download
        const base64Data = imgSrc.startsWith('data:') ? imgSrc : `data:image/png;base64,${imgSrc}`;
        const a = document.createElement('a');
        a.href = base64Data;
        a.download = filename.includes('/') ? filename.split('/').pop()! : filename;
        a.click();
      }
      return;
    }
    
    const blob = new Blob([selectedArtifact.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename.includes('/') ? filename.split('/').pop()! : filename;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  // Convert markdown to PDF and download via API
  const downloadPdf = useCallback(async () => {
    if (!selectedArtifact || selectedArtifact.type !== 'markdown') return;
    
    setIsGeneratingPdf(true);
    
    try {
      // Get filename
      const filename = selectedArtifact.filename 
        ? selectedArtifact.filename.replace(/\.md$/i, '.pdf').split('/').pop()
        : `${selectedArtifact.title.replace(/\s+/g, '_')}.pdf`;
      
      // Call backend API to convert markdown to PDF
      const response = await api.post('/utils/markdown-to-pdf', {
        content: selectedArtifact.content,
        filename: filename,
        title: selectedArtifact.title
      }, {
        responseType: 'blob'
      });
      
      // Download the PDF
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename || 'document.pdf';
      a.click();
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Failed to generate PDF:', error);
    } finally {
      setIsGeneratingPdf(false);
    }
  }, [selectedArtifact]);
  
  // Export mermaid diagram as PNG with transparency
  const exportMermaidPng = useCallback(async () => {
    if (!selectedArtifact || selectedArtifact.type !== 'mermaid') return;
    if (!iframeRef.current) return;
    
    setIsExportingPng(true);
    
    try {
      // Get the iframe's document
      const iframeDoc = iframeRef.current.contentDocument || iframeRef.current.contentWindow?.document;
      if (!iframeDoc) throw new Error('Cannot access iframe');
      
      // Find the mermaid SVG
      const svg = iframeDoc.querySelector('svg');
      if (!svg) throw new Error('No SVG found');
      
      // Clone and prepare SVG for export
      const svgClone = svg.cloneNode(true) as SVGSVGElement;
      
      // Ensure SVG has explicit dimensions
      const bbox = svg.getBBox();
      const width = Math.ceil(bbox.width + 40);
      const height = Math.ceil(bbox.height + 40);
      svgClone.setAttribute('width', String(width));
      svgClone.setAttribute('height', String(height));
      svgClone.setAttribute('viewBox', `${bbox.x - 20} ${bbox.y - 20} ${width} ${height}`);
      
      // Make background transparent by removing any fill
      svgClone.style.background = 'transparent';
      
      // Serialize SVG to string
      const serializer = new XMLSerializer();
      const svgString = serializer.serializeToString(svgClone);
      const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
      const svgUrl = URL.createObjectURL(svgBlob);
      
      // Create canvas and draw SVG
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = width * 2; // 2x for retina
        canvas.height = height * 2;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Cannot get canvas context');
        
        // Don't fill background - keep transparent
        ctx.scale(2, 2);
        ctx.drawImage(img, 0, 0);
        
        // Export as PNG
        canvas.toBlob((blob) => {
          if (!blob) {
            console.error('Failed to create PNG blob');
            return;
          }
          
          const filename = selectedArtifact.filename 
            ? selectedArtifact.filename.replace(/\.mmd$/i, '.png').split('/').pop()
            : `${selectedArtifact.title.replace(/\s+/g, '_')}.png`;
          
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = filename || 'diagram.png';
          a.click();
          URL.revokeObjectURL(url);
          URL.revokeObjectURL(svgUrl);
          setIsExportingPng(false);
        }, 'image/png');
      };
      
      img.onerror = () => {
        console.error('Failed to load SVG as image');
        URL.revokeObjectURL(svgUrl);
        setIsExportingPng(false);
      };
      
      img.src = svgUrl;
      
    } catch (error) {
      console.error('Failed to export PNG:', error);
      setIsExportingPng(false);
    }
  }, [selectedArtifact]);
  
  const getFileExtension = (art: Artifact): string => {
    switch (art.type) {
      case 'html': return 'html';
      case 'react': return 'jsx';
      case 'svg': return 'svg';
      case 'markdown': return 'md';
      case 'mermaid': return 'mmd';
      case 'json': return 'json';
      case 'csv': return 'csv';
      case 'code': return art.language || 'txt';
      case 'image': {
        // Try to get extension from filename or URL
        const filename = art.filename || art.imageData?.url || '';
        const ext = filename.split('.').pop()?.toLowerCase();
        if (ext && ['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(ext)) {
          return ext;
        }
        return 'png';
      }
      default: return 'txt';
    }
  };
  
  // Page view for markdown - document-style rendering
  const getPageViewContent = (art: Artifact): string => {
    // Enhanced markdown to HTML conversion
    let html = art.content
      // Code blocks (must be before inline code)
      .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => 
        `<pre class="code-block"><code class="language-${lang || 'text'}">${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>`)
      // Headers
      .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h2>$1</h2>')
      .replace(/^# (.*$)/gm, '<h1>$1</h1>')
      // Bold and italic
      .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Inline code
      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
      // Horizontal rules
      .replace(/^---$/gm, '<hr>')
      // Unordered lists
      .replace(/^\s*[-*]\s+(.*)$/gm, '<li>$1</li>')
      // Ordered lists
      .replace(/^\s*\d+\.\s+(.*)$/gm, '<li class="ordered">$1</li>')
      // Blockquotes
      .replace(/^>\s+(.*)$/gm, '<blockquote>$1</blockquote>')
      // Paragraphs (double newlines)
      .replace(/\n\n/g, '</p><p>')
      // Single newlines within paragraphs
      .replace(/\n/g, '<br>');
    
    // Wrap consecutive list items
    html = html.replace(/(<li>.*?<\/li>)+/gs, '<ul>$&</ul>');
    html = html.replace(/(<li class="ordered">.*?<\/li>)+/gs, match => 
      '<ol>' + match.replace(/ class="ordered"/g, '') + '</ol>');
    
    // Merge consecutive blockquotes
    html = html.replace(/(<blockquote>.*?<\/blockquote>)+/gs, match =>
      '<blockquote>' + match.replace(/<\/?blockquote>/g, '<br>').slice(4, -4) + '</blockquote>');
    
    return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    * { box-sizing: border-box; }
    html, body { 
      margin: 0; 
      padding: 0;
      background: #f5f5f5;
      font-family: 'Georgia', 'Times New Roman', serif;
    }
    .page {
      background: white;
      width: 8.5in;
      min-height: 11in;
      margin: 20px auto;
      padding: 1in;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 { font-size: 28px; margin: 0 0 16px 0; color: #1a1a1a; border-bottom: 2px solid #333; padding-bottom: 8px; }
    h2 { font-size: 22px; margin: 24px 0 12px 0; color: #2a2a2a; }
    h3 { font-size: 18px; margin: 20px 0 10px 0; color: #3a3a3a; }
    h4 { font-size: 16px; margin: 16px 0 8px 0; color: #4a4a4a; }
    p { margin: 0 0 12px 0; line-height: 1.7; color: #333; font-size: 14px; }
    a { color: #0066cc; text-decoration: none; }
    a:hover { text-decoration: underline; }
    strong { font-weight: 700; }
    em { font-style: italic; }
    .inline-code { 
      background: #f0f0f0; 
      padding: 2px 6px; 
      border-radius: 3px; 
      font-family: 'Consolas', 'Monaco', monospace;
      font-size: 13px;
      color: #d14;
    }
    .code-block {
      background: #2d2d2d;
      color: #f8f8f2;
      padding: 16px;
      border-radius: 6px;
      overflow-x: auto;
      margin: 16px 0;
      font-family: 'Consolas', 'Monaco', monospace;
      font-size: 13px;
      line-height: 1.5;
    }
    .code-block code { background: none; padding: 0; color: inherit; }
    ul, ol { margin: 12px 0; padding-left: 24px; }
    li { margin: 6px 0; line-height: 1.6; color: #333; font-size: 14px; }
    blockquote {
      margin: 16px 0;
      padding: 12px 20px;
      border-left: 4px solid #ddd;
      background: #fafafa;
      color: #666;
      font-style: italic;
    }
    hr { border: none; border-top: 1px solid #ddd; margin: 24px 0; }
    @media print {
      html, body { background: white; }
      .page { box-shadow: none; margin: 0; width: 100%; padding: 0.5in; }
    }
  </style>
</head>
<body>
  <div class="page">
    <p>${html}</p>
  </div>
</body>
</html>`;
  };
  
  const getPreviewContent = (art: Artifact): string => {
    switch (art.type) {
      case 'html':
        return art.content;
      case 'image':
        // For images, render as a centered image preview
        const imgSrc = art.imageData?.url || art.imageData?.base64 || art.content;
        const isBase64 = imgSrc.startsWith('data:') || (!imgSrc.startsWith('http') && !imgSrc.startsWith('/'));
        // For relative URLs (starting with /), make them absolute so they work in iframe srcDoc
        const src = isBase64 && !imgSrc.startsWith('data:') 
          ? `data:image/png;base64,${imgSrc}` 
          : imgSrc.startsWith('/') 
            ? `${window.location.origin}${imgSrc}` 
            : imgSrc;
        return `
<!DOCTYPE html>
<html>
<head>
  <style>
    body{margin:0;display:flex;flex-direction:column;justify-content:center;align-items:center;min-height:100vh;padding:20px;background:#1a1a1a;box-sizing:border-box;font-family:system-ui,sans-serif;color:#e5e5e5}
    img{max-width:100%;max-height:80vh;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.5)}
    .info{margin-top:16px;text-align:center;font-size:14px;opacity:0.7}
  </style>
</head>
<body>
  <img src="${src}" alt="${art.imageData?.prompt || 'Generated image'}" />
  <div class="info">
    ${art.imageData?.width || '?'}×${art.imageData?.height || '?'} • Seed: ${art.imageData?.seed || '?'}
  </div>
</body>
</html>`;
      case 'react':
        return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>body { margin: 0; font-family: system-ui, sans-serif; }</style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    ${art.content}
    
    // Try to render the default export or Component
    const Component = typeof exports !== 'undefined' ? exports.default : 
                      typeof App !== 'undefined' ? App : 
                      () => <div>No component found</div>;
    ReactDOM.createRoot(document.getElementById('root')).render(<Component />);
  </script>
</body>
</html>`;
      case 'svg':
        return `<!DOCTYPE html><html><body style="margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background:#1a1a1a">${art.content}</body></html>`;
      case 'markdown':
        const html = art.content
          .replace(/^### (.*$)/gm, '<h3>$1</h3>')
          .replace(/^## (.*$)/gm, '<h2>$1</h2>')
          .replace(/^# (.*$)/gm, '<h1>$1</h1>')
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          .replace(/\*(.*?)\*/g, '<em>$1</em>')
          .replace(/`(.*?)`/g, '<code>$1</code>')
          .replace(/\n/g, '<br>');
        return `<!DOCTYPE html><html><head><style>body{font-family:system-ui;padding:20px;max-width:800px;margin:0 auto;color:#e5e5e5;background:#1a1a1a}code{background:#333;padding:2px 6px;border-radius:4px}h1,h2,h3{color:#fff}</style></head><body>${html}</body></html>`;
      case 'mermaid':
        return `
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body{margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;padding:20px;background:#1a1a1a;box-sizing:border-box}
    .mermaid{color:#e5e5e5}
    .mermaid svg{max-width:100%;height:auto}
  </style>
</head>
<body>
  <pre class="mermaid">${art.content}</pre>
  <script>
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      securityLevel: 'loose',
      flowchart: { useMaxWidth: true, htmlLabels: true }
    });
  </script>
</body>
</html>`;
      default:
        return '';
    }
  };
  
  const getLanguageLabel = (art: Artifact): string => {
    if (art.language) return art.language.toUpperCase();
    switch (art.type) {
      case 'html': return 'HTML';
      case 'react': return 'React JSX';
      case 'svg': return 'SVG';
      case 'markdown': return 'Markdown';
      case 'mermaid': return 'Mermaid';
      case 'json': return 'JSON';
      case 'csv': return 'CSV';
      case 'image': return 'Image';
      default: return 'Code';
    }
  };
  
  // File type icons as simple inline SVG components
  const TypeIcon = ({ type }: { type: Artifact['type'] }) => {
    const iconClass = "w-4 h-4 text-[var(--color-text-secondary)]";
    switch (type) {
      case 'html':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" /></svg>;
      case 'react':
        return <svg className={iconClass} viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="2.5"/><ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="currentColor" strokeWidth="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="currentColor" strokeWidth="1.5" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="currentColor" strokeWidth="1.5" transform="rotate(120 12 12)"/></svg>;
      case 'svg':
      case 'image':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>;
      case 'markdown':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>;
      case 'mermaid':
      case 'csv':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>;
      case 'json':
        return <span className="text-xs font-mono text-[var(--color-text-secondary)]">{'{}'}</span>;
      case 'code':
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>;
      default:
        return <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>;
    }
  };
  
  // Use shared formatters from lib/formatters.ts
  // formatFileSize and formatMessageTime are imported
  
  const handleFileClick = (group: ArtifactGroup) => {
    setSelectedGroup(group);
    setView('versions');
  };
  
  const handleVersionClick = (version: Artifact) => {
    onSelect(version);
    setView('detail');
  };
  
  // NC-0.8.0.7: Folder navigation handlers
  const handleFolderClick = (folderName: string) => {
    setCurrentPath([...currentPath, folderName]);
  };
  
  const handleBreadcrumbClick = (index: number) => {
    // -1 means root (Artifacts)
    if (index === -1) {
      setCurrentPath([]);
    } else {
      setCurrentPath(currentPath.slice(0, index + 1));
    }
  };
  
  const handleBackToList = () => {
    setView('list');
    setSelectedGroup(null);
    onSelect(null);
  };
  
  const handleBackToVersions = () => {
    setView('versions');
    onSelect(null);
  };
  
  // Breadcrumb component - NC-0.8.0.7: Enhanced with folder navigation
  const Breadcrumbs = () => {
    // Show breadcrumbs when in folder, versions, or detail view
    const showFolderBreadcrumbs = currentPath.length > 0 && view === 'list';
    
    if (view === 'list' && currentPath.length === 0) return null;
    
    return (
      <div className="flex items-center gap-1 px-4 py-2 border-b border-[var(--color-border)] text-sm overflow-x-auto">
        <button 
          onClick={() => {
            if (view !== 'list') {
              handleBackToList();
            }
            setCurrentPath([]);
          }}
          className="text-[var(--color-primary)] hover:underline whitespace-nowrap flex-shrink-0"
        >
          Artifacts
        </button>
        
        {/* Folder path breadcrumbs */}
        {currentPath.map((folder, idx) => (
          <span key={idx} className="flex items-center gap-1 flex-shrink-0">
            <span className="text-[var(--color-text-secondary)]">/</span>
            {idx === currentPath.length - 1 && view === 'list' ? (
              <span className="text-[var(--color-text)] truncate max-w-[120px]">{folder}</span>
            ) : (
              <button 
                onClick={() => {
                  handleBreadcrumbClick(idx);
                  if (view !== 'list') {
                    setView('list');
                    setSelectedGroup(null);
                    onSelect(null);
                  }
                }}
                className="text-[var(--color-primary)] hover:underline truncate max-w-[120px]"
              >
                {folder}
              </button>
            )}
          </span>
        ))}
        
        {/* File/version breadcrumbs */}
        {view === 'versions' && selectedGroup && (
          <>
            <span className="text-[var(--color-text-secondary)]">/</span>
            <span className="text-[var(--color-text)] truncate max-w-[150px]">{selectedGroup.displayName}</span>
          </>
        )}
        {view === 'detail' && selectedGroup && (
          <>
            <span className="text-[var(--color-text-secondary)]">/</span>
            <button 
              onClick={handleBackToVersions}
              className="text-[var(--color-primary)] hover:underline truncate max-w-[150px]"
            >
              {selectedGroup.displayName}
            </button>
            <span className="text-[var(--color-text-secondary)]">/</span>
            <span className="text-[var(--color-text)]">
              v{selectedGroup.versions.length - selectedGroup.versions.findIndex(v => v.id === selectedArtifact?.id)}
            </span>
          </>
        )}
      </div>
    );
  };
  
  // Folder icon component
  const FolderIcon = () => (
    <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
      <path d="M10 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V8a2 2 0 00-2-2h-8l-2-2z" />
    </svg>
  );
  
  // VIEW 1: File list with folder navigation
  if (view === 'list') {
    const hasContent = folders.length > 0 || files.length > 0;
    const itemCount = folders.length + files.length;
    
    return (
      <div className="w-96 border-l border-[var(--color-border)] bg-[var(--color-surface)] flex flex-col">
        <div className="p-4 border-b border-[var(--color-border)] flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-[var(--color-text)]">Artifacts</h2>
            <p className="text-xs text-[var(--color-text-secondary)]">
              {currentPath.length > 0 
                ? `${itemCount} item${itemCount !== 1 ? 's' : ''} in /${currentPath.join('/')}`
                : `${uniqueFileCount} unique file${uniqueFileCount !== 1 ? 's' : ''}`
              }
            </p>
          </div>
          <button onClick={onClose} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text)]">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Breadcrumbs for folder navigation */}
        <Breadcrumbs />
        
        <div className="flex-1 overflow-y-auto p-4">
          {artifactGroups.length === 0 ? (
            <div className="text-center py-8 text-[var(--color-text-secondary)]">
              <svg className="w-12 h-12 mx-auto mb-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
              <p>No artifacts yet</p>
              <p className="text-sm mt-1">Artifacts will appear here when code or files are created</p>
            </div>
          ) : !hasContent ? (
            <div className="text-center py-8 text-[var(--color-text-secondary)]">
              <p>Empty folder</p>
            </div>
          ) : (
            <div className="space-y-2">
              {/* Folders first */}
              {folders.map((folderName) => (
                <button
                  key={`folder-${folderName}`}
                  onClick={() => handleFolderClick(folderName)}
                  className="w-full p-3 rounded-lg border border-[var(--color-border)] hover:border-[var(--color-primary)] text-left transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <FolderIcon />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-[var(--color-text)] truncate">{folderName}</p>
                      <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                        <span>Folder</span>
                      </div>
                    </div>
                    <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </button>
              ))}
              
              {/* Files */}
              {files.map((group) => {
                const sigCount = group.latestVersion.signatures?.length || 0;
                // Compute size from content if not provided
                const fileSize = group.latestVersion.size || (group.latestVersion.content?.length || 0);
                return (
                  <button
                    key={group.filename}
                    onClick={() => handleFileClick(group)}
                    className="w-full p-3 rounded-lg border border-[var(--color-border)] hover:border-[var(--color-primary)] text-left transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <TypeIcon type={group.type} />
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-[var(--color-text)] truncate">{group.displayName}</p>
                        <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                          <span>{group.language?.toUpperCase() || getLanguageLabel(group.latestVersion)}</span>
                          <span>•</span>
                          <span>{group.versions.length} version{group.versions.length !== 1 ? 's' : ''}</span>
                          {sigCount > 0 && (
                            <>
                              <span>•</span>
                              <span title="Code signatures (functions, classes, etc.)">{sigCount} signature{sigCount !== 1 ? 's' : ''}</span>
                            </>
                          )}
                          {fileSize > 0 && (
                            <>
                              <span>•</span>
                              <span>{formatFileSize(fileSize)}</span>
                            </>
                          )}
                        </div>
                      </div>
                      <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    );
  }
  
  // VIEW 2: Version list for a file
  if (view === 'versions' && selectedGroup) {
    return (
      <div className="w-96 border-l border-[var(--color-border)] bg-[var(--color-surface)] flex flex-col">
        <div className="p-4 border-b border-[var(--color-border)] flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-[var(--color-text)]">Artifacts</h2>
            <p className="text-xs text-[var(--color-text-secondary)]">{uniqueFileCount} unique file{uniqueFileCount !== 1 ? 's' : ''}</p>
          </div>
          <button onClick={onClose} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text)]">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <Breadcrumbs />
        
        <div className="flex-1 overflow-y-auto p-4">
          {/* File info header */}
          <div className="flex items-center gap-3 mb-4 pb-4 border-b border-[var(--color-border)]">
            <div className="w-6 h-6"><TypeIcon type={selectedGroup.type} /></div>
            <div>
              <p className="font-medium text-[var(--color-text)]">{selectedGroup.displayName}</p>
              <p className="text-xs text-[var(--color-text-secondary)]">
                {selectedGroup.language?.toUpperCase() || getLanguageLabel(selectedGroup.latestVersion)}
              </p>
            </div>
          </div>
          
          {/* Version list */}
          <div className="space-y-2">
            {selectedGroup.versions.map((version, idx) => {
              const versionNum = selectedGroup.versions.length - idx;
              const isLatest = idx === 0;
              
              return (
                <button
                  key={version.id}
                  onClick={() => handleVersionClick(version)}
                  className="w-full p-3 rounded-lg border border-[var(--color-border)] hover:border-[var(--color-primary)] text-left transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-mono text-[var(--color-primary)] w-8">v{versionNum}</span>
                    <div className="flex-1">
                      <p className="text-sm text-[var(--color-text)]">{formatMessageTime(version.created_at)}</p>
                    </div>
                    {isLatest && (
                      <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--color-success)]/10 text-[var(--color-success)]">
                        Latest
                      </span>
                    )}
                    <svg className="w-4 h-4 text-[var(--color-text-secondary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    );
  }
  
  // VIEW 3: Detail view
  if (view === 'detail' && selectedArtifact && selectedGroup) {
    const versionIndex = selectedGroup.versions.findIndex(v => v.id === selectedArtifact.id);
    const versionNum = selectedGroup.versions.length - versionIndex;
    const isLatest = versionIndex === 0;
    const isMarkdown = selectedArtifact.type === 'markdown';
    const isMermaid = selectedArtifact.type === 'mermaid';
    
    // Get the current preview content based on mode
    const getCurrentPreviewContent = () => {
      if (isMarkdown && previewMode === 'page') {
        return getPageViewContent(selectedArtifact);
      }
      return getPreviewContent(selectedArtifact);
    };
    
    return (
      <>
        {/* Fullscreen Modal */}
        {isFullscreen && (
          <div className="fixed inset-0 z-50 bg-black/90 flex flex-col">
            {/* Fullscreen Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-[var(--color-surface)] border-b border-[var(--color-border)]">
              <div className="flex items-center gap-3">
                <TypeIcon type={selectedGroup.type} />
                <span className="font-medium text-[var(--color-text)]">{selectedGroup.displayName}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-[var(--color-button)]/80 text-[var(--color-button-text)]">
                  v{versionNum}
                </span>
                {isMarkdown && (
                  <div className="flex items-center gap-1 ml-4 bg-[var(--color-background)] rounded-lg p-1">
                    <button
                      onClick={() => setPreviewMode('preview')}
                      className={`px-3 py-1 text-xs rounded transition-colors ${
                        previewMode === 'preview'
                          ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      }`}
                    >
                      Preview
                    </button>
                    <button
                      onClick={() => setPreviewMode('page')}
                      className={`px-3 py-1 text-xs rounded transition-colors ${
                        previewMode === 'page'
                          ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                      }`}
                    >
                      Page
                    </button>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                {isMarkdown && (
                  <button
                    onClick={downloadPdf}
                    disabled={isGeneratingPdf}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] text-sm hover:opacity-90 disabled:opacity-50"
                    title="Download as PDF"
                  >
                    {isGeneratingPdf ? (
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                    )}
                    PDF
                  </button>
                )}
                {isMermaid && (
                  <button
                    onClick={exportMermaidPng}
                    disabled={isExportingPng}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--color-button)] text-[var(--color-button-text)] text-sm hover:opacity-90 disabled:opacity-50"
                    title="Export as PNG"
                  >
                    {isExportingPng ? (
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    )}
                    PNG
                  </button>
                )}
                <button
                  onClick={() => setIsFullscreen(false)}
                  className="p-2 rounded-lg hover:bg-[var(--color-background)] text-[var(--color-text-secondary)]"
                  title="Exit fullscreen (Esc)"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
            
            {/* Fullscreen Content */}
            <div className="flex-1 overflow-hidden">
              <iframe
                ref={fullscreenIframeRef}
                srcDoc={getCurrentPreviewContent()}
                className="w-full h-full border-0"
                sandbox="allow-scripts allow-same-origin"
                title={`${selectedArtifact.title} - Fullscreen`}
              />
            </div>
          </div>
        )}
        
        {/* Regular Panel */}
        <div className="w-[500px] border-l border-[var(--color-border)] bg-[var(--color-surface)] flex flex-col">
          {/* Header */}
          <div className="p-3 border-b border-[var(--color-border)] flex items-center justify-between">
            <h2 className="font-semibold text-[var(--color-text)]">Artifacts</h2>
            <button onClick={onClose} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text)]">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <Breadcrumbs />
          
          {/* File/version info */}
          <div className="px-4 py-3 border-b border-[var(--color-border)]">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5"><TypeIcon type={selectedGroup.type} /></div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="font-medium text-[var(--color-text)] truncate">{selectedGroup.displayName}</p>
                  <span className={`text-xs px-1.5 py-0.5 rounded ${
                    isLatest 
                      ? 'bg-[var(--color-success)]/10 text-[var(--color-success)]'
                      : 'bg-[var(--color-text-secondary)]/10 text-[var(--color-text-secondary)]'
                  }`}>
                    v{versionNum}{isLatest ? ' (Latest)' : ''}
                  </span>
                </div>
                <p className="text-xs text-[var(--color-text-secondary)]">
                  {getLanguageLabel(selectedArtifact)} • {formatMessageTime(selectedArtifact.created_at)}
                </p>
              </div>
              <div className="flex items-center gap-1">
                {/* Fullscreen button - only for previewable artifacts */}
                {canPreview(selectedArtifact) && activeTab === 'preview' && (
                  <button
                    onClick={() => setIsFullscreen(true)}
                    className="p-2 rounded-lg hover:bg-[var(--color-background)] text-[var(--color-text-secondary)]"
                    title="Fullscreen"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                    </svg>
                  </button>
                )}
                {/* PDF download for markdown */}
                {isMarkdown && (
                  <button
                    onClick={downloadPdf}
                    disabled={isGeneratingPdf}
                    className="p-2 rounded-lg hover:bg-[var(--color-background)] text-[var(--color-text-secondary)] disabled:opacity-50"
                    title="Download as PDF"
                  >
                    {isGeneratingPdf ? (
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                    )}
                  </button>
                )}
                {/* PNG export for mermaid */}
                {isMermaid && activeTab === 'preview' && (
                  <button
                    onClick={exportMermaidPng}
                    disabled={isExportingPng}
                    className="p-2 rounded-lg hover:bg-[var(--color-background)] text-[var(--color-text-secondary)] disabled:opacity-50"
                    title="Export as PNG"
                  >
                    {isExportingPng ? (
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    )}
                  </button>
                )}
                <button
                  onClick={copyToClipboard}
                  className="p-2 rounded-lg hover:bg-[var(--color-background)] text-[var(--color-text-secondary)]"
                  title="Copy code"
                >
                  {copied ? (
                    <svg className="w-4 h-4 text-[var(--color-success)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  )}
                </button>
                <button
                  onClick={downloadArtifact}
                  className="p-2 rounded-lg hover:bg-[var(--color-background)] text-[var(--color-text-secondary)]"
                  title="Download"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
          
          {/* Tabs */}
          <div className="flex border-b border-[var(--color-border)]">
            {canPreview(selectedArtifact) && (
              <button
                onClick={() => setActiveTab('preview')}
                className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                  activeTab === 'preview'
                    ? 'text-[var(--color-text)] border-b-2 border-[var(--color-text)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                }`}
              >
                Preview
              </button>
            )}
            <button
              onClick={() => setActiveTab('code')}
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === 'code'
                  ? 'text-[var(--color-text)] border-b-2 border-[var(--color-text)]'
                  : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
              }`}
            >
              Code
            </button>
            {selectedArtifact.signatures && selectedArtifact.signatures.length > 0 && (
              <button
                onClick={() => setActiveTab('signatures')}
                className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                  activeTab === 'signatures'
                    ? 'text-[var(--color-text)] border-b-2 border-[var(--color-text)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                }`}
              >
                Signatures ({selectedArtifact.signatures.length})
              </button>
            )}
          </div>
          
          {/* Preview mode toggle for markdown */}
          {activeTab === 'preview' && isMarkdown && (
            <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--color-border)] bg-[var(--color-background)]">
              <span className="text-xs text-[var(--color-text-secondary)]">View:</span>
              <div className="flex items-center gap-1 bg-[var(--color-surface)] rounded-lg p-0.5">
                <button
                  onClick={() => setPreviewMode('preview')}
                  className={`px-3 py-1 text-xs rounded transition-colors ${
                    previewMode === 'preview'
                      ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                      : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                  }`}
                >
                  Preview
                </button>
                <button
                  onClick={() => setPreviewMode('page')}
                  className={`px-3 py-1 text-xs rounded transition-colors ${
                    previewMode === 'page'
                      ? 'bg-[var(--color-button)] text-[var(--color-button-text)]'
                      : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)]'
                  }`}
                >
                  Page
                </button>
              </div>
            </div>
          )}
          
          {/* Content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'preview' && canPreview(selectedArtifact) ? (
              <iframe
                ref={iframeRef}
                srcDoc={isMarkdown && previewMode === 'page' ? getPageViewContent(selectedArtifact) : getPreviewContent(selectedArtifact)}
                className="w-full h-full border-0 bg-white"
                sandbox="allow-scripts allow-same-origin"
                title={selectedArtifact.title}
              />
            ) : activeTab === 'signatures' && selectedArtifact.signatures ? (
              <div className="h-full overflow-auto p-4">
                <div className="space-y-3">
                  {selectedArtifact.signatures.map((sig, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-[var(--color-background)] border border-[var(--color-border)]">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                          sig.kind === 'function' || sig.kind === 'method' ? 'bg-blue-500/20 text-blue-400' :
                          sig.kind === 'class' || sig.kind === 'struct' ? 'bg-purple-500/20 text-purple-400' :
                          sig.kind === 'interface' || sig.kind === 'type' ? 'bg-green-500/20 text-green-400' :
                          sig.kind === 'import' ? 'bg-orange-500/20 text-orange-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {sig.kind}
                        </span>
                        <span className="font-medium text-[var(--color-text)]">{sig.name}</span>
                        <span className="text-xs text-[var(--color-text-secondary)] ml-auto">line {sig.line}</span>
                      </div>
                      {sig.signature && (
                        <code className="text-xs text-[var(--color-text-secondary)] font-mono block mt-1 truncate">
                          {sig.signature}
                        </code>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-full overflow-auto p-4">
                <pre className="text-sm font-mono text-[var(--color-text)] whitespace-pre-wrap break-words">
                  {selectedArtifact.content}
                </pre>
              </div>
            )}
          </div>
        </div>
      </>
    );
  }
  
  // Fallback - shouldn't happen
  return null;
}
