/**
 * GeneratedImageCard - Display generated images with action buttons
 */
import { useState, useCallback } from 'react';
import { Download, RefreshCw, Edit3, Copy, Check, Maximize2, X, Settings2 } from 'lucide-react';
import type { GeneratedImage } from '../types';
import ImageSizeSelector, { IMAGE_SIZES, findSizeByDimensions, type ImageSize } from './ImageSizeSelector';

interface GeneratedImageCardProps {
  image: GeneratedImage;
  messageId: string;
  onRetry?: (prompt: string, width?: number, height?: number, seed?: number) => void;
  onEdit?: (prompt: string) => void;
}

export default function GeneratedImageCard({ 
  image, 
  messageId,
  onRetry,
  onEdit,
}: GeneratedImageCardProps) {
  const [copied, setCopied] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSizeSelector, setShowSizeSelector] = useState(false);
  
  // Find current size or default
  const currentSize = findSizeByDimensions(image.width, image.height) || IMAGE_SIZES[0];
  const [selectedSize, setSelectedSize] = useState<ImageSize>(currentSize);
  
  // Use base64 if available (real-time), otherwise fall back to URL (persisted)
  const imageUrl = image.base64 
    ? `data:image/png;base64,${image.base64}` 
    : image.url || '';
  
  // If no image source available, don't render
  if (!imageUrl) {
    return null;
  }
  
  const handleDownload = useCallback(async () => {
    try {
      // For URL-based images, fetch and convert to blob for download
      if (image.url && !image.base64) {
        const response = await fetch(image.url);
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = `generated-${image.seed || 'image'}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(blobUrl);
      } else {
        // For base64 images, use data URL directly
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = `generated-${image.seed || 'image'}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  }, [imageUrl, image.seed, image.url, image.base64]);
  
  const handleCopyPrompt = useCallback(async () => {
    if (image.prompt) {
      await navigator.clipboard.writeText(image.prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [image.prompt]);
  
  const handleRetry = useCallback(() => {
    if (onRetry && image.prompt) {
      // Retry with new seed but same size
      onRetry(image.prompt, selectedSize.width, selectedSize.height);
    }
  }, [onRetry, image.prompt, selectedSize]);
  
  const handleRetryWithSize = useCallback((size: ImageSize) => {
    if (onRetry && image.prompt) {
      setSelectedSize(size);
      onRetry(image.prompt, size.width, size.height);
    }
  }, [onRetry, image.prompt]);
  
  const handleRetrySameSeed = useCallback(() => {
    if (onRetry && image.prompt) {
      onRetry(image.prompt, selectedSize.width, selectedSize.height, image.seed);
    }
  }, [onRetry, image.prompt, image.seed, selectedSize]);
  
  const handleEdit = useCallback(() => {
    if (onEdit && image.prompt) {
      onEdit(image.prompt);
    }
  }, [onEdit, image.prompt]);
  
  return (
    <>
      <div className="mt-3 rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
        {/* Image */}
        <div className="relative group">
          <img
            src={imageUrl}
            alt={image.prompt || 'Generated image'}
            className="w-full max-w-lg mx-auto cursor-pointer transition-transform hover:scale-[1.02]"
            onClick={() => setIsFullscreen(true)}
          />
          
          {/* Fullscreen button overlay */}
          <button
            onClick={() => setIsFullscreen(true)}
            className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/50 text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/70"
            title="View fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
        
        {/* Info and actions */}
        <div className="p-3 space-y-2">
          {/* Prompt */}
          {image.prompt && (
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium">Prompt:</span>{' '}
              <span className="italic">{image.prompt}</span>
            </div>
          )}
          
          {/* Metadata */}
          <div className="flex flex-wrap gap-2 text-xs text-gray-500 dark:text-gray-500">
            {image.width && image.height && (
              <span>{image.width}×{image.height}</span>
            )}
            {image.seed !== undefined && (
              <span>Seed: {image.seed}</span>
            )}
            {image.generation_time && (
              <span>{image.generation_time}s</span>
            )}
          </div>
          
          {/* Action buttons */}
          <div className="flex flex-wrap items-center gap-2 pt-2">
            {/* Download */}
            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
            
            {/* Retry with new seed */}
            {onRetry && (
              <button
                onClick={handleRetry}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                title="Generate again with a new random seed"
              >
                <RefreshCw className="w-4 h-4" />
                Retry
              </button>
            )}
            
            {/* Size selector toggle */}
            {onRetry && (
              <button
                onClick={() => setShowSizeSelector(!showSizeSelector)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg transition-colors
                  ${showSizeSelector 
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300' 
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                title="Change size and regenerate"
              >
                <Settings2 className="w-4 h-4" />
                Size
              </button>
            )}
            
            {/* Edit prompt */}
            {onEdit && (
              <button
                onClick={handleEdit}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                title="Edit the prompt and regenerate"
              >
                <Edit3 className="w-4 h-4" />
                Edit
              </button>
            )}
            
            {/* Copy prompt */}
            {image.prompt && (
              <button
                onClick={handleCopyPrompt}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                title="Copy prompt to clipboard"
              >
                {copied ? (
                  <>
                    <Check className="w-4 h-4 text-green-500" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    Copy Prompt
                  </>
                )}
              </button>
            )}
          </div>
          
          {/* Size selector panel */}
          {showSizeSelector && onRetry && (
            <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Select size:
                </span>
                <ImageSizeSelector
                  selectedSize={selectedSize}
                  onSizeChange={(size) => {
                    setSelectedSize(size);
                  }}
                  compact
                />
              </div>
              
              {/* Quick aspect ratio buttons */}
              <div className="flex flex-wrap gap-1">
                {['1:1', '16:9', '9:16', '4:3', '3:2', '21:9'].map((aspect) => {
                  const sizes = IMAGE_SIZES.filter(s => s.aspect === aspect);
                  const defaultSize = sizes[0];
                  if (!defaultSize) return null;
                  
                  return (
                    <button
                      key={aspect}
                      onClick={() => handleRetryWithSize(defaultSize)}
                      className={`px-2 py-1 text-xs rounded transition-colors
                        ${selectedSize.aspect === aspect
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                        }`}
                      title={`${defaultSize.width}×${defaultSize.height}`}
                    >
                      {aspect}
                    </button>
                  );
                })}
              </div>
              
              {/* Generate with selected size button */}
              <button
                onClick={handleRetry}
                className="mt-2 w-full flex items-center justify-center gap-2 px-3 py-2 text-sm rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Generate at {selectedSize.width}×{selectedSize.height}
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Fullscreen modal */}
      {isFullscreen && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setIsFullscreen(false)}
        >
          <button
            onClick={() => setIsFullscreen(false)}
            className="absolute top-4 right-4 p-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
          
          <img
            src={imageUrl}
            alt={image.prompt || 'Generated image'}
            className="max-w-full max-h-full object-contain"
            onClick={(e) => e.stopPropagation()}
          />
          
          {/* Info bar at bottom */}
          <div className="absolute bottom-4 left-4 right-4 flex justify-center">
            <div className="bg-black/70 text-white text-sm px-4 py-2 rounded-lg max-w-2xl truncate">
              {image.prompt} • {image.width}×{image.height} • Seed: {image.seed}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
