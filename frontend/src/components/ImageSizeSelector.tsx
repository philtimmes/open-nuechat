/**
 * ImageSizeSelector - Select image resolution/aspect ratio for generation
 */
import { useState } from 'react';
import { ChevronDown, Check } from 'lucide-react';

export interface ImageSize {
  id: string;
  label: string;
  width: number;
  height: number;
  aspect: string;
  description?: string;
}

export const IMAGE_SIZES: ImageSize[] = [
  // Square
  { id: 'square-1024', label: '1024×1024', width: 1024, height: 1024, aspect: '1:1', description: 'Square (Default)' },
  { id: 'square-768', label: '768×768', width: 768, height: 768, aspect: '1:1', description: 'Square Medium' },
  { id: 'square-512', label: '512×512', width: 512, height: 512, aspect: '1:1', description: 'Square Small' },
  
  // Landscape 16:9
  { id: 'landscape-16-9-1280', label: '1280×720', width: 1280, height: 720, aspect: '16:9', description: 'HD Landscape' },
  { id: 'landscape-16-9-1920', label: '1920×1080', width: 1920, height: 1080, aspect: '16:9', description: 'Full HD Landscape' },
  
  // Portrait 9:16
  { id: 'portrait-9-16-720', label: '720×1280', width: 720, height: 1280, aspect: '9:16', description: 'HD Portrait' },
  { id: 'portrait-9-16-1080', label: '1080×1920', width: 1080, height: 1920, aspect: '9:16', description: 'Full HD Portrait' },
  
  // 4:3
  { id: 'landscape-4-3-1024', label: '1024×768', width: 1024, height: 768, aspect: '4:3', description: 'Standard Landscape' },
  { id: 'portrait-3-4-768', label: '768×1024', width: 768, height: 1024, aspect: '3:4', description: 'Standard Portrait' },
  
  // 3:2
  { id: 'landscape-3-2-1200', label: '1200×800', width: 1200, height: 800, aspect: '3:2', description: 'Photo Landscape' },
  { id: 'portrait-2-3-800', label: '800×1200', width: 800, height: 1200, aspect: '2:3', description: 'Photo Portrait' },
  
  // Cinematic 21:9
  { id: 'ultrawide-21-9', label: '1344×576', width: 1344, height: 576, aspect: '21:9', description: 'Ultrawide Cinematic' },
  
  // Social Media
  { id: 'instagram-square', label: '1080×1080', width: 1080, height: 1080, aspect: '1:1', description: 'Instagram Square' },
  { id: 'instagram-story', label: '1080×1920', width: 1080, height: 1920, aspect: '9:16', description: 'Instagram Story' },
  { id: 'twitter-post', label: '1200×675', width: 1200, height: 675, aspect: '16:9', description: 'Twitter Post' },
];

// Group sizes by aspect ratio
export const ASPECT_GROUPS = [
  { aspect: '1:1', label: 'Square', sizes: IMAGE_SIZES.filter(s => s.aspect === '1:1') },
  { aspect: '16:9', label: 'Landscape 16:9', sizes: IMAGE_SIZES.filter(s => s.aspect === '16:9') },
  { aspect: '9:16', label: 'Portrait 9:16', sizes: IMAGE_SIZES.filter(s => s.aspect === '9:16') },
  { aspect: '4:3', label: 'Standard 4:3', sizes: IMAGE_SIZES.filter(s => s.aspect === '4:3') },
  { aspect: '3:4', label: 'Portrait 3:4', sizes: IMAGE_SIZES.filter(s => s.aspect === '3:4') },
  { aspect: '3:2', label: 'Photo 3:2', sizes: IMAGE_SIZES.filter(s => s.aspect === '3:2') },
  { aspect: '2:3', label: 'Photo Portrait', sizes: IMAGE_SIZES.filter(s => s.aspect === '2:3') },
  { aspect: '21:9', label: 'Ultrawide', sizes: IMAGE_SIZES.filter(s => s.aspect === '21:9') },
];

interface ImageSizeSelectorProps {
  selectedSize: ImageSize;
  onSizeChange: (size: ImageSize) => void;
  compact?: boolean;
  className?: string;
}

export default function ImageSizeSelector({
  selectedSize,
  onSizeChange,
  compact = false,
  className = '',
}: ImageSizeSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className={`relative ${className}`}>
      {/* Trigger button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-600 
          bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors
          ${compact ? 'text-xs' : 'text-sm'}`}
      >
        <span className="font-medium">{selectedSize.label}</span>
        <span className="text-gray-500 dark:text-gray-400">({selectedSize.aspect})</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      
      {/* Dropdown */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setIsOpen(false)} 
          />
          
          {/* Menu */}
          <div className="absolute left-0 top-full mt-1 z-50 w-64 max-h-80 overflow-y-auto
            bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 
            rounded-lg shadow-lg">
            {ASPECT_GROUPS.map((group) => (
              <div key={group.aspect}>
                <div className="px-3 py-1.5 text-xs font-semibold text-gray-500 dark:text-gray-400 
                  bg-gray-50 dark:bg-gray-900 sticky top-0">
                  {group.label}
                </div>
                {group.sizes.map((size) => (
                  <button
                    key={size.id}
                    onClick={() => {
                      onSizeChange(size);
                      setIsOpen(false);
                    }}
                    className={`w-full flex items-center justify-between px-3 py-2 text-sm
                      hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors
                      ${selectedSize.id === size.id ? 'bg-blue-50 dark:bg-blue-900/30' : ''}`}
                  >
                    <div className="flex flex-col items-start">
                      <span className="font-medium">{size.label}</span>
                      {size.description && (
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {size.description}
                        </span>
                      )}
                    </div>
                    {selectedSize.id === size.id && (
                      <Check className="w-4 h-4 text-blue-500" />
                    )}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// Quick aspect ratio buttons for inline use
interface AspectRatioButtonsProps {
  selectedAspect: string;
  onAspectChange: (aspect: string) => void;
  className?: string;
}

export function AspectRatioButtons({
  selectedAspect,
  onAspectChange,
  className = '',
}: AspectRatioButtonsProps) {
  const aspects = ['1:1', '16:9', '9:16', '4:3', '3:2'];
  
  return (
    <div className={`flex gap-1 ${className}`}>
      {aspects.map((aspect) => (
        <button
          key={aspect}
          onClick={() => onAspectChange(aspect)}
          className={`px-2 py-1 text-xs rounded transition-colors
            ${selectedAspect === aspect
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
        >
          {aspect}
        </button>
      ))}
    </div>
  );
}

// Helper to get default size for an aspect ratio
export function getDefaultSizeForAspect(aspect: string): ImageSize {
  const group = ASPECT_GROUPS.find(g => g.aspect === aspect);
  if (group && group.sizes.length > 0) {
    return group.sizes[0];
  }
  return IMAGE_SIZES[0]; // Default to 1024x1024 square
}

// Helper to find size by dimensions
export function findSizeByDimensions(width: number, height: number): ImageSize | undefined {
  return IMAGE_SIZES.find(s => s.width === width && s.height === height);
}
