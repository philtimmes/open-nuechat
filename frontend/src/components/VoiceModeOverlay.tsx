import React, { useState, useEffect, useCallback, useRef } from 'react';

interface VoiceModeOverlayProps {
  isActive: boolean;
  isListening: boolean;
  isReading: boolean;
  isProcessing: boolean;
  currentText?: string;
  onInterrupt: () => void;  // Single tap - stop reading but stay in voice mode
  onExit: () => void;       // Double tap or X button - exit voice mode
  onRetry?: () => void;     // Swipe left - retry/regenerate last response
  onBranchLeft?: () => void;  // Swipe right - navigate to previous branch
  onBranchRight?: () => void; // Reserved for future - navigate to next branch
  hasBranches?: boolean;    // Whether there are branches to navigate
  debugInfo?: string;       // Debug info to display (tap status text to toggle)
}

// Sentiment analysis for emoji selection
function analyzeTextForEmoji(text: string): string {
  if (!text) return '🎙️'; // Default mic emoji when no text
  
  const lowerText = text.toLowerCase();
  
  // Animal mentions (check first for specificity)
  if (/\b(cat|cats|kitten|kittens|kitty|feline)\b/.test(lowerText)) return '🐱';
  if (/\b(dog|dogs|puppy|puppies|doggy|canine|pup)\b/.test(lowerText)) return '🐕';
  if (/\b(bird|birds|parrot|eagle|owl)\b/.test(lowerText)) return '🐦';
  if (/\b(fish|fishes|shark|whale|dolphin)\b/.test(lowerText)) return '🐟';
  if (/\b(bear|bears)\b/.test(lowerText)) return '🐻';
  if (/\b(rabbit|bunny|bunnies)\b/.test(lowerText)) return '🐰';
  if (/\b(horse|horses|pony)\b/.test(lowerText)) return '🐴';
  if (/\b(monkey|ape|gorilla|chimp)\b/.test(lowerText)) return '🐵';
  if (/\b(snake|snakes|serpent)\b/.test(lowerText)) return '🐍';
  if (/\b(bug|insect|spider|ant|bee)\b/.test(lowerText)) return '🐛';
  
  // Topic-based emojis
  if (/\b(code|coding|program|software|developer|javascript|python|react)\b/.test(lowerText)) return '💻';
  if (/\b(music|song|singing|melody|concert)\b/.test(lowerText)) return '🎵';
  if (/\b(food|eat|eating|delicious|tasty|cook|recipe)\b/.test(lowerText)) return '🍽️';
  if (/\b(love|heart|romantic|romance)\b/.test(lowerText)) return '❤️';
  if (/\b(money|dollar|cash|rich|wealth|expensive|price)\b/.test(lowerText)) return '💰';
  if (/\b(time|clock|hour|minute|schedule)\b/.test(lowerText)) return '⏰';
  if (/\b(weather|rain|sun|sunny|cloud|storm)\b/.test(lowerText)) return '🌤️';
  if (/\b(book|books|reading|read|story|novel)\b/.test(lowerText)) return '📚';
  if (/\b(game|games|gaming|play|player)\b/.test(lowerText)) return '🎮';
  if (/\b(phone|call|mobile|smartphone)\b/.test(lowerText)) return '📱';
  if (/\b(car|cars|driving|vehicle|road)\b/.test(lowerText)) return '🚗';
  if (/\b(plane|airplane|flight|flying|airport)\b/.test(lowerText)) return '✈️';
  if (/\b(home|house|room|apartment)\b/.test(lowerText)) return '🏠';
  if (/\b(work|job|office|career|business)\b/.test(lowerText)) return '💼';
  if (/\b(school|learn|study|education|university|college)\b/.test(lowerText)) return '🎓';
  if (/\b(health|doctor|medical|hospital|sick|medicine)\b/.test(lowerText)) return '🏥';
  if (/\b(sport|exercise|gym|workout|fitness)\b/.test(lowerText)) return '🏃';
  if (/\b(art|artist|paint|painting|draw|creative)\b/.test(lowerText)) return '🎨';
  if (/\b(camera|photo|picture|photograph)\b/.test(lowerText)) return '📷';
  if (/\b(star|stars|space|galaxy|universe|planet)\b/.test(lowerText)) return '⭐';
  if (/\b(fire|hot|burn|flame)\b/.test(lowerText)) return '🔥';
  if (/\b(water|ocean|sea|swim|swimming|pool)\b/.test(lowerText)) return '🌊';
  if (/\b(tree|trees|forest|plant|nature|garden)\b/.test(lowerText)) return '🌳';
  if (/\b(birthday|party|celebrate|celebration)\b/.test(lowerText)) return '🎉';
  if (/\b(sleep|sleeping|tired|bed|dream)\b/.test(lowerText)) return '😴';
  if (/\b(ai|artificial intelligence|machine learning|neural|model)\b/.test(lowerText)) return '🤖';
  
  // Punctuation-based sentiment (check the actual text, not lowered)
  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;
  const commaCount = (text.match(/,/g) || []).length;
  const colonCount = (text.match(/:/g) || []).length;
  
  // Multiple exclamations = very excited
  if (exclamationCount >= 3) return '🤩';
  if (exclamationCount >= 2) return '😄';
  if (exclamationCount === 1) return '😊';
  
  // Multiple questions = confused or curious
  if (questionCount >= 2) return '🤔';
  if (questionCount === 1) return '❓';
  
  // Lists/outlines (multiple commas, colons, or bullet-like patterns)
  if (commaCount >= 5 || colonCount >= 2 || /\d+\.\s/.test(text) || /^[-•*]\s/m.test(text)) return '🤔';
  
  // Emotional words
  if (/\b(happy|happiness|joy|joyful|great|wonderful|fantastic|amazing|awesome)\b/.test(lowerText)) return '😊';
  if (/\b(sad|sadness|unhappy|sorry|unfortunately|regret)\b/.test(lowerText)) return '😢';
  if (/\b(angry|anger|mad|furious|annoyed)\b/.test(lowerText)) return '😠';
  if (/\b(scared|fear|afraid|terrified|scary)\b/.test(lowerText)) return '😨';
  if (/\b(surprised|surprise|wow|unexpected|shocking)\b/.test(lowerText)) return '😮';
  if (/\b(confused|confusing|unclear|don't understand)\b/.test(lowerText)) return '😕';
  if (/\b(laugh|funny|hilarious|lol|haha|joke)\b/.test(lowerText)) return '😂';
  if (/\b(thank|thanks|grateful|appreciate)\b/.test(lowerText)) return '🙏';
  if (/\b(yes|correct|right|exactly|absolutely)\b/.test(lowerText)) return '✅';
  if (/\b(no|wrong|incorrect|mistake|error)\b/.test(lowerText)) return '❌';
  if (/\b(warning|caution|careful|danger|risk)\b/.test(lowerText)) return '⚠️';
  if (/\b(idea|ideas|suggest|suggestion|perhaps|maybe|could)\b/.test(lowerText)) return '💡';
  if (/\b(important|note|remember|key|crucial)\b/.test(lowerText)) return '📌';
  if (/\b(hello|hi|hey|greetings|welcome)\b/.test(lowerText)) return '👋';
  if (/\b(goodbye|bye|farewell|see you)\b/.test(lowerText)) return '👋';
  
  // Default based on text length
  if (text.length > 200) return '📝'; // Long text = document
  
  return '💬'; // Default speech bubble
}

export const VoiceModeOverlay: React.FC<VoiceModeOverlayProps> = ({
  isActive,
  isListening,
  isReading,
  isProcessing,
  currentText,
  onInterrupt,
  onExit,
  onRetry,
  onBranchLeft,
  hasBranches,
  debugInfo,
}) => {
  const [emoji, setEmoji] = useState('🎙️');
  const [statusText, setStatusText] = useState('Listening...');
  const [pulseClass, setPulseClass] = useState('');
  const [swipeIndicator, setSwipeIndicator] = useState<'left' | 'right' | null>(null);
  const lastTextRef = useRef('');
  const lastTapRef = useRef<number>(0);
  const tapTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Swipe detection refs
  const touchStartRef = useRef<{ x: number; y: number; time: number } | null>(null);
  const SWIPE_THRESHOLD = 100; // Minimum distance for swipe
  const SWIPE_TIMEOUT = 500; // Max time for swipe gesture
  
  // Handle touch start
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0];
    touchStartRef.current = {
      x: touch.clientX,
      y: touch.clientY,
      time: Date.now()
    };
  }, []);
  
  // Handle touch end (detect swipe)
  const handleTouchEnd = useCallback((e: React.TouchEvent) => {
    if (!touchStartRef.current) return;
    
    const touch = e.changedTouches[0];
    const deltaX = touch.clientX - touchStartRef.current.x;
    const deltaY = touch.clientY - touchStartRef.current.y;
    const elapsed = Date.now() - touchStartRef.current.time;
    
    // Check if it's a valid swipe (horizontal, fast enough, long enough)
    if (elapsed < SWIPE_TIMEOUT && Math.abs(deltaX) > SWIPE_THRESHOLD && Math.abs(deltaX) > Math.abs(deltaY) * 2) {
      if (deltaX < 0) {
        // Swipe LEFT - Retry
        console.log('Swipe left detected - retry');
        setSwipeIndicator('left');
        setTimeout(() => setSwipeIndicator(null), 300);
        if (onRetry) onRetry();
      } else {
        // Swipe RIGHT - Branch navigation
        console.log('Swipe right detected - branch navigation');
        setSwipeIndicator('right');
        setTimeout(() => setSwipeIndicator(null), 300);
        if (onBranchLeft && hasBranches) onBranchLeft();
      }
      touchStartRef.current = null;
      return;
    }
    
    // Not a swipe - treat as tap
    touchStartRef.current = null;
  }, [onRetry, onBranchLeft, hasBranches]);
  
  // Update emoji based on current text being read
  useEffect(() => {
    if (currentText && currentText !== lastTextRef.current) {
      const newEmoji = analyzeTextForEmoji(currentText);
      setEmoji(newEmoji);
      lastTextRef.current = currentText;
    }
  }, [currentText]);
  
  // Update status and animation based on state
  useEffect(() => {
    if (isProcessing) {
      setStatusText('Processing...');
      setPulseClass('animate-pulse-slow');
      setEmoji('⏳');
    } else if (isReading) {
      setStatusText('Speaking...');
      setPulseClass('animate-bounce-slow');
      // Emoji is set by currentText
    } else if (isListening) {
      setStatusText('Listening...');
      setPulseClass('animate-pulse');
      if (!currentText) setEmoji('🎙️');
    } else {
      setStatusText('Ready');
      setPulseClass('');
      if (!currentText) setEmoji('🎙️');
    }
  }, [isListening, isReading, isProcessing, currentText]);
  
  // Handle tap/double-tap (only for click events, not touch swipes)
  const handleTap = useCallback(() => {
    const now = Date.now();
    const timeSinceLastTap = now - lastTapRef.current;
    
    // Clear any pending single-tap action
    if (tapTimeoutRef.current) {
      clearTimeout(tapTimeoutRef.current);
      tapTimeoutRef.current = null;
    }
    
    if (timeSinceLastTap < 300) {
      // Double tap - exit voice mode
      console.log('Double tap detected - exiting voice mode');
      onExit();
      lastTapRef.current = 0;
    } else {
      // Potential single tap - wait to see if it's a double tap
      lastTapRef.current = now;
      tapTimeoutRef.current = setTimeout(() => {
        // Single tap confirmed - interrupt (stop reading)
        console.log('Single tap detected - interrupting');
        onInterrupt();
        tapTimeoutRef.current = null;
      }, 300);
    }
  }, [onInterrupt, onExit]);
  
  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isActive) {
        onExit();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isActive, onExit]);
  
  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (tapTimeoutRef.current) {
        clearTimeout(tapTimeoutRef.current);
      }
    };
  }, []);
  
  if (!isActive) return null;
  
  return (
    <div 
      className="fixed inset-0 z-[100] bg-[var(--color-background)] flex flex-col items-center justify-center select-none"
      style={{ touchAction: 'none' }} // Disable browser touch handling for our custom gestures
      onClick={handleTap}
      onTouchStart={handleTouchStart}
      onTouchEnd={(e) => {
        // Check if it was a swipe first
        if (touchStartRef.current) {
          const touch = e.changedTouches[0];
          const deltaX = touch.clientX - touchStartRef.current.x;
          const deltaY = touch.clientY - touchStartRef.current.y;
          const elapsed = Date.now() - touchStartRef.current.time;
          
          // If it's a valid horizontal swipe, let handleTouchEnd deal with it
          if (elapsed < SWIPE_TIMEOUT && Math.abs(deltaX) > SWIPE_THRESHOLD && Math.abs(deltaX) > Math.abs(deltaY) * 2) {
            handleTouchEnd(e);
            e.preventDefault();
            return;
          }
        }
        
        // Not a swipe - treat as tap
        e.preventDefault();
        handleTap();
        touchStartRef.current = null;
      }}
    >
      {/* Swipe indicators */}
      {swipeIndicator === 'left' && (
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-4xl animate-bounce-left">
          ↩️
        </div>
      )}
      {swipeIndicator === 'right' && (
        <div className="absolute right-4 top-1/2 -translate-y-1/2 text-4xl animate-bounce-right">
          ↪️
        </div>
      )}
      
      {/* Close (X) button in top right */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onExit();
        }}
        className="absolute top-4 right-4 p-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] rounded-full transition-colors z-10"
        title="Exit Voice Mode"
      >
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
      
      {/* Debug info display in top left (when enabled in Admin > Site Dev) */}
      {debugInfo && (
        <div 
          className="absolute top-4 left-4 p-3 bg-black/70 rounded-lg max-w-[60%] overflow-auto pointer-events-auto z-10"
          onClick={(e) => e.stopPropagation()}
        >
          <pre className="text-xs text-green-400 whitespace-pre-wrap font-mono">
            {debugInfo}
          </pre>
        </div>
      )}
      
      {/* Animated background gradient */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div 
          className={`absolute inset-0 opacity-20`}
          style={{
            background: isListening 
              ? 'radial-gradient(circle at center, var(--color-primary) 0%, transparent 70%)'
              : isReading
              ? 'radial-gradient(circle at center, var(--color-success, #22c55e) 0%, transparent 70%)'
              : 'none',
            animation: isListening || isReading ? 'pulse-bg 2s ease-in-out infinite' : 'none',
          }}
        />
      </div>
      
      {/* Main emoji display */}
      <div className="relative z-10 flex flex-col items-center pointer-events-none">
        <div 
          className={`text-[120px] sm:text-[180px] md:text-[220px] transition-all duration-300 ${pulseClass}`}
          style={{
            filter: isReading ? 'drop-shadow(0 0 30px rgba(34, 197, 94, 0.5))' : 
                   isListening ? 'drop-shadow(0 0 30px rgba(var(--color-primary-rgb, 59, 130, 246), 0.5))' : 'none',
          }}
        >
          {emoji}
        </div>
        
        {/* Status indicator */}
        <div className="mt-8 flex items-center gap-3">
          <div 
            className={`w-3 h-3 rounded-full ${
              isListening ? 'bg-[var(--color-primary)] animate-pulse' :
              isReading ? 'bg-green-500 animate-pulse' :
              isProcessing ? 'bg-yellow-500 animate-pulse' :
              'bg-gray-400'
            }`}
          />
          <span className="text-xl text-[var(--color-text-secondary)]">
            {statusText}
          </span>
        </div>
        
        {/* Visual audio indicator when listening */}
        {isListening && (
          <div className="mt-6 flex items-center gap-1">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="w-1 bg-[var(--color-primary)] rounded-full animate-sound-wave"
                style={{
                  height: '20px',
                  animationDelay: `${i * 0.1}s`,
                }}
              />
            ))}
          </div>
        )}
      </div>
      
      {/* Instructions at bottom */}
      <div className="absolute bottom-8 left-0 right-0 flex flex-col items-center gap-2 pointer-events-none">
        <div className="text-sm text-[var(--color-text-secondary)] opacity-70 text-center px-4">
          <div className="mb-1">
            <span className="font-medium">Tap</span> to interrupt • <span className="font-medium">Double-tap</span> to exit
          </div>
          <div className="mb-1">
            <span className="font-medium">Swipe left</span> to retry • <span className="font-medium">Swipe right</span> for branches
          </div>
          <div className="mb-1">
            Say <span className="font-medium">"stop"</span> to exit
          </div>
          <div className="text-xs opacity-80">
            <span className="font-medium">"Change voice, Natural, Adam"</span> or <span className="font-medium">"Change voice, OS, English"</span>
          </div>
        </div>
        <div className="text-xs text-[var(--color-text-secondary)] opacity-50">
          Press ESC to exit
        </div>
      </div>
      
      {/* Styles */}
      <style>{`
        @keyframes pulse-bg {
          0%, 100% { transform: scale(1); opacity: 0.2; }
          50% { transform: scale(1.2); opacity: 0.3; }
        }
        
        @keyframes bounce-left {
          0%, 100% { transform: translateY(-50%) translateX(0); }
          50% { transform: translateY(-50%) translateX(-20px); }
        }
        
        @keyframes bounce-right {
          0%, 100% { transform: translateY(-50%) translateX(0); }
          50% { transform: translateY(-50%) translateX(20px); }
        }
        
        .animate-bounce-left {
          animation: bounce-left 0.3s ease-out;
        }
        
        .animate-bounce-right {
          animation: bounce-right 0.3s ease-out;
        }
        
        @keyframes sound-wave {
          0%, 100% { height: 8px; }
          50% { height: 24px; }
        }
        
        .animate-sound-wave {
          animation: sound-wave 0.5s ease-in-out infinite;
        }
        
        .animate-pulse-slow {
          animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        .animate-bounce-slow {
          animation: bounce 1.5s infinite;
        }
        
        @keyframes bounce {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-10px);
          }
        }
      `}</style>
    </div>
  );
};

export default VoiceModeOverlay;
