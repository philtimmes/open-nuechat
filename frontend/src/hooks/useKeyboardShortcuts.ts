import { useEffect, useCallback, useRef } from 'react';

/**
 * Keyboard shortcut configuration
 */
export interface KeyboardShortcut {
  /** Key to press (e.g., 'n', 'Enter', 'Escape') */
  key: string;
  /** Require Ctrl/Cmd key */
  ctrl?: boolean;
  /** Require Shift key */
  shift?: boolean;
  /** Require Alt/Option key */
  alt?: boolean;
  /** Callback when shortcut is triggered */
  handler: (event: KeyboardEvent) => void;
  /** Description for help display */
  description?: string;
  /** Prevent default browser behavior */
  preventDefault?: boolean;
  /** Only trigger when not in an input/textarea */
  ignoreInInputs?: boolean;
}

/**
 * Check if the current platform is Mac
 */
export function isMac(): boolean {
  if (typeof navigator === 'undefined') return false;
  return navigator.platform.toLowerCase().includes('mac');
}

/**
 * Get the modifier key name for display
 */
export function getModifierKey(): string {
  return isMac() ? '⌘' : 'Ctrl';
}

/**
 * Format a shortcut for display
 */
export function formatShortcut(shortcut: KeyboardShortcut): string {
  const parts: string[] = [];
  
  if (shortcut.ctrl) {
    parts.push(isMac() ? '⌘' : 'Ctrl');
  }
  if (shortcut.alt) {
    parts.push(isMac() ? '⌥' : 'Alt');
  }
  if (shortcut.shift) {
    parts.push('Shift');
  }
  
  // Format the key nicely
  let keyDisplay = shortcut.key;
  if (keyDisplay === ' ') keyDisplay = 'Space';
  if (keyDisplay.length === 1) keyDisplay = keyDisplay.toUpperCase();
  parts.push(keyDisplay);
  
  return parts.join(isMac() ? '' : '+');
}

/**
 * Check if an element is an input that should capture keyboard events
 */
function isInputElement(element: Element | null): boolean {
  if (!element) return false;
  const tagName = element.tagName.toLowerCase();
  return (
    tagName === 'input' ||
    tagName === 'textarea' ||
    tagName === 'select' ||
    (element as HTMLElement).isContentEditable
  );
}

/**
 * Hook for registering keyboard shortcuts
 * 
 * @param shortcuts Array of shortcut configurations
 * @param deps Dependency array for the shortcuts (like useEffect deps)
 * 
 * @example
 * useKeyboardShortcuts([
 *   {
 *     key: 'n',
 *     ctrl: true,
 *     handler: () => createNewChat(),
 *     description: 'New chat',
 *   },
 *   {
 *     key: 'Escape',
 *     handler: () => closePanel(),
 *     description: 'Close panel',
 *   },
 * ]);
 */
export function useKeyboardShortcuts(
  shortcuts: KeyboardShortcut[],
  deps: React.DependencyList = []
): void {
  const shortcutsRef = useRef(shortcuts);
  shortcutsRef.current = shortcuts;

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    for (const shortcut of shortcutsRef.current) {
      // Check modifier keys
      const ctrlPressed = isMac() ? event.metaKey : event.ctrlKey;
      const ctrlMatch = shortcut.ctrl ? ctrlPressed : !ctrlPressed;
      const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
      const altMatch = shortcut.alt ? event.altKey : !event.altKey;
      
      // Check the key
      const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
      
      if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
        // Check if we should ignore inputs
        if (shortcut.ignoreInInputs !== false && isInputElement(document.activeElement)) {
          // Allow Escape in inputs by default
          if (shortcut.key !== 'Escape') {
            continue;
          }
        }
        
        if (shortcut.preventDefault !== false) {
          event.preventDefault();
        }
        
        shortcut.handler(event);
        return;
      }
    }
  }, []);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown, ...deps]);
}

/**
 * Pre-configured chat shortcuts
 * 
 * @param handlers Object with handler functions for each shortcut
 */
export function useChatShortcuts(handlers: {
  onNewChat?: () => void;
  onFocusInput?: () => void;
  onToggleSidebar?: () => void;
  onDeleteChat?: () => void;
  onToggleArtifacts?: () => void;
  onClosePanel?: () => void;
  onSearch?: () => void;
  onSettings?: () => void;
}): void {
  const shortcuts: KeyboardShortcut[] = [];

  if (handlers.onNewChat) {
    shortcuts.push({
      key: 'n',
      ctrl: true,
      handler: handlers.onNewChat,
      description: 'New chat',
    });
  }

  if (handlers.onFocusInput) {
    shortcuts.push({
      key: '/',
      ctrl: true,
      handler: handlers.onFocusInput,
      description: 'Focus input',
    });
  }

  if (handlers.onToggleSidebar) {
    shortcuts.push({
      key: 'b',
      ctrl: true,
      handler: handlers.onToggleSidebar,
      description: 'Toggle sidebar',
    });
  }

  if (handlers.onDeleteChat) {
    shortcuts.push({
      key: 'Backspace',
      ctrl: true,
      shift: true,
      handler: handlers.onDeleteChat,
      description: 'Delete chat',
    });
  }

  if (handlers.onToggleArtifacts) {
    shortcuts.push({
      key: 'a',
      ctrl: true,
      shift: true,
      handler: handlers.onToggleArtifacts,
      description: 'Toggle artifacts panel',
    });
  }

  if (handlers.onClosePanel) {
    shortcuts.push({
      key: 'Escape',
      handler: handlers.onClosePanel,
      description: 'Close panel',
      ignoreInInputs: false, // Allow Escape in inputs
    });
  }

  if (handlers.onSearch) {
    shortcuts.push({
      key: 'k',
      ctrl: true,
      handler: handlers.onSearch,
      description: 'Search',
    });
  }

  if (handlers.onSettings) {
    shortcuts.push({
      key: ',',
      ctrl: true,
      handler: handlers.onSettings,
      description: 'Settings',
    });
  }

  useKeyboardShortcuts(shortcuts, [
    handlers.onNewChat,
    handlers.onFocusInput,
    handlers.onToggleSidebar,
    handlers.onDeleteChat,
    handlers.onToggleArtifacts,
    handlers.onClosePanel,
    handlers.onSearch,
    handlers.onSettings,
  ]);
}

/**
 * Get all available shortcuts for help display
 */
export function getChatShortcuts(): Array<{ shortcut: string; description: string }> {
  const mod = getModifierKey();
  return [
    { shortcut: `${mod}+N`, description: 'New chat' },
    { shortcut: `${mod}+/`, description: 'Focus input' },
    { shortcut: `${mod}+B`, description: 'Toggle sidebar' },
    { shortcut: `${mod}+Shift+Backspace`, description: 'Delete chat' },
    { shortcut: `${mod}+Shift+A`, description: 'Toggle artifacts' },
    { shortcut: `${mod}+K`, description: 'Search' },
    { shortcut: `${mod}+,`, description: 'Settings' },
    { shortcut: 'Escape', description: 'Close panel' },
  ];
}
