/**
 * Frontend Validation Utilities
 * 
 * Provides comprehensive input validation for:
 * - User inputs (forms, chat messages)
 * - File uploads
 * - URL/link validation
 * - Text sanitization
 */

// ============ Constants ============

export const LIMITS = {
  MESSAGE_MAX_LENGTH: 32000,
  CHAT_TITLE_MAX_LENGTH: 255,
  USERNAME_MIN_LENGTH: 3,
  USERNAME_MAX_LENGTH: 50,
  PASSWORD_MIN_LENGTH: 8,
  PASSWORD_MAX_LENGTH: 128,
  EMAIL_MAX_LENGTH: 255,
  FILE_MAX_SIZE_MB: 50,
  ZIP_MAX_SIZE_MB: 100,
  IMAGE_MAX_SIZE_MB: 10,
  MAX_FILES_PER_UPLOAD: 20,
  ALLOWED_IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
  ALLOWED_DOC_TYPES: [
    'application/pdf',
    'text/plain',
    'text/markdown',
    'text/csv',
    'application/json',
    'application/xml',
  ],
  ALLOWED_CODE_EXTENSIONS: [
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala', '.r',
    '.sql', '.sh', '.bash', '.zsh', '.ps1', '.bat', '.yml', '.yaml', '.toml',
    '.ini', '.cfg', '.conf', '.md', '.rst', '.txt', '.json', '.xml', '.html',
    '.css', '.scss', '.less', '.vue', '.svelte',
  ],
} as const;

// ============ Types ============

export interface ValidationResult {
  valid: boolean;
  error?: string;
  warnings?: string[];
}

export interface FileValidationResult extends ValidationResult {
  sanitizedName?: string;
}

// ============ Text Validation ============

/**
 * Validate chat message content
 */
export function validateMessage(content: string): ValidationResult {
  if (!content || content.trim().length === 0) {
    return { valid: false, error: 'Message cannot be empty' };
  }
  
  if (content.length > LIMITS.MESSAGE_MAX_LENGTH) {
    return { 
      valid: false, 
      error: `Message exceeds maximum length of ${LIMITS.MESSAGE_MAX_LENGTH.toLocaleString()} characters` 
    };
  }
  
  return { valid: true };
}

/**
 * Validate chat title
 */
export function validateChatTitle(title: string): ValidationResult {
  if (!title || title.trim().length === 0) {
    return { valid: false, error: 'Title cannot be empty' };
  }
  
  if (title.length > LIMITS.CHAT_TITLE_MAX_LENGTH) {
    return { 
      valid: false, 
      error: `Title exceeds maximum length of ${LIMITS.CHAT_TITLE_MAX_LENGTH} characters` 
    };
  }
  
  return { valid: true };
}

/**
 * Validate username
 */
export function validateUsername(username: string): ValidationResult {
  if (!username || username.trim().length === 0) {
    return { valid: false, error: 'Username is required' };
  }
  
  if (username.length < LIMITS.USERNAME_MIN_LENGTH) {
    return { 
      valid: false, 
      error: `Username must be at least ${LIMITS.USERNAME_MIN_LENGTH} characters` 
    };
  }
  
  if (username.length > LIMITS.USERNAME_MAX_LENGTH) {
    return { 
      valid: false, 
      error: `Username cannot exceed ${LIMITS.USERNAME_MAX_LENGTH} characters` 
    };
  }
  
  // Only alphanumeric, underscores, and hyphens
  if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
    return { 
      valid: false, 
      error: 'Username can only contain letters, numbers, underscores, and hyphens' 
    };
  }
  
  return { valid: true };
}

/**
 * Validate email address
 */
export function validateEmail(email: string): ValidationResult {
  if (!email || email.trim().length === 0) {
    return { valid: false, error: 'Email is required' };
  }
  
  if (email.length > LIMITS.EMAIL_MAX_LENGTH) {
    return { 
      valid: false, 
      error: `Email cannot exceed ${LIMITS.EMAIL_MAX_LENGTH} characters` 
    };
  }
  
  // Basic email regex - not exhaustive but covers most cases
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return { valid: false, error: 'Invalid email address format' };
  }
  
  return { valid: true };
}

/**
 * Validate password strength
 */
export function validatePassword(password: string): ValidationResult {
  if (!password) {
    return { valid: false, error: 'Password is required' };
  }
  
  if (password.length < LIMITS.PASSWORD_MIN_LENGTH) {
    return { 
      valid: false, 
      error: `Password must be at least ${LIMITS.PASSWORD_MIN_LENGTH} characters` 
    };
  }
  
  if (password.length > LIMITS.PASSWORD_MAX_LENGTH) {
    return { 
      valid: false, 
      error: `Password cannot exceed ${LIMITS.PASSWORD_MAX_LENGTH} characters` 
    };
  }
  
  const warnings: string[] = [];
  
  if (!/[A-Z]/.test(password)) {
    warnings.push('Consider adding uppercase letters');
  }
  if (!/[a-z]/.test(password)) {
    warnings.push('Consider adding lowercase letters');
  }
  if (!/[0-9]/.test(password)) {
    warnings.push('Consider adding numbers');
  }
  if (!/[^A-Za-z0-9]/.test(password)) {
    warnings.push('Consider adding special characters');
  }
  
  return { valid: true, warnings: warnings.length > 0 ? warnings : undefined };
}

// ============ File Validation ============

/**
 * Validate file size
 */
export function validateFileSize(file: File, maxSizeMB: number = LIMITS.FILE_MAX_SIZE_MB): ValidationResult {
  const maxBytes = maxSizeMB * 1024 * 1024;
  
  if (file.size > maxBytes) {
    return { 
      valid: false, 
      error: `File size (${formatFileSize(file.size)}) exceeds maximum of ${maxSizeMB}MB` 
    };
  }
  
  return { valid: true };
}

/**
 * Validate file type
 */
export function validateFileType(
  file: File, 
  allowedTypes: readonly string[]
): ValidationResult {
  if (!allowedTypes.includes(file.type)) {
    return { 
      valid: false, 
      error: `File type "${file.type || 'unknown'}" is not allowed` 
    };
  }
  
  return { valid: true };
}

/**
 * Validate file extension
 */
export function validateFileExtension(
  filename: string, 
  allowedExtensions: readonly string[]
): ValidationResult {
  const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
  
  if (!allowedExtensions.includes(ext)) {
    return { 
      valid: false, 
      error: `File extension "${ext}" is not allowed` 
    };
  }
  
  return { valid: true };
}

/**
 * Sanitize filename for safe storage
 */
export function sanitizeFilename(filename: string): string {
  // Remove path separators and null bytes
  let sanitized = filename
    .replace(/[/\\]/g, '_')
    .replace(/\0/g, '')
    .replace(/\.\./g, '_');
  
  // Remove leading/trailing dots and spaces
  sanitized = sanitized.replace(/^[\s.]+|[\s.]+$/g, '');
  
  // Limit length
  if (sanitized.length > 255) {
    const ext = sanitized.substring(sanitized.lastIndexOf('.'));
    sanitized = sanitized.substring(0, 255 - ext.length) + ext;
  }
  
  // Ensure not empty
  if (!sanitized) {
    sanitized = 'unnamed_file';
  }
  
  return sanitized;
}

/**
 * Comprehensive file validation
 */
export function validateFile(
  file: File,
  options: {
    maxSizeMB?: number;
    allowedTypes?: readonly string[];
    allowedExtensions?: readonly string[];
  } = {}
): FileValidationResult {
  const {
    maxSizeMB = LIMITS.FILE_MAX_SIZE_MB,
    allowedTypes,
    allowedExtensions,
  } = options;
  
  // Check size
  const sizeResult = validateFileSize(file, maxSizeMB);
  if (!sizeResult.valid) {
    return sizeResult;
  }
  
  // Check type if specified
  if (allowedTypes) {
    const typeResult = validateFileType(file, allowedTypes);
    if (!typeResult.valid) {
      return typeResult;
    }
  }
  
  // Check extension if specified
  if (allowedExtensions) {
    const extResult = validateFileExtension(file.name, allowedExtensions);
    if (!extResult.valid) {
      return extResult;
    }
  }
  
  return {
    valid: true,
    sanitizedName: sanitizeFilename(file.name),
  };
}

/**
 * Validate image file
 */
export function validateImageFile(file: File): FileValidationResult {
  return validateFile(file, {
    maxSizeMB: LIMITS.IMAGE_MAX_SIZE_MB,
    allowedTypes: LIMITS.ALLOWED_IMAGE_TYPES,
  });
}

/**
 * Validate document file
 */
export function validateDocumentFile(file: File): FileValidationResult {
  return validateFile(file, {
    maxSizeMB: LIMITS.FILE_MAX_SIZE_MB,
    allowedTypes: LIMITS.ALLOWED_DOC_TYPES,
  });
}

/**
 * Validate zip file
 */
export function validateZipFile(file: File): FileValidationResult {
  if (!file.name.toLowerCase().endsWith('.zip')) {
    return { valid: false, error: 'File must be a .zip archive' };
  }
  
  return validateFile(file, {
    maxSizeMB: LIMITS.ZIP_MAX_SIZE_MB,
    allowedTypes: ['application/zip', 'application/x-zip-compressed'],
  });
}

// ============ URL Validation ============

/**
 * Validate URL format
 */
export function validateUrl(url: string): ValidationResult {
  if (!url || url.trim().length === 0) {
    return { valid: false, error: 'URL is required' };
  }
  
  try {
    const parsed = new URL(url);
    
    // Only allow http and https
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      return { valid: false, error: 'URL must use HTTP or HTTPS protocol' };
    }
    
    return { valid: true };
  } catch {
    return { valid: false, error: 'Invalid URL format' };
  }
}

/**
 * Check if URL is from a trusted domain
 */
export function isUrlFromDomain(url: string, domains: string[]): boolean {
  try {
    const parsed = new URL(url);
    return domains.some(domain => 
      parsed.hostname === domain || 
      parsed.hostname.endsWith(`.${domain}`)
    );
  } catch {
    return false;
  }
}

// ============ Sanitization ============

/**
 * Sanitize text input (remove control characters, normalize whitespace)
 */
export function sanitizeText(text: string): string {
  return text
    // Remove control characters except newlines and tabs
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
    // Normalize line endings
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    // Trim excessive whitespace
    .replace(/[ \t]+/g, ' ')
    // Limit consecutive newlines
    .replace(/\n{4,}/g, '\n\n\n');
}

/**
 * Escape HTML special characters
 */
export function escapeHtml(text: string): string {
  const htmlEscapes: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  };
  
  return text.replace(/[&<>"']/g, char => htmlEscapes[char]);
}

/**
 * Strip HTML tags from text
 */
export function stripHtml(html: string): string {
  return html.replace(/<[^>]*>/g, '');
}

// ============ Helpers ============

/**
 * Format file size for display
 */
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Check if a value is empty (null, undefined, empty string, empty array)
 */
export function isEmpty(value: unknown): boolean {
  if (value == null) return true;
  if (typeof value === 'string') return value.trim().length === 0;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;
  return false;
}

/**
 * Debounce validation for real-time input
 */
export function createDebouncedValidator<T>(
  validator: (value: T) => ValidationResult,
  delayMs: number = 300
): (value: T, callback: (result: ValidationResult) => void) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  
  return (value: T, callback: (result: ValidationResult) => void) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    
    timeoutId = setTimeout(() => {
      callback(validator(value));
    }, delayMs);
  };
}
