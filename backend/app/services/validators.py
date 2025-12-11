"""
Input validation utilities for Open-NueChat

Provides:
- File upload validation
- User input sanitization
- Request parameter validation
- Content type validation
"""
import re
import os
import mimetypes
from typing import Optional, Tuple, List, Set
from pydantic import validator, field_validator
from fastapi import UploadFile, HTTPException

# =============================================================================
# Constants
# =============================================================================

# Maximum file sizes (in bytes)
MAX_AVATAR_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Allowed file extensions by type
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}
ALLOWED_DOCUMENT_EXTENSIONS: Set[str] = {
    '.txt', '.md', '.pdf', '.doc', '.docx', '.rtf',
    '.csv', '.json', '.xml', '.yaml', '.yml',
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
    '.java', '.cpp', '.c', '.h', '.hpp', '.rs', '.go',
    '.rb', '.php', '.swift', '.kt', '.scala',
    '.sql', '.sh', '.bash', '.zsh', '.ps1',
}
ALLOWED_ARCHIVE_EXTENSIONS: Set[str] = {'.zip', '.tar', '.gz', '.tgz', '.tar.gz'}

# Dangerous file patterns
DANGEROUS_PATTERNS = [
    r'\.exe$', r'\.dll$', r'\.bat$', r'\.cmd$', r'\.com$',
    r'\.scr$', r'\.pif$', r'\.msi$', r'\.msp$',
    r'\.vbs$', r'\.vbe$', r'\.js$', r'\.jse$',  # Only dangerous as executables
    r'\.ws[fh]$', r'\.ps1$', r'\.ps1xml$', r'\.ps2$',
    r'\.reg$', r'\.inf$', r'\.lnk$',
    r'\.jar$', r'\.class$',
    r'\.app$', r'\.dmg$',  # macOS
    r'\.deb$', r'\.rpm$',  # Linux packages
]

# Username/slug validation
USERNAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{2,29}$')
SLUG_PATTERN = re.compile(r'^[a-z][a-z0-9-]{2,49}$')
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Content limits
MAX_CHAT_TITLE_LENGTH = 255
MAX_MESSAGE_LENGTH = 100_000
MAX_SYSTEM_PROMPT_LENGTH = 50_000
MAX_ASSISTANT_NAME_LENGTH = 100
MAX_ASSISTANT_DESCRIPTION_LENGTH = 10_000

# =============================================================================
# File Validation
# =============================================================================

class FileValidationError(Exception):
    """Raised when file validation fails"""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


def validate_file_extension(
    filename: str,
    allowed_extensions: Set[str],
    category: str = "file"
) -> Tuple[str, str]:
    """
    Validate file extension against allowed list.
    
    Args:
        filename: Original filename
        allowed_extensions: Set of allowed extensions (with dots)
        category: Category name for error messages
    
    Returns:
        Tuple of (extension, mime_type)
    
    Raises:
        FileValidationError: If extension not allowed
    """
    if not filename:
        raise FileValidationError("Filename is required")
    
    # Get extension (handle compound extensions like .tar.gz)
    lower_filename = filename.lower()
    extension = None
    
    for ext in allowed_extensions:
        if lower_filename.endswith(ext):
            extension = ext
            break
    
    if extension is None:
        _, extension = os.path.splitext(lower_filename)
    
    if extension not in allowed_extensions:
        raise FileValidationError(
            f"File type '{extension}' not allowed for {category}",
            {"allowed": list(allowed_extensions), "received": extension}
        )
    
    mime_type, _ = mimetypes.guess_type(filename)
    return extension, mime_type or 'application/octet-stream'


def validate_file_size(
    file_size: int,
    max_size: int,
    category: str = "file"
) -> None:
    """
    Validate file size against maximum.
    
    Args:
        file_size: Size in bytes
        max_size: Maximum size in bytes
        category: Category name for error messages
    
    Raises:
        FileValidationError: If file too large
    """
    if file_size > max_size:
        raise FileValidationError(
            f"File too large for {category}. Maximum size is {max_size / (1024*1024):.1f} MB",
            {"max_bytes": max_size, "received_bytes": file_size}
        )


def is_dangerous_file(filename: str) -> bool:
    """
    Check if filename matches dangerous patterns.
    
    Args:
        filename: Filename to check
    
    Returns:
        True if file is potentially dangerous
    """
    lower_filename = filename.lower()
    return any(re.search(pattern, lower_filename) for pattern in DANGEROUS_PATTERNS)


async def validate_upload_file(
    file: UploadFile,
    allowed_extensions: Set[str],
    max_size: int,
    category: str = "file",
    check_dangerous: bool = True
) -> Tuple[bytes, str, str]:
    """
    Fully validate an uploaded file.
    
    Args:
        file: FastAPI UploadFile
        allowed_extensions: Allowed extensions set
        max_size: Maximum file size
        category: Category for error messages
        check_dangerous: Whether to check for dangerous patterns
    
    Returns:
        Tuple of (content_bytes, extension, mime_type)
    
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Validate filename
        if not file.filename:
            raise FileValidationError("Filename is required")
        
        # Check for dangerous files
        if check_dangerous and is_dangerous_file(file.filename):
            raise FileValidationError(
                "File type not allowed for security reasons",
                {"filename": file.filename}
            )
        
        # Validate extension
        extension, mime_type = validate_file_extension(
            file.filename, allowed_extensions, category
        )
        
        # Read and validate size
        content = await file.read()
        validate_file_size(len(content), max_size, category)
        
        return content, extension, mime_type
        
    except FileValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)


# =============================================================================
# Text Input Validation
# =============================================================================

def sanitize_string(
    value: str,
    max_length: int,
    strip: bool = True,
    allow_newlines: bool = True,
    field_name: str = "field"
) -> str:
    """
    Sanitize and validate a string input.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        strip: Whether to strip whitespace
        allow_newlines: Whether to allow newlines
        field_name: Field name for error messages
    
    Returns:
        Sanitized string
    
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        raise ValueError(f"{field_name} is required")
    
    if strip:
        value = value.strip()
    
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    
    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length of {max_length}")
    
    if not allow_newlines:
        value = value.replace('\n', ' ').replace('\r', '')
    
    # Remove null bytes
    value = value.replace('\x00', '')
    
    return value


def validate_username(username: str) -> str:
    """
    Validate username format.
    
    Rules:
    - 3-30 characters
    - Starts with letter
    - Contains only letters, numbers, underscores, hyphens
    
    Returns:
        Validated username
    
    Raises:
        ValueError: If validation fails
    """
    username = username.strip()
    
    if not USERNAME_PATTERN.match(username):
        raise ValueError(
            "Username must be 3-30 characters, start with a letter, "
            "and contain only letters, numbers, underscores, and hyphens"
        )
    
    return username


def validate_email(email: str) -> str:
    """
    Validate email format.
    
    Returns:
        Validated email (lowercase)
    
    Raises:
        ValueError: If validation fails
    """
    email = email.strip().lower()
    
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Invalid email format")
    
    if len(email) > 255:
        raise ValueError("Email exceeds maximum length of 255 characters")
    
    return email


def validate_slug(slug: str) -> str:
    """
    Validate URL slug format.
    
    Rules:
    - 3-50 characters
    - Lowercase letters, numbers, hyphens only
    - Starts with letter
    
    Returns:
        Validated slug
    
    Raises:
        ValueError: If validation fails
    """
    slug = slug.strip().lower()
    
    if not SLUG_PATTERN.match(slug):
        raise ValueError(
            "Slug must be 3-50 characters, start with a letter, "
            "and contain only lowercase letters, numbers, and hyphens"
        )
    
    return slug


def validate_password(password: str, min_length: int = 8) -> str:
    """
    Validate password strength.
    
    Rules:
    - Minimum length (default 8)
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    
    Returns:
        Password (unchanged)
    
    Raises:
        ValueError: If validation fails
    """
    if len(password) < min_length:
        raise ValueError(f"Password must be at least {min_length} characters")
    
    if not re.search(r'[A-Z]', password):
        raise ValueError("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        raise ValueError("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        raise ValueError("Password must contain at least one digit")
    
    return password


def validate_hex_color(color: str) -> str:
    """
    Validate hex color format.
    
    Args:
        color: Color string (e.g., "#ff0000" or "ff0000")
    
    Returns:
        Normalized color with # prefix
    
    Raises:
        ValueError: If validation fails
    """
    color = color.strip().lower()
    
    if not color.startswith('#'):
        color = '#' + color
    
    if not re.match(r'^#[0-9a-f]{6}$', color):
        raise ValueError("Invalid hex color format. Use #RRGGBB format")
    
    return color


def validate_url(url: str, allowed_schemes: Set[str] = {'http', 'https'}) -> str:
    """
    Validate URL format.
    
    Args:
        url: URL string
        allowed_schemes: Allowed URL schemes
    
    Returns:
        Validated URL
    
    Raises:
        ValueError: If validation fails
    """
    from urllib.parse import urlparse
    
    url = url.strip()
    
    try:
        parsed = urlparse(url)
        
        if parsed.scheme not in allowed_schemes:
            raise ValueError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
        
        if not parsed.netloc:
            raise ValueError("Invalid URL format")
        
        return url
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")


# =============================================================================
# List/Array Validation
# =============================================================================

def validate_string_list(
    items: List[str],
    max_items: int,
    max_item_length: int,
    field_name: str = "list"
) -> List[str]:
    """
    Validate a list of strings.
    
    Args:
        items: List of strings
        max_items: Maximum number of items
        max_item_length: Maximum length per item
        field_name: Field name for error messages
    
    Returns:
        Validated list
    
    Raises:
        ValueError: If validation fails
    """
    if len(items) > max_items:
        raise ValueError(f"{field_name} exceeds maximum of {max_items} items")
    
    validated = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{i}] must be a string")
        
        item = item.strip()
        if len(item) > max_item_length:
            raise ValueError(
                f"{field_name}[{i}] exceeds maximum length of {max_item_length}"
            )
        
        if item:  # Skip empty strings
            validated.append(item)
    
    return validated


def validate_id_list(
    ids: List[str],
    max_items: int = 100,
    field_name: str = "IDs"
) -> List[str]:
    """
    Validate a list of UUIDs.
    
    Args:
        ids: List of UUID strings
        max_items: Maximum number of items
        field_name: Field name for error messages
    
    Returns:
        Validated list
    
    Raises:
        ValueError: If validation fails
    """
    import uuid
    
    if len(ids) > max_items:
        raise ValueError(f"{field_name} exceeds maximum of {max_items}")
    
    validated = []
    for i, id_str in enumerate(ids):
        try:
            uuid.UUID(id_str)
            validated.append(id_str)
        except ValueError:
            raise ValueError(f"{field_name}[{i}] is not a valid UUID")
    
    return validated


# =============================================================================
# Pydantic Validators (for use in schemas)
# =============================================================================

def username_validator(cls, v):
    """Pydantic validator for username"""
    return validate_username(v)


def email_validator(cls, v):
    """Pydantic validator for email"""
    return validate_email(v)


def slug_validator(cls, v):
    """Pydantic validator for slug"""
    return validate_slug(v)


def hex_color_validator(cls, v):
    """Pydantic validator for hex color"""
    if v is None:
        return v
    return validate_hex_color(v)


def url_validator(cls, v):
    """Pydantic validator for URL"""
    if v is None:
        return v
    return validate_url(v)
