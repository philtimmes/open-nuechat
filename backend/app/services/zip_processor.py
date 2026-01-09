"""
Zip file processing service.
Extracts files, generates signatures (functions, classes, exports), and creates file manifests.
"""

import zipfile
import io
import re
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeSignature:
    """A signature extracted from code (function, class, variable, etc.)"""
    name: str
    kind: str  # 'function', 'class', 'method', 'variable', 'export', 'import', 'interface', 'type'
    line: int
    signature: Optional[str] = None  # Full signature line
    docstring: Optional[str] = None
    

@dataclass
class FileInfo:
    """Information about a file in the zip"""
    path: str
    filename: str
    extension: str
    size: int
    language: Optional[str] = None
    is_binary: bool = False
    signatures: List[CodeSignature] = None
    content: Optional[str] = None  # Only for text files under size limit
    
    def __post_init__(self):
        if self.signatures is None:
            self.signatures = []


@dataclass
class ZipManifest:
    """Complete manifest of a processed zip file"""
    total_files: int
    total_size: int
    files: List[FileInfo]
    signature_index: Dict[str, List[Dict]]  # filename -> signatures
    file_tree: Dict  # Nested dict representing directory structure
    languages: Dict[str, int]  # language -> count
    

# Extension to language mapping
EXT_TO_LANG = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.cs': 'csharp',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.hpp': 'cpp',
    '.swift': 'swift',
    '.scala': 'scala',
    '.lua': 'lua',
    '.r': 'r',
    '.sql': 'sql',
    '.sh': 'bash',
    '.bash': 'bash',
    '.zsh': 'zsh',
    '.ps1': 'powershell',
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'sass',
    '.less': 'less',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.xml': 'xml',
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.txt': 'text',
    '.env': 'dotenv',
    '.vue': 'vue',
    '.svelte': 'svelte',
    '.ex': 'elixir',
    '.exs': 'elixir',
    '.erl': 'erlang',
    '.hs': 'haskell',
    '.ml': 'ocaml',
    '.clj': 'clojure',
    '.pl': 'perl',
    '.pm': 'perl',
}


def detect_language(filename: str) -> Optional[str]:
    """
    Detect programming language from filename extension.
    
    Args:
        filename: The filename to detect language for
        
    Returns:
        Language name string or None if not detected
    """
    if not filename:
        return None
    ext = os.path.splitext(filename)[1].lower()
    return EXT_TO_LANG.get(ext)


def extract_signatures(content: str, language: str) -> List['CodeSignature']:
    """
    Extract code signatures (functions, classes, etc.) from source code.
    
    Args:
        content: The source code content
        language: The programming language name
        
    Returns:
        List of CodeSignature objects
    """
    return SignatureExtractor.extract(content, language)


# Binary file extensions to skip content extraction
BINARY_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.svg',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.webm',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.exe', '.dll', '.so', '.dylib', '.bin',
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    '.pyc', '.pyo', '.class', '.o', '.obj',
    '.db', '.sqlite', '.sqlite3',
}

# Maximum file size for content extraction (500KB)
MAX_CONTENT_SIZE = 500 * 1024

# Security limits
MAX_FILES_IN_ARCHIVE = 10000
MAX_TOTAL_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500MB
MAX_PATH_DEPTH = 50
MAX_PATH_COMPONENT_LENGTH = 255


class ZipSecurityError(Exception):
    """Raised when a zip file fails security validation"""
    pass


def validate_zip_path(filename: str) -> None:
    """
    Validate a zip entry path for security issues.
    
    Checks for:
    - Null bytes
    - Absolute paths
    - Directory traversal (.. components)
    - Excessive path depth
    - Overly long path components
    
    Args:
        filename: The path from the zip entry
        
    Raises:
        ZipSecurityError: If the path fails validation
    """
    # Check for null bytes
    if '\x00' in filename:
        raise ZipSecurityError(f"Null byte in filename: {repr(filename)}")
    
    # Check for absolute paths
    if filename.startswith('/') or filename.startswith('\\'):
        raise ZipSecurityError(f"Absolute path not allowed: {filename}")
    
    # Windows absolute path check
    if len(filename) > 1 and filename[1] == ':':
        raise ZipSecurityError(f"Windows absolute path not allowed: {filename}")
    
    # Normalize path separators and split
    normalized = filename.replace('\\', '/')
    components = normalized.split('/')
    
    # Check path depth
    if len(components) > MAX_PATH_DEPTH:
        raise ZipSecurityError(f"Path too deep ({len(components)} levels): {filename}")
    
    # Check each component
    for component in components:
        # Directory traversal check
        if component == '..':
            raise ZipSecurityError(f"Directory traversal not allowed: {filename}")
        
        # Component length check
        if len(component) > MAX_PATH_COMPONENT_LENGTH:
            raise ZipSecurityError(f"Path component too long ({len(component)} chars): {component[:50]}...")


def is_symlink(zip_info: zipfile.ZipInfo) -> bool:
    """
    Check if a zip entry is a symbolic link.
    
    Symlinks in zip files have a special external_attr that indicates
    Unix file mode with symlink bit set.
    
    Args:
        zip_info: ZipInfo object from the archive
        
    Returns:
        True if the entry appears to be a symlink
    """
    # Unix symlink: external_attr high 16 bits contain Unix mode
    # Symlink mode is 0o120000
    unix_mode = (zip_info.external_attr >> 16) & 0xFFFF
    return (unix_mode & 0o170000) == 0o120000


def validate_zip_archive(zf: zipfile.ZipFile) -> None:
    """
    Validate an entire zip archive for security issues.
    
    Checks:
    - Total number of files
    - Total uncompressed size
    - Each file path
    - Symlinks
    
    Args:
        zf: Open ZipFile object
        
    Raises:
        ZipSecurityError: If the archive fails validation
    """
    info_list = zf.infolist()
    
    # Check file count
    if len(info_list) > MAX_FILES_IN_ARCHIVE:
        raise ZipSecurityError(
            f"Too many files in archive: {len(info_list)} (max: {MAX_FILES_IN_ARCHIVE})"
        )
    
    # Calculate total uncompressed size and validate each entry
    total_size = 0
    for info in info_list:
        # Validate path
        validate_zip_path(info.filename)
        
        # Check for symlinks
        if is_symlink(info):
            raise ZipSecurityError(f"Symlinks not allowed: {info.filename}")
        
        # Accumulate size
        total_size += info.file_size
        
        # Early exit if size exceeded
        if total_size > MAX_TOTAL_UNCOMPRESSED_SIZE:
            raise ZipSecurityError(
                f"Total uncompressed size exceeds limit: {total_size} bytes "
                f"(max: {MAX_TOTAL_UNCOMPRESSED_SIZE})"
            )


class SignatureExtractor:
    """Extract code signatures from various programming languages"""
    
    @staticmethod
    def extract_python(content: str) -> List[CodeSignature]:
        """Extract Python functions, classes, and variables"""
        signatures = []
        lines = content.split('\n')
        
        # Function pattern: def name(args):
        func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\((.*?)\).*?:', re.MULTILINE)
        # Class pattern: class Name(bases):
        class_pattern = re.compile(r'^(\s*)class\s+(\w+)(?:\s*\((.*?)\))?\s*:', re.MULTILINE)
        # Variable assignment at module level
        var_pattern = re.compile(r'^([A-Z][A-Z0-9_]*)\s*=', re.MULTILINE)
        
        for match in func_pattern.finditer(content):
            indent, name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            kind = 'method' if indent else 'function'
            signatures.append(CodeSignature(
                name=name,
                kind=kind,
                line=line_num,
                signature=f"def {name}({args})"
            ))
        
        for match in class_pattern.finditer(content):
            indent, name, bases = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}({bases or ''})"
            ))
        
        for match in var_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='variable',
                line=line_num,
            ))
        
        return signatures
    
    @staticmethod
    def extract_javascript(content: str) -> List[CodeSignature]:
        """Extract JavaScript/TypeScript functions, classes, and exports"""
        signatures = []
        
        # Function declarations: function name(args)
        func_pattern = re.compile(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\((.*?)\)', re.MULTILINE)
        # Arrow functions: const name = (args) => or const name = async (args) =>
        arrow_pattern = re.compile(r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>', re.MULTILINE)
        # Class declarations
        class_pattern = re.compile(r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?', re.MULTILINE)
        # Interface declarations (TypeScript)
        interface_pattern = re.compile(r'(?:export\s+)?interface\s+(\w+)', re.MULTILINE)
        # Type declarations (TypeScript)
        type_pattern = re.compile(r'(?:export\s+)?type\s+(\w+)\s*=', re.MULTILINE)
        # Export patterns
        export_pattern = re.compile(r'export\s+(?:default\s+)?(?:const|let|var|function|class)\s+(\w+)', re.MULTILINE)
        # Import patterns
        import_pattern = re.compile(r'import\s+(?:{([^}]+)}|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE)
        
        for match in func_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"function {name}({args})"
            ))
        
        for match in arrow_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"const {name} = ({args}) =>"
            ))
        
        for match in class_pattern.finditer(content):
            name, extends = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            sig = f"class {name}"
            if extends:
                sig += f" extends {extends}"
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=sig
            ))
        
        for match in interface_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='interface',
                line=line_num,
                signature=f"interface {name}"
            ))
        
        for match in type_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='type',
                line=line_num,
                signature=f"type {name}"
            ))
        
        for match in import_pattern.finditer(content):
            named, default, source = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            if named:
                for item in named.split(','):
                    item = item.strip().split(' as ')[0].strip()
                    if item:
                        signatures.append(CodeSignature(
                            name=item,
                            kind='import',
                            line=line_num,
                            signature=f"import {{ {item} }} from '{source}'"
                        ))
            if default:
                signatures.append(CodeSignature(
                    name=default,
                    kind='import',
                    line=line_num,
                    signature=f"import {default} from '{source}'"
                ))
        
        return signatures
    
    @staticmethod
    def extract_go(content: str) -> List[CodeSignature]:
        """Extract Go functions, types, and structs"""
        signatures = []
        
        # Function: func name(args) returns
        func_pattern = re.compile(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)(?:\s*\(([^)]*)\)|\s*(\w+))?', re.MULTILINE)
        # Type declarations: type Name struct/interface
        type_pattern = re.compile(r'type\s+(\w+)\s+(struct|interface)', re.MULTILINE)
        # Package declaration
        package_pattern = re.compile(r'package\s+(\w+)', re.MULTILINE)
        
        for match in func_pattern.finditer(content):
            name = match.group(1)
            args = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"func {name}({args})"
            ))
        
        for match in type_pattern.finditer(content):
            name, kind = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind=kind,
                line=line_num,
                signature=f"type {name} {kind}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_rust(content: str) -> List[CodeSignature]:
        """Extract Rust functions, structs, enums, and traits"""
        signatures = []
        
        # Function: fn name(args) -> return
        func_pattern = re.compile(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)', re.MULTILINE)
        # Struct: struct Name
        struct_pattern = re.compile(r'(?:pub\s+)?struct\s+(\w+)', re.MULTILINE)
        # Enum: enum Name
        enum_pattern = re.compile(r'(?:pub\s+)?enum\s+(\w+)', re.MULTILINE)
        # Trait: trait Name
        trait_pattern = re.compile(r'(?:pub\s+)?trait\s+(\w+)', re.MULTILINE)
        # Impl: impl Name
        impl_pattern = re.compile(r'impl(?:<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)', re.MULTILINE)
        
        for match in func_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"fn {name}({args})"
            ))
        
        for match in struct_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='struct',
                line=line_num,
                signature=f"struct {name}"
            ))
        
        for match in enum_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='enum',
                line=line_num,
                signature=f"enum {name}"
            ))
        
        for match in trait_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='trait',
                line=line_num,
                signature=f"trait {name}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_java(content: str) -> List[CodeSignature]:
        """Extract Java classes, interfaces, and methods"""
        signatures = []
        
        # Class: public class Name extends/implements
        class_pattern = re.compile(r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)', re.MULTILINE)
        # Interface
        interface_pattern = re.compile(r'(?:public\s+)?interface\s+(\w+)', re.MULTILINE)
        # Method: access return name(args)
        method_pattern = re.compile(r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)
        
        for match in class_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}"
            ))
        
        for match in interface_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='interface',
                line=line_num,
                signature=f"interface {name}"
            ))
        
        for match in method_pattern.finditer(content):
            name, args = match.groups()
            # Skip constructors (same name as class)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='method',
                line=line_num,
                signature=f"{name}({args})"
            ))
        
        return signatures
    
    @classmethod
    def extract(cls, content: str, language: str) -> List[CodeSignature]:
        """Extract signatures based on language"""
        extractors = {
            'python': cls.extract_python,
            'javascript': cls.extract_javascript,
            'typescript': cls.extract_javascript,  # Same patterns work
            'go': cls.extract_go,
            'rust': cls.extract_rust,
            'java': cls.extract_java,
            'kotlin': cls.extract_kotlin,
            'c': cls.extract_c,
            'cpp': cls.extract_cpp,
            'csharp': cls.extract_csharp,
            'ruby': cls.extract_ruby,
            'php': cls.extract_php,
            'swift': cls.extract_swift,
            'scala': cls.extract_scala,
            'bash': cls.extract_shell,
            'zsh': cls.extract_shell,
            'elixir': cls.extract_elixir,
            'haskell': cls.extract_haskell,
            'vue': cls.extract_javascript,  # Vue files contain JS
            'svelte': cls.extract_javascript,  # Svelte files contain JS
        }
        
        extractor = extractors.get(language)
        if extractor:
            try:
                return extractor(content)
            except Exception as e:
                logger.warning(f"Error extracting signatures for {language}: {e}")
                return []
        return []
    
    @staticmethod
    def extract_c(content: str) -> List[CodeSignature]:
        """Extract C functions, structs, and typedefs"""
        signatures = []
        
        # Function: return_type name(args) - at start of line, not indented
        func_pattern = re.compile(
            r'^(?:static\s+)?(?:inline\s+)?(?:extern\s+)?'
            r'(?:const\s+)?(?:unsigned\s+)?(?:signed\s+)?'
            r'(?:void|int|char|float|double|long|short|struct\s+\w+|\w+_t|\w+)\s*\*?\s*'
            r'(\w+)\s*\(([^)]*)\)\s*(?:\{|;)',
            re.MULTILINE
        )
        # Struct: struct Name {
        struct_pattern = re.compile(r'(?:typedef\s+)?struct\s+(\w+)\s*\{', re.MULTILINE)
        # Typedef
        typedef_pattern = re.compile(r'typedef\s+(?:struct\s+\w+\s+)?(?:\w+\s+)+(\w+)\s*;', re.MULTILINE)
        # Enum
        enum_pattern = re.compile(r'(?:typedef\s+)?enum\s+(\w+)?\s*\{', re.MULTILINE)
        # Define
        define_pattern = re.compile(r'#define\s+(\w+)(?:\(([^)]*)\))?', re.MULTILINE)
        
        for match in func_pattern.finditer(content):
            name, args = match.groups()
            if name not in ('if', 'while', 'for', 'switch', 'return'):
                line_num = content[:match.start()].count('\n') + 1
                signatures.append(CodeSignature(
                    name=name,
                    kind='function',
                    line=line_num,
                    signature=f"{name}({args.strip()})"
                ))
        
        for match in struct_pattern.finditer(content):
            name = match.group(1)
            if name:
                line_num = content[:match.start()].count('\n') + 1
                signatures.append(CodeSignature(
                    name=name,
                    kind='struct',
                    line=line_num,
                    signature=f"struct {name}"
                ))
        
        for match in typedef_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='typedef',
                line=line_num,
                signature=f"typedef {name}"
            ))
        
        for match in enum_pattern.finditer(content):
            name = match.group(1)
            if name:
                line_num = content[:match.start()].count('\n') + 1
                signatures.append(CodeSignature(
                    name=name,
                    kind='enum',
                    line=line_num,
                    signature=f"enum {name}"
                ))
        
        for match in define_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            kind = 'macro' if args is not None else 'define'
            sig = f"#define {name}({args})" if args else f"#define {name}"
            signatures.append(CodeSignature(
                name=name,
                kind=kind,
                line=line_num,
                signature=sig
            ))
        
        return signatures
    
    @staticmethod
    def extract_cpp(content: str) -> List[CodeSignature]:
        """Extract C++ functions, classes, structs, and namespaces"""
        signatures = []
        
        # Class: class Name : public Base
        class_pattern = re.compile(
            r'(?:template\s*<[^>]*>\s*)?class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?',
            re.MULTILINE
        )
        # Namespace
        namespace_pattern = re.compile(r'namespace\s+(\w+)\s*\{', re.MULTILINE)
        # Function/Method with return type
        func_pattern = re.compile(
            r'^(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?(?:explicit\s+)?'
            r'(?:const\s+)?(?:\w+(?:::\w+)?(?:\s*<[^>]*>)?\s*[&*]?\s+)'
            r'(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        )
        # Template
        template_pattern = re.compile(r'template\s*<([^>]+)>', re.MULTILINE)
        
        for match in class_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}"
            ))
        
        for match in namespace_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='namespace',
                line=line_num,
                signature=f"namespace {name}"
            ))
        
        for match in func_pattern.finditer(content):
            name, args = match.groups()
            if name not in ('if', 'while', 'for', 'switch', 'return', 'class', 'struct'):
                line_num = content[:match.start()].count('\n') + 1
                signatures.append(CodeSignature(
                    name=name,
                    kind='function',
                    line=line_num,
                    signature=f"{name}({args.strip()})"
                ))
        
        # Also include C-style signatures
        signatures.extend(SignatureExtractor.extract_c(content))
        
        return signatures
    
    @staticmethod
    def extract_csharp(content: str) -> List[CodeSignature]:
        """Extract C# classes, interfaces, methods, and properties"""
        signatures = []
        
        # Class
        class_pattern = re.compile(
            r'(?:public|private|protected|internal)?\s*(?:abstract|sealed|static|partial)?\s*'
            r'class\s+(\w+)(?:<[^>]+>)?',
            re.MULTILINE
        )
        # Interface
        interface_pattern = re.compile(r'(?:public\s+)?interface\s+(\w+)(?:<[^>]+>)?', re.MULTILINE)
        # Struct
        struct_pattern = re.compile(r'(?:public\s+)?struct\s+(\w+)', re.MULTILINE)
        # Method
        method_pattern = re.compile(
            r'(?:public|private|protected|internal)\s+(?:static\s+)?(?:virtual\s+)?(?:override\s+)?'
            r'(?:async\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        )
        # Property
        prop_pattern = re.compile(
            r'(?:public|private|protected|internal)\s+(?:static\s+)?(?:virtual\s+)?'
            r'(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\{\s*(?:get|set)',
            re.MULTILINE
        )
        # Namespace
        namespace_pattern = re.compile(r'namespace\s+([\w.]+)', re.MULTILINE)
        
        for match in class_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}"
            ))
        
        for match in interface_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='interface',
                line=line_num,
                signature=f"interface {name}"
            ))
        
        for match in struct_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='struct',
                line=line_num,
                signature=f"struct {name}"
            ))
        
        for match in method_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='method',
                line=line_num,
                signature=f"{name}({args.strip()})"
            ))
        
        for match in prop_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='property',
                line=line_num,
                signature=f"{name} {{ get; set; }}"
            ))
        
        for match in namespace_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='namespace',
                line=line_num,
                signature=f"namespace {name}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_ruby(content: str) -> List[CodeSignature]:
        """Extract Ruby classes, modules, and methods"""
        signatures = []
        
        # Class
        class_pattern = re.compile(r'class\s+(\w+)(?:\s*<\s*(\w+))?', re.MULTILINE)
        # Module
        module_pattern = re.compile(r'module\s+(\w+)', re.MULTILINE)
        # Method def
        method_pattern = re.compile(r'def\s+(?:self\.)?(\w+[?!=]?)\s*(?:\(([^)]*)\))?', re.MULTILINE)
        # attr_accessor, attr_reader, attr_writer
        attr_pattern = re.compile(r'attr_(?:accessor|reader|writer)\s+:(\w+)', re.MULTILINE)
        
        for match in class_pattern.finditer(content):
            name, parent = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            sig = f"class {name}"
            if parent:
                sig += f" < {parent}"
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=sig
            ))
        
        for match in module_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='module',
                line=line_num,
                signature=f"module {name}"
            ))
        
        for match in method_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            sig = f"def {name}"
            if args:
                sig += f"({args})"
            signatures.append(CodeSignature(
                name=name,
                kind='method',
                line=line_num,
                signature=sig
            ))
        
        for match in attr_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='attribute',
                line=line_num,
                signature=f"attr :{name}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_php(content: str) -> List[CodeSignature]:
        """Extract PHP classes, interfaces, and functions"""
        signatures = []
        
        # Class
        class_pattern = re.compile(
            r'(?:abstract\s+)?(?:final\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?',
            re.MULTILINE
        )
        # Interface
        interface_pattern = re.compile(r'interface\s+(\w+)', re.MULTILINE)
        # Trait
        trait_pattern = re.compile(r'trait\s+(\w+)', re.MULTILINE)
        # Function
        func_pattern = re.compile(
            r'(?:public|private|protected|static|\s)*function\s+(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        )
        # Namespace
        namespace_pattern = re.compile(r'namespace\s+([\w\\]+)\s*;', re.MULTILINE)
        
        for match in class_pattern.finditer(content):
            name, extends = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            sig = f"class {name}"
            if extends:
                sig += f" extends {extends}"
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=sig
            ))
        
        for match in interface_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='interface',
                line=line_num,
                signature=f"interface {name}"
            ))
        
        for match in trait_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='trait',
                line=line_num,
                signature=f"trait {name}"
            ))
        
        for match in func_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"function {name}({args.strip()})"
            ))
        
        for match in namespace_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='namespace',
                line=line_num,
                signature=f"namespace {name}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_swift(content: str) -> List[CodeSignature]:
        """Extract Swift classes, structs, protocols, and functions"""
        signatures = []
        
        # Class
        class_pattern = re.compile(
            r'(?:public\s+|private\s+|internal\s+|open\s+|final\s+)*class\s+(\w+)',
            re.MULTILINE
        )
        # Struct
        struct_pattern = re.compile(
            r'(?:public\s+|private\s+|internal\s+)*struct\s+(\w+)',
            re.MULTILINE
        )
        # Protocol
        protocol_pattern = re.compile(r'protocol\s+(\w+)', re.MULTILINE)
        # Enum
        enum_pattern = re.compile(r'(?:public\s+|private\s+)?enum\s+(\w+)', re.MULTILINE)
        # Function
        func_pattern = re.compile(
            r'(?:public\s+|private\s+|internal\s+|open\s+|override\s+|static\s+|class\s+)*'
            r'func\s+(\w+)\s*(?:<[^>]+>)?\s*\(([^)]*)\)',
            re.MULTILINE
        )
        # Extension
        extension_pattern = re.compile(r'extension\s+(\w+)', re.MULTILINE)
        
        for match in class_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}"
            ))
        
        for match in struct_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='struct',
                line=line_num,
                signature=f"struct {name}"
            ))
        
        for match in protocol_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='protocol',
                line=line_num,
                signature=f"protocol {name}"
            ))
        
        for match in enum_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='enum',
                line=line_num,
                signature=f"enum {name}"
            ))
        
        for match in func_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"func {name}({args.strip()})"
            ))
        
        for match in extension_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='extension',
                line=line_num,
                signature=f"extension {name}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_kotlin(content: str) -> List[CodeSignature]:
        """Extract Kotlin classes, objects, and functions"""
        signatures = []
        
        # Class
        class_pattern = re.compile(
            r'(?:public\s+|private\s+|internal\s+|open\s+|abstract\s+|sealed\s+|data\s+)*'
            r'class\s+(\w+)',
            re.MULTILINE
        )
        # Object
        object_pattern = re.compile(r'(?:companion\s+)?object\s+(\w+)?', re.MULTILINE)
        # Interface
        interface_pattern = re.compile(r'interface\s+(\w+)', re.MULTILINE)
        # Function
        func_pattern = re.compile(
            r'(?:public\s+|private\s+|internal\s+|override\s+|suspend\s+)*'
            r'fun\s+(?:<[^>]+>\s*)?(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        )
        # Property
        prop_pattern = re.compile(
            r'(?:public\s+|private\s+|internal\s+|override\s+)*'
            r'(?:val|var)\s+(\w+)\s*:',
            re.MULTILINE
        )
        
        for match in class_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}"
            ))
        
        for match in object_pattern.finditer(content):
            name = match.group(1)
            if name:
                line_num = content[:match.start()].count('\n') + 1
                signatures.append(CodeSignature(
                    name=name,
                    kind='object',
                    line=line_num,
                    signature=f"object {name}"
                ))
        
        for match in interface_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='interface',
                line=line_num,
                signature=f"interface {name}"
            ))
        
        for match in func_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"fun {name}({args.strip()})"
            ))
        
        return signatures
    
    @staticmethod
    def extract_scala(content: str) -> List[CodeSignature]:
        """Extract Scala classes, objects, traits, and functions"""
        signatures = []
        
        # Class
        class_pattern = re.compile(r'(?:case\s+)?class\s+(\w+)', re.MULTILINE)
        # Object
        object_pattern = re.compile(r'(?:case\s+)?object\s+(\w+)', re.MULTILINE)
        # Trait
        trait_pattern = re.compile(r'trait\s+(\w+)', re.MULTILINE)
        # Def
        def_pattern = re.compile(r'def\s+(\w+)\s*(?:\[([^\]]*)\])?\s*\(([^)]*)\)', re.MULTILINE)
        # Val/Var
        val_pattern = re.compile(r'(?:val|var)\s+(\w+)\s*:', re.MULTILINE)
        
        for match in class_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}"
            ))
        
        for match in object_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='object',
                line=line_num,
                signature=f"object {name}"
            ))
        
        for match in trait_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='trait',
                line=line_num,
                signature=f"trait {name}"
            ))
        
        for match in def_pattern.finditer(content):
            name, types, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"def {name}({args.strip()})"
            ))
        
        return signatures
    
    @staticmethod
    def extract_shell(content: str) -> List[CodeSignature]:
        """Extract shell functions"""
        signatures = []
        
        # Function: name() { or function name {
        func_pattern1 = re.compile(r'^(\w+)\s*\(\)\s*\{', re.MULTILINE)
        func_pattern2 = re.compile(r'^function\s+(\w+)\s*(?:\(\))?\s*\{', re.MULTILINE)
        # Exported variables
        export_pattern = re.compile(r'^export\s+(\w+)=', re.MULTILINE)
        
        for match in func_pattern1.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"{name}()"
            ))
        
        for match in func_pattern2.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"function {name}"
            ))
        
        for match in export_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='export',
                line=line_num,
                signature=f"export {name}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_elixir(content: str) -> List[CodeSignature]:
        """Extract Elixir modules and functions"""
        signatures = []
        
        # Module
        module_pattern = re.compile(r'defmodule\s+([\w.]+)', re.MULTILINE)
        # Function def
        def_pattern = re.compile(r'(?:def|defp)\s+(\w+)\s*(?:\(([^)]*)\))?', re.MULTILINE)
        # Macro
        macro_pattern = re.compile(r'defmacro\s+(\w+)', re.MULTILINE)
        
        for match in module_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='module',
                line=line_num,
                signature=f"defmodule {name}"
            ))
        
        for match in def_pattern.finditer(content):
            name, args = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            sig = f"def {name}"
            if args:
                sig += f"({args})"
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=sig
            ))
        
        for match in macro_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='macro',
                line=line_num,
                signature=f"defmacro {name}"
            ))
        
        return signatures
    
    @staticmethod
    def extract_haskell(content: str) -> List[CodeSignature]:
        """Extract Haskell type signatures and functions"""
        signatures = []
        
        # Type signature: name :: Type
        type_sig_pattern = re.compile(r'^(\w+)\s*::\s*(.+)$', re.MULTILINE)
        # Data type
        data_pattern = re.compile(r'^data\s+(\w+)', re.MULTILINE)
        # Newtype
        newtype_pattern = re.compile(r'^newtype\s+(\w+)', re.MULTILINE)
        # Type alias
        type_pattern = re.compile(r'^type\s+(\w+)', re.MULTILINE)
        # Class
        class_pattern = re.compile(r'^class\s+(?:\([^)]+\)\s*=>)?\s*(\w+)', re.MULTILINE)
        # Instance
        instance_pattern = re.compile(r'^instance\s+(?:\([^)]+\)\s*=>)?\s*(\w+)\s+(\w+)', re.MULTILINE)
        # Module
        module_pattern = re.compile(r'^module\s+([\w.]+)', re.MULTILINE)
        
        for match in type_sig_pattern.finditer(content):
            name, type_info = match.groups()
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='function',
                line=line_num,
                signature=f"{name} :: {type_info[:50]}"
            ))
        
        for match in data_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='data',
                line=line_num,
                signature=f"data {name}"
            ))
        
        for match in newtype_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='newtype',
                line=line_num,
                signature=f"newtype {name}"
            ))
        
        for match in type_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='type',
                line=line_num,
                signature=f"type {name}"
            ))
        
        for match in class_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='class',
                line=line_num,
                signature=f"class {name}"
            ))
        
        for match in module_pattern.finditer(content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            signatures.append(CodeSignature(
                name=name,
                kind='module',
                line=line_num,
                signature=f"module {name}"
            ))
        
        return signatures


class ZipProcessor:
    """Process zip files and extract metadata"""
    
    def __init__(self, max_content_size: int = MAX_CONTENT_SIZE):
        self.max_content_size = max_content_size
    
    def process(self, zip_data: bytes) -> ZipManifest:
        """Process a zip file and return the manifest"""
        files: List[FileInfo] = []
        signature_index: Dict[str, List[Dict]] = {}
        languages: Dict[str, int] = {}
        total_size = 0
        
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            # Security validation first
            validate_zip_archive(zf)
            
            for info in zf.infolist():
                # Skip directories
                if info.is_dir():
                    continue
                
                # Skip hidden files and common non-code directories
                path_parts = info.filename.split('/')
                if any(part.startswith('.') for part in path_parts):
                    continue
                if any(part in ('node_modules', '__pycache__', '.git', 'venv', 'env', 'dist', 'build', 'target') for part in path_parts):
                    continue
                
                filename = os.path.basename(info.filename)
                ext = os.path.splitext(filename)[1].lower()
                language = EXT_TO_LANG.get(ext)
                is_binary = ext in BINARY_EXTENSIONS
                
                file_info = FileInfo(
                    path=info.filename,
                    filename=filename,
                    extension=ext,
                    size=info.file_size,
                    language=language,
                    is_binary=is_binary,
                )
                
                total_size += info.file_size
                
                # Track languages
                if language:
                    languages[language] = languages.get(language, 0) + 1
                
                # Extract content for text files under size limit
                if not is_binary and info.file_size <= self.max_content_size:
                    try:
                        content = zf.read(info.filename).decode('utf-8', errors='replace')
                        file_info.content = content
                        
                        # Extract signatures for code files
                        if language:
                            signatures = SignatureExtractor.extract(content, language)
                            file_info.signatures = signatures
                            if signatures:
                                signature_index[info.filename] = [asdict(s) for s in signatures]
                    except Exception as e:
                        logger.warning(f"Error reading {info.filename}: {e}")
                
                files.append(file_info)
        
        # Build file tree
        file_tree = self._build_file_tree(files)
        
        return ZipManifest(
            total_files=len(files),
            total_size=total_size,
            files=files,
            signature_index=signature_index,
            file_tree=file_tree,
            languages=languages,
        )
    
    def _build_file_tree(self, files: List[FileInfo]) -> Dict:
        """Build a nested dictionary representing the file tree"""
        tree = {}
        
        for file_info in files:
            parts = file_info.path.split('/')
            current = tree
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Leaf node (file)
                    current[part] = {
                        '_type': 'file',
                        '_path': file_info.path,
                        '_size': file_info.size,
                        '_language': file_info.language,
                    }
                else:
                    # Directory
                    if part not in current:
                        current[part] = {'_type': 'directory'}
                    current = current[part]
        
        return tree
    
    def to_artifacts(self, manifest: ZipManifest) -> List[Dict[str, Any]]:
        """Convert processed files to artifact format"""
        artifacts = []
        
        for file_info in manifest.files:
            if file_info.content is not None:
                artifact = {
                    'id': f"zip-{hash(file_info.path) & 0xFFFFFFFF}",
                    'title': file_info.filename,
                    'filename': file_info.path,
                    'type': self._get_artifact_type(file_info.language, file_info.extension),
                    'language': file_info.language,
                    'content': file_info.content,
                    'size': file_info.size,
                    'signatures': [asdict(s) for s in file_info.signatures] if file_info.signatures else [],
                }
                artifacts.append(artifact)
        
        return artifacts
    
    def _get_artifact_type(self, language: Optional[str], ext: str) -> str:
        """Map language/extension to artifact type"""
        if ext == '.html':
            return 'html'
        elif ext in ('.jsx', '.tsx'):
            return 'react'
        elif ext == '.svg':
            return 'svg'
        elif ext in ('.md', '.markdown'):
            return 'markdown'
        elif ext == '.json':
            return 'json'
        elif ext == '.csv':
            return 'csv'
        elif ext == '.mmd' or ext == '.mermaid':
            return 'mermaid'
        else:
            return 'code'


def format_signature_summary(manifest: ZipManifest) -> str:
    """Format a readable summary of signatures for LLM context"""
    lines = [
        f"# Zip Archive Summary",
        f"",
        f"**Files:** {manifest.total_files}",
        f"**Total Size:** {manifest.total_size:,} bytes",
        f"**Languages:** {', '.join(f'{lang} ({count})' for lang, count in sorted(manifest.languages.items(), key=lambda x: -x[1]))}",
        f"",
        f"## File Structure",
        f"```",
    ]
    
    # Add file tree
    def format_tree(tree: Dict, prefix: str = "") -> List[str]:
        result = []
        items = [(k, v) for k, v in tree.items() if not k.startswith('_')]
        dirs = [(k, v) for k, v in items if v.get('_type') == 'directory']
        files = [(k, v) for k, v in items if v.get('_type') == 'file']
        
        # Sort directories first, then files
        for name, node in sorted(dirs) + sorted(files):
            if node.get('_type') == 'file':
                size = node.get('_size', 0)
                lang = node.get('_language', '')
                result.append(f"{prefix}{name} ({size:,} bytes, {lang})")
            else:
                result.append(f"{prefix}{name}/")
                result.extend(format_tree(node, prefix + "  "))
        return result
    
    lines.extend(format_tree(manifest.file_tree))
    lines.append("```")
    lines.append("")
    
    # Add signature index
    if manifest.signature_index:
        lines.append("## Code Signatures")
        lines.append("")
        
        for filepath, sigs in sorted(manifest.signature_index.items()):
            if sigs:
                lines.append(f"### {filepath}")
                for sig in sigs:
                    kind = sig.get('kind', 'unknown')
                    name = sig.get('name', 'unknown')
                    line = sig.get('line', 0)
                    signature = sig.get('signature', name)
                    lines.append(f"- **{kind}** `{signature}` (line {line})")
                lines.append("")
    
    return "\n".join(lines)


def format_llm_manifest(manifest: ZipManifest, filename: str, upload_timestamp: str, include_small_files: bool = True, small_file_threshold: int = 4000) -> str:
    """
    Format a manifest specifically for LLM context injection.
    Includes clear provenance markers, full signatures, and optionally small file contents.
    
    Args:
        manifest: The processed zip manifest
        filename: Original zip filename
        upload_timestamp: When the file was uploaded
        include_small_files: If True, include contents of small files directly
        small_file_threshold: Files smaller than this (bytes) are included inline
    """
    lines = [
        f"[USER_ARCHIVE: {filename} | uploaded: {upload_timestamp}]",
        "This archive was uploaded by the user. Contents are READ-ONLY source material.",
        "Do not claim to have created or modified these files unless you generate new artifacts.",
        "",
        f"FILES: {manifest.total_files} | SIZE: {manifest.total_size:,} bytes",
        f"LANGUAGES: {', '.join(f'{lang}({count})' for lang, count in sorted(manifest.languages.items(), key=lambda x: -x[1]))}",
        "",
        "STRUCTURE:",
    ]
    
    # Compact tree format
    def format_compact_tree(tree: Dict, prefix: str = "") -> List[str]:
        result = []
        items = [(k, v) for k, v in tree.items() if not k.startswith('_')]
        dirs = [(k, v) for k, v in items if v.get('_type') == 'directory']
        files = [(k, v) for k, v in items if v.get('_type') == 'file']
        
        for name, node in sorted(dirs) + sorted(files):
            if node.get('_type') == 'file':
                size = node.get('_size', 0)
                lang = node.get('_language', '')
                # Count lines if content available
                line_info = ""
                result.append(f"{prefix}{name} ({size:,}b, {lang})" if lang else f"{prefix}{name} ({size:,}b)")
            else:
                result.append(f"{prefix}{name}/")
                result.extend(format_compact_tree(node, prefix + "  "))
        return result
    
    lines.extend(format_compact_tree(manifest.file_tree))
    lines.append("")
    
    # FULL Signatures section - more detailed for LLM context
    if manifest.signature_index:
        lines.append("=" * 60)
        lines.append("CODE SIGNATURES (functions, classes, types defined in each file):")
        lines.append("=" * 60)
        for filepath, sigs in sorted(manifest.signature_index.items()):
            if sigs:
                lines.append(f"\n### {filepath}")
                for sig in sigs[:20]:  # Limit to 20 signatures per file
                    kind = sig.get('kind', '?')
                    name = sig.get('name', '?')
                    line = sig.get('line', 0)
                    params = sig.get('params', '')
                    return_type = sig.get('return_type', '')
                    
                    # Format based on kind
                    if kind in ('function', 'method'):
                        sig_str = f"  Line {line}: {kind} {name}({params})"
                        if return_type:
                            sig_str += f" -> {return_type}"
                    elif kind == 'class':
                        sig_str = f"  Line {line}: class {name}"
                        if params:  # inheritance
                            sig_str += f" : {params}"
                    elif kind in ('interface', 'struct', 'type'):
                        sig_str = f"  Line {line}: {kind} {name}"
                    else:
                        sig_str = f"  Line {line}: {kind} {name}"
                    
                    lines.append(sig_str)
                
                if len(sigs) > 20:
                    lines.append(f"  ... and {len(sigs) - 20} more")
        lines.append("")
    
    # Include small files directly in the manifest
    if include_small_files:
        small_files = []
        for file_info in manifest.files:
            if file_info.content and file_info.size <= small_file_threshold:
                small_files.append(file_info)
        
        if small_files:
            lines.append("=" * 60)
            lines.append(f"SMALL FILE CONTENTS (files under {small_file_threshold} bytes, included for quick reference):")
            lines.append("=" * 60)
            
            for file_info in small_files[:15]:  # Limit to 15 small files
                lang = file_info.language or ''
                lines.append(f"\n### {file_info.path} ({file_info.size} bytes)")
                lines.append(f"```{lang}")
                lines.append(file_info.content)
                lines.append("```")
            
            if len(small_files) > 15:
                lines.append(f"\n... and {len(small_files) - 15} more small files (use request_file to view)")
            lines.append("")
    
    # Instructions
    lines.extend([
        "─" * 60,
        "FILE ACCESS INSTRUCTIONS:",
        "• Small files shown above - use their content directly",
        "• For larger files: <request_file path=\"path/to/file\"/>",
        "• To modify: Create an artifact with FULL file content (same filename)",
        "• NEVER use placeholders like '// ... existing code ...'",
        "• NEVER truncate code - always output complete files",
        "[END_ARCHIVE_MANIFEST]",
    ])
    
    return "\n".join(lines)


def format_file_content_for_llm(filepath: str, content: str, source: str = "user_upload") -> str:
    """
    Format file content for injection into LLM context.
    Includes clear provenance markers to prevent mimicry.
    """
    # Detect language for syntax highlighting hint
    ext = os.path.splitext(filepath)[1].lower()
    lang = EXT_TO_LANG.get(ext, '')
    
    lines = [
        f"[UPLOADED_FILE_CONTENT: {filepath}]",
        f"[SOURCE: {source} - This is external content, not AI-generated]",
        f"```{lang}",
        content,
        "```",
        "[END_UPLOADED_FILE]",
    ]
    
    return "\n".join(lines)
