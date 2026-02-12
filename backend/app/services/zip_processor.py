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
    Also handles special filenames like Dockerfile, Makefile, etc.
    """
    if not filename:
        return None
    
    basename = os.path.basename(filename).lower()
    
    # Special filename detection
    if basename == 'dockerfile' or basename.startswith('dockerfile.'):
        return 'dockerfile'
    if basename == 'makefile' or basename.startswith('makefile.') or basename == 'gnumakefile':
        return 'makefile'
    if basename == 'docker-compose.yml' or basename == 'docker-compose.yaml' or basename.startswith('compose.'):
        return 'docker-compose'
    if basename == 'cmakelists.txt':
        return 'cmake'
    
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
            'typescript': cls.extract_javascript,
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
            'vue': cls.extract_javascript,
            'svelte': cls.extract_javascript,
            'dockerfile': cls.extract_dockerfile,
            'makefile': cls.extract_makefile,
            'docker-compose': cls.extract_docker_compose,
            'markdown': cls.extract_markdown_gist,
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
    def extract_dockerfile(content: str) -> List[CodeSignature]:
        """Extract Dockerfile stages, exposed ports, base images, and key instructions"""
        signatures = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # FROM — base image, possibly named stage
            m = re.match(r'^FROM\s+(\S+)(?:\s+[Aa][Ss]\s+(\w+))?', stripped, re.IGNORECASE)
            if m:
                image, stage = m.groups()
                name = f"stage:{stage}" if stage else f"base:{image}"
                signatures.append(CodeSignature(
                    name=name, kind='stage' if stage else 'from',
                    line=i, signature=stripped,
                ))
            
            # EXPOSE
            m = re.match(r'^EXPOSE\s+(.+)', stripped, re.IGNORECASE)
            if m:
                signatures.append(CodeSignature(name=m.group(1).strip(), kind='expose', line=i))
            
            # ENTRYPOINT / CMD
            m = re.match(r'^(ENTRYPOINT|CMD)\s+(.+)', stripped, re.IGNORECASE)
            if m:
                signatures.append(CodeSignature(
                    name=m.group(2).strip()[:80], kind=m.group(1).lower(), line=i,
                ))
            
            # COPY --from (multi-stage reference)
            m = re.match(r'^COPY\s+--from=(\S+)', stripped, re.IGNORECASE)
            if m:
                signatures.append(CodeSignature(
                    name=f"from:{m.group(1)}", kind='copy_from', line=i, signature=stripped,
                ))
            
            # ENV key=value (important config)
            m = re.match(r'^ENV\s+(\w+)[=\s](.+)', stripped, re.IGNORECASE)
            if m:
                signatures.append(CodeSignature(name=m.group(1), kind='env', line=i))
            
            # WORKDIR
            m = re.match(r'^WORKDIR\s+(\S+)', stripped, re.IGNORECASE)
            if m:
                signatures.append(CodeSignature(name=m.group(1), kind='workdir', line=i))
        
        return signatures
    
    @staticmethod
    def extract_makefile(content: str) -> List[CodeSignature]:
        """Extract Makefile targets, variables, and includes"""
        signatures = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.rstrip()
            if not stripped or stripped.startswith('\t') or stripped.startswith('#'):
                continue
            
            # Targets: name: [deps]
            m = re.match(r'^([a-zA-Z_][\w.-]*)\s*:(?!=)', stripped)
            if m:
                target = m.group(1)
                # Get deps
                deps_part = stripped[stripped.index(':') + 1:].strip()
                sig = f"{target}: {deps_part}" if deps_part else target
                signatures.append(CodeSignature(
                    name=target, kind='target', line=i, signature=sig,
                ))
            
            # Variables: NAME = value or NAME := value
            m = re.match(r'^([A-Z_][A-Z0-9_]*)\s*[:?]?=\s*(.+)', stripped)
            if m:
                val = m.group(2).strip()
                if len(val) > 60:
                    val = val[:60] + "..."
                signatures.append(CodeSignature(
                    name=m.group(1), kind='variable', line=i, signature=f"{m.group(1)} = {val}",
                ))
            
            # Include
            m = re.match(r'^-?include\s+(.+)', stripped)
            if m:
                signatures.append(CodeSignature(name=m.group(1).strip(), kind='include', line=i))
        
        return signatures
    
    @staticmethod
    def extract_docker_compose(content: str) -> List[CodeSignature]:
        """Extract docker-compose services, ports, volumes"""
        signatures = []
        lines = content.split('\n')
        current_service = None
        in_services = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.rstrip()
            
            # Top-level 'services:' key
            if re.match(r'^services:\s*$', stripped):
                in_services = True
                continue
            
            if in_services:
                # Service name (2-space indent, no further indent)
                m = re.match(r'^  ([a-zA-Z_][\w-]*):\s*$', stripped)
                if m:
                    current_service = m.group(1)
                    signatures.append(CodeSignature(name=current_service, kind='service', line=i))
                    continue
                
                if current_service:
                    # Image
                    m = re.match(r'^\s+image:\s*(.+)', stripped)
                    if m:
                        signatures.append(CodeSignature(
                            name=f"{current_service}→{m.group(1).strip()}", kind='image', line=i,
                        ))
                    # Build context
                    m = re.match(r'^\s+build:\s*(.+)', stripped)
                    if m and m.group(1).strip() != '':
                        signatures.append(CodeSignature(
                            name=f"{current_service}→build:{m.group(1).strip()}", kind='build', line=i,
                        ))
                    # Ports
                    m = re.match(r'^\s+-\s*["\']?(\d+:\d+)', stripped)
                    if m:
                        signatures.append(CodeSignature(
                            name=f"{current_service}:{m.group(1)}", kind='port', line=i,
                        ))
                    # Depends_on
                    m = re.match(r'^\s+-\s*(\w[\w-]*)\s*$', stripped)
                    if m and 'depends_on' in lines[max(0, i-3):i-1].__repr__():
                        signatures.append(CodeSignature(
                            name=f"{current_service}→{m.group(1)}", kind='depends_on', line=i,
                        ))
                
                # End of services block
                if re.match(r'^[a-z]', stripped) and not stripped.startswith(' '):
                    in_services = False
                    current_service = None
        
        return signatures
    
    @staticmethod
    def extract_markdown_gist(content: str, max_chars: int = 500) -> List[CodeSignature]:
        """Extract headings and first paragraph from markdown as a gist"""
        signatures = []
        lines = content.split('\n')
        
        # Extract headings
        for i, line in enumerate(lines, 1):
            m = re.match(r'^(#{1,3})\s+(.+)', line)
            if m:
                level = len(m.group(1))
                signatures.append(CodeSignature(
                    name=m.group(2).strip(), kind=f'h{level}', line=i,
                ))
        
        # Extract first non-heading, non-empty paragraph as docstring on the first heading
        first_para = []
        in_para = False
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('```'):
                in_para = True
                first_para.append(stripped)
                if len(' '.join(first_para)) > max_chars:
                    break
            elif in_para and not stripped:
                break
        
        if first_para and signatures:
            gist = ' '.join(first_para)[:max_chars]
            signatures[0] = CodeSignature(
                name=signatures[0].name, kind=signatures[0].kind,
                line=signatures[0].line, docstring=gist,
            )
        
        return signatures
    
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
                language = detect_language(info.filename)  # Handles Dockerfile, Makefile, etc.
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


def extract_associations(manifest: ZipManifest) -> Dict[str, Any]:
    """
    Extract cross-file associations and flows from a processed manifest.
    Builds: import graph, class hierarchies, service topology, entry points.
    """
    imports = {}  # file -> [imported modules/files]
    class_hierarchy = {}  # class -> parent(s)
    entry_points = []  # files that are likely entry points
    services = {}  # from docker-compose
    dockerfile_info = []  # from Dockerfiles
    
    project_files = {f.path for f in manifest.files}
    # Map basenames for local import resolution
    basename_to_path = {}
    for f in manifest.files:
        stem = os.path.splitext(f.filename)[0]
        basename_to_path[stem] = f.path
        basename_to_path[f.filename] = f.path
    
    for f in manifest.files:
        if not f.content or not f.language:
            continue
        
        file_imports = []
        
        # Python imports
        if f.language == 'python':
            for m in re.finditer(r'^(?:from\s+([\w.]+)\s+)?import\s+([\w., ]+)', f.content, re.MULTILINE):
                module = m.group(1) or m.group(2).split(',')[0].strip()
                # Check if it's a local import
                top_module = module.split('.')[0]
                if top_module in basename_to_path:
                    file_imports.append(basename_to_path[top_module])
                else:
                    file_imports.append(module)
            
            # Entry point detection
            if '__main__' in f.content or 'if __name__' in f.content:
                entry_points.append(f.path)
            if f.filename in ('main.py', 'app.py', 'server.py', 'manage.py', 'wsgi.py', 'asgi.py'):
                entry_points.append(f.path)
        
        # JS/TS imports
        elif f.language in ('javascript', 'typescript'):
            for m in re.finditer(r'(?:import|require)\s*(?:\(?\s*[\'"]([^\'"]+)[\'"]|.+from\s+[\'"]([^\'"]+)[\'"])', f.content):
                target = m.group(1) or m.group(2)
                if target.startswith('.'):
                    # Resolve relative import
                    base_dir = os.path.dirname(f.path)
                    resolved = os.path.normpath(os.path.join(base_dir, target))
                    # Try common extensions
                    for ext in ['', '.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.js']:
                        if resolved + ext in project_files:
                            file_imports.append(resolved + ext)
                            break
                    else:
                        file_imports.append(target)
                else:
                    file_imports.append(target)  # npm package
            
            if f.filename in ('index.js', 'index.ts', 'main.ts', 'main.js', 'app.ts', 'app.js', 'server.ts', 'server.js'):
                entry_points.append(f.path)
        
        # Go imports
        elif f.language == 'go':
            for m in re.finditer(r'"([^"]+)"', f.content[:2000]):
                file_imports.append(m.group(1))
            if f.filename == 'main.go':
                entry_points.append(f.path)
        
        if file_imports:
            imports[f.path] = file_imports
        
        # Class hierarchy from signatures
        if f.signatures:
            for sig in f.signatures:
                if sig.kind == 'class' and sig.signature:
                    # Extract parent from signature like "class Foo(Bar, Baz)"
                    m = re.search(r'\(([^)]+)\)', sig.signature)
                    if m:
                        parents = [p.strip() for p in m.group(1).split(',') if p.strip()]
                        if parents and parents != ['']:
                            class_hierarchy[f"{f.path}:{sig.name}"] = parents
        
        # Docker-compose services
        if f.language == 'docker-compose' and f.signatures:
            for sig in f.signatures:
                if sig.kind == 'service':
                    services[sig.name] = {'file': f.path, 'line': sig.line}
                elif sig.kind in ('image', 'build', 'port', 'depends_on'):
                    svc_name = sig.name.split('→')[0] if '→' in sig.name else sig.name.split(':')[0]
                    if svc_name in services:
                        services[svc_name][sig.kind] = sig.name
        
        # Dockerfile info
        if f.language == 'dockerfile' and f.signatures:
            info = {'file': f.path, 'stages': [], 'ports': [], 'cmd': None}
            for sig in f.signatures:
                if sig.kind in ('from', 'stage'):
                    info['stages'].append(sig.name)
                elif sig.kind == 'expose':
                    info['ports'].append(sig.name)
                elif sig.kind in ('cmd', 'entrypoint'):
                    info['cmd'] = sig.name
            dockerfile_info.append(info)
    
    return {
        'imports': imports,
        'class_hierarchy': class_hierarchy,
        'entry_points': list(set(entry_points)),
        'services': services,
        'dockerfile_info': dockerfile_info,
    }


def extract_call_graph(manifest: ZipManifest) -> Dict[str, List[str]]:
    """
    Extract function-level call graph from source files.
    Returns dict of 'file:function' -> ['file:called_function', ...].
    Static analysis — best effort, not perfect.
    """
    # Build lookup: function_name -> [(file, full_key)]
    func_index = {}  # name -> [(file_path, "file:name")]
    method_owners = {}  # (file, method_name) -> class_name
    
    for f in manifest.files:
        if not f.signatures:
            continue
        current_class = None
        for sig in f.signatures:
            if sig.kind == 'class':
                current_class = sig.name
            elif sig.kind in ('function', 'method'):
                key = f"{f.path}:{sig.name}"
                if sig.name not in func_index:
                    func_index[sig.name] = []
                func_index[sig.name].append((f.path, key))
                if sig.kind == 'method' and current_class:
                    method_owners[(f.path, sig.name)] = current_class
    
    call_graph = {}  # "file:func" -> ["file:called_func", ...]
    
    for f in manifest.files:
        if not f.content or not f.signatures or not f.language:
            continue
        
        lines = f.content.split('\n')
        
        # Build line ranges for each function/method in this file
        func_ranges = []  # [(start_line, end_line, key)]
        sorted_sigs = sorted(
            [s for s in f.signatures if s.kind in ('function', 'method')],
            key=lambda s: s.line
        )
        for i, sig in enumerate(sorted_sigs):
            start = sig.line - 1  # 0-indexed
            end = sorted_sigs[i + 1].line - 2 if i + 1 < len(sorted_sigs) else len(lines) - 1
            key = f"{f.path}:{sig.name}"
            func_ranges.append((start, end, key))
        
        if not func_ranges:
            continue
        
        # For each function body, find calls to known functions
        # Pattern matches: word( — function call syntax
        call_pattern = re.compile(r'(?<![.\w])(\w+)\s*\(')
        # self.method() or obj.method()
        method_call_pattern = re.compile(r'(?:self|this|cls)\s*\.\s*(\w+)\s*\(')
        
        for start, end, caller_key in func_ranges:
            calls = set()
            body = '\n'.join(lines[start:end + 1])
            
            # Direct function calls
            for m in call_pattern.finditer(body):
                called_name = m.group(1)
                # Skip built-ins and common non-functions
                if called_name in ('if', 'for', 'while', 'return', 'print', 'len', 'str',
                                   'int', 'float', 'list', 'dict', 'set', 'tuple', 'range',
                                   'isinstance', 'hasattr', 'getattr', 'setattr', 'super',
                                   'type', 'map', 'filter', 'sorted', 'enumerate', 'zip',
                                   'True', 'False', 'None', 'self', 'cls', 'this'):
                    continue
                if called_name in func_index:
                    for target_file, target_key in func_index[called_name]:
                        if target_key != caller_key:  # Skip self-recursion noise
                            calls.add(target_key)
            
            # Method calls (self.method, this.method)
            for m in method_call_pattern.finditer(body):
                called_name = m.group(1)
                if called_name in func_index:
                    # Prefer same-file match
                    for target_file, target_key in func_index[called_name]:
                        if target_file == f.path and target_key != caller_key:
                            calls.add(target_key)
                            break
                    else:
                        # Cross-file
                        for target_file, target_key in func_index[called_name]:
                            if target_key != caller_key:
                                calls.add(target_key)
            
            if calls:
                call_graph[caller_key] = sorted(calls)
    
    return call_graph


def build_manifest_from_session_files(session_files: Dict[str, str]) -> ZipManifest:
    """
    Build a lightweight ZipManifest from session files (for refresh after tool edits).
    Used by the in-loop signature/association refresh.
    """
    files = []
    signature_index = {}
    languages = {}
    total_size = 0
    
    for filepath, content in session_files.items():
        if not content:
            continue
        
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        language = detect_language(filepath)
        size = len(content.encode('utf-8', errors='replace'))
        total_size += size
        
        if language:
            languages[language] = languages.get(language, 0) + 1
        
        sigs = []
        if language:
            sigs = SignatureExtractor.extract(content, language)
        
        fi = FileInfo(
            path=filepath,
            filename=filename,
            extension=ext,
            size=size,
            language=language,
            is_binary=False,
            signatures=sigs,
            content=content,
        )
        files.append(fi)
        
        if sigs:
            from dataclasses import asdict
            signature_index[filepath] = [asdict(s) for s in sigs]
    
    return ZipManifest(
        total_files=len(files),
        total_size=total_size,
        files=files,
        signature_index=signature_index,
        file_tree={},  # Not needed for refresh
        languages=languages,
    )


def extract_markdown_gists(manifest: ZipManifest, max_per_file: int = 300) -> Dict[str, str]:
    """
    Extract gists from README and .md files (excluding agent/memory files).
    Returns dict of filepath -> gist string.
    """
    gists = {}
    
    for f in manifest.files:
        if not f.content or f.language != 'markdown':
            continue
        basename = f.filename.lower()
        # Skip agent memory files
        if basename.startswith('agent') and basename.endswith('.md'):
            continue
        if 'learnedsummary' in basename:
            continue
        
        lines = f.content.split('\n')
        # Get title (first heading)
        title = None
        gist_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not title and stripped.startswith('#'):
                title = re.sub(r'^#+\s*', '', stripped)
                continue
            # Get headings as outline
            if stripped.startswith('#'):
                heading = re.sub(r'^#+\s*', '', stripped)
                level = len(stripped) - len(stripped.lstrip('#'))
                gist_lines.append(f"{'  ' * (level - 1)}• {heading}")
            # Get first paragraph
            elif not gist_lines and stripped and not stripped.startswith('```'):
                gist_lines.append(stripped)
                if len(' '.join(gist_lines)) > max_per_file:
                    break
        
        if title or gist_lines:
            parts = []
            if title:
                parts.append(title)
            if gist_lines:
                parts.append('\n'.join(gist_lines[:10]))
            gists[f.path] = '\n'.join(parts)[:max_per_file]
    
    return gists


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
    
    # NC-0.8.0.12: Associations and flows
    assoc = extract_associations(manifest)
    
    has_assoc = any([assoc['entry_points'], assoc['imports'], assoc['class_hierarchy'],
                     assoc['services'], assoc['dockerfile_info']])
    if has_assoc:
        lines.append("=" * 60)
        lines.append("PROJECT ARCHITECTURE & FLOWS:")
        lines.append("=" * 60)
        
        # Entry points
        if assoc['entry_points']:
            lines.append("\nENTRY POINTS:")
            for ep in assoc['entry_points']:
                lines.append(f"  → {ep}")
        
        # Service topology (docker-compose)
        if assoc['services']:
            lines.append("\nSERVICES (docker-compose):")
            for svc_name, info in assoc['services'].items():
                parts = [f"  {svc_name}"]
                if 'image' in info:
                    parts.append(f"image={info['image'].split('→', 1)[-1]}")
                if 'build' in info:
                    parts.append(f"build={info['build'].split('→', 1)[-1]}")
                if 'port' in info:
                    parts.append(f"port={info['port'].split(':', 1)[-1]}")
                if 'depends_on' in info:
                    parts.append(f"depends_on={info['depends_on'].split('→', 1)[-1]}")
                lines.append(' | '.join(parts))
        
        # Dockerfile info
        if assoc['dockerfile_info']:
            lines.append("\nDOCKER BUILDS:")
            for d in assoc['dockerfile_info']:
                parts = [f"  {d['file']}"]
                if d['stages']:
                    parts.append(f"stages=[{', '.join(d['stages'])}]")
                if d['ports']:
                    parts.append(f"ports=[{', '.join(d['ports'])}]")
                if d['cmd']:
                    parts.append(f"cmd={d['cmd'][:60]}")
                lines.append(' | '.join(parts))
        
        # Import graph (local files only, limit to top importers)
        local_imports = {}
        for src, targets in assoc['imports'].items():
            local = [t for t in targets if t in {f.path for f in manifest.files}]
            if local:
                local_imports[src] = local
        
        if local_imports:
            lines.append("\nIMPORT GRAPH (local files):")
            # Sort by number of imports descending
            for src, targets in sorted(local_imports.items(), key=lambda x: -len(x[1]))[:20]:
                lines.append(f"  {src} → {', '.join(targets)}")
        
        # Class hierarchy
        if assoc['class_hierarchy']:
            lines.append("\nCLASS HIERARCHY:")
            for cls, parents in sorted(assoc['class_hierarchy'].items()):
                lines.append(f"  {cls.split(':')[-1]} extends {', '.join(parents)}")
        
        lines.append("")
    
    # NC-0.8.0.12: Call graph
    call_graph = extract_call_graph(manifest)
    if call_graph:
        lines.append("=" * 60)
        lines.append("CALL GRAPH (function → functions it calls):")
        lines.append("=" * 60)
        
        # Build reverse graph for "called by" info
        called_by = {}  # target -> [callers]
        for caller, callees in call_graph.items():
            for callee in callees:
                if callee not in called_by:
                    called_by[callee] = []
                called_by[callee].append(caller)
        
        # Output: group by file, show calls and callers
        by_file = {}
        all_keys = set(call_graph.keys()) | set(called_by.keys())
        for key in all_keys:
            file_path = key.split(':')[0]
            func_name = key.split(':')[1] if ':' in key else key
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append((func_name, key))
        
        for file_path in sorted(by_file.keys()):
            lines.append(f"\n  {file_path}:")
            for func_name, key in sorted(by_file[file_path]):
                parts = []
                if key in call_graph:
                    callees = [c.split(':')[-1] for c in call_graph[key][:5]]
                    parts.append(f"calls: {', '.join(callees)}")
                if key in called_by:
                    callers = [c.split(':')[-1] for c in called_by[key][:5]]
                    parts.append(f"called_by: {', '.join(callers)}")
                if parts:
                    lines.append(f"    {func_name} | {' | '.join(parts)}")
        
        lines.append("")
    
    # NC-0.8.0.12: Markdown/README gists
    gists = extract_markdown_gists(manifest)
    if gists:
        lines.append("=" * 60)
        lines.append("DOCUMENTATION OVERVIEW:")
        lines.append("=" * 60)
        for path, gist in sorted(gists.items()):
            lines.append(f"\n### {path}")
            lines.append(gist)
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
