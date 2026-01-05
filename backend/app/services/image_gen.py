"""
Image Generation Service
Detects image generation requests and routes to image generation backend
"""
import logging
import os
import re
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Service configuration
IMAGE_GEN_SERVICE_URL = os.getenv("IMAGE_GEN_SERVICE_URL", "http://localhost:8034")
# Image generation can take 30-120+ seconds depending on size and GPU
IMAGE_GEN_TIMEOUT = float(os.getenv("IMAGE_GEN_TIMEOUT", "600"))  # 10 minutes default
# Whether to use LLM confirmation before generating images
IMAGE_CONFIRM_WITH_LLM = os.getenv("IMAGE_CONFIRM_WITH_LLM", "true").lower() == "true"

logger.info(f"Image gen service URL: {IMAGE_GEN_SERVICE_URL}, timeout: {IMAGE_GEN_TIMEOUT}s, LLM confirm: {IMAGE_CONFIRM_WITH_LLM}")

# Pattern detection for image generation requests (quick pre-filter)
# Includes full words and common short forms
# IMPORTANT: Patterns must be specific enough to avoid matching legal/technical text
IMAGE_GEN_PATTERNS = [
    # Direct creation requests - require image-related word within 20 chars
    r"\b(create|make|generate|design|produce|render|gen)\b.{0,20}\b(image|picture|photo|illustration|artwork|graphic|icon|logo|banner|poster|avatar|thumbnail|pic|img|gfx)\b",
    # Image word followed by "of/for/showing" - indicates describing what to create
    r"\b(image|picture|photo|illustration|artwork|graphic|icon|logo|banner|poster|avatar|thumbnail|pic|img|gfx)\b.{0,20}\b(of|for|showing|depicting|with)\b",
    # Art verbs that strongly imply visual creation - require "me" or direct object pattern
    r"\b(paint|sketch|draw)\s+me\s+(a\s+|an\s+)?",  # "paint me a sunset"
    r"\b(paint|sketch|draw)\s+(a|an)\s+\w+\s+(of|for|with)\b",  # "draw a picture of"
    # Logo and design requests - specific context required
    r"\b(logo|icon|badge|emblem|logotype)\s+(for|design|idea)\b",
    r"\b(design|create|make)\s+(a\s+|an\s+|my\s+)?(logo|icon|badge|emblem)\b",
    # Portrait and character requests
    r"\b(portrait|headshot|character\s+art|avatar|pfp|profile\s*pic)\s+(of|for)\b",
    # Style-based requests - art style references
    r"\b(in the style of|like a|as a)\b.{0,30}\b(painting|drawing|photograph|illustration)\b",
    # Explicit image generation requests
    r"\bgenerate\s+(a\s+|an\s+)?(image|picture|photo|pic|img)\b",
    r"\bgen\s+(a\s+|an\s+)?(image|picture|photo|pic|img)\b",
    r"\bcan\s+you\s+(create|make|draw|generate).{0,15}(image|picture|photo|logo|pic|img)\b",
    r"\bi\s+(want|need|would\s+like)\s+(a\s+|an\s+)?(image|picture|photo|logo|illustration|pic|img)\b",
    # Short form requests - very specific patterns
    r"\b(make|create|gen)\s+me\s+(a\s+)?pic\b",
    r"\b(make|create|gen)\s+(a\s+)?pic\s+of\b",
    r"\bpic\s+of\s+(a\s+|an\s+|the\s+|my\s+)?\w+",  # "pic of a cat"
    r"\bimg\s+of\s+(a\s+|an\s+|the\s+|my\s+)?\w+",  # "img of a sunset"
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in IMAGE_GEN_PATTERNS]

# Negative patterns - these indicate NOT an image generation request
NEGATIVE_PATTERNS = [
    r"\b(analyze|describe|explain|what is|tell me about|look at|what's in)\b.{0,20}\b(this|the|my)\b.{0,10}\b(image|picture|photo|pic|img)\b",
    r"\b(upload|attach|send|share)\b.{0,20}\b(image|picture|photo|pic|img)\b",
    r"\b(image|picture|photo|pic|img)\b.{0,10}\b(you (sent|shared|uploaded))\b",
    r"\b(this|the)\s+(image|picture|photo|pic|img)\b",
]
COMPILED_NEGATIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]


async def confirm_image_request_with_llm(text: str, db: "AsyncSession") -> Tuple[bool, Optional[str]]:
    """
    Use the LLM to confirm if this is an image generation request.
    
    Uses LLMService.simple_completion which:
    - Does NOT track token usage against the user's quota
    - Uses admin panel LLM settings (not .env)
    
    Returns:
        Tuple of (is_image_request, extracted_prompt)
    """
    from app.services.llm import LLMService
    
    try:
        # Get LLM service with admin panel settings
        llm_service = await LLMService.from_database(db)
        
        # Use a fast, focused prompt
        system_prompt = """You are a classifier that determines if the user wants to generate/create an image.
Answer with ONLY "YES" or "NO" followed by the image prompt if YES.

Format:
- If user wants an image generated: YES: <the image description to generate>
- If user does NOT want an image generated: NO

Examples:
User: "Create an image of a sunset over mountains"
Answer: YES: a sunset over mountains

User: "What's in this picture?"
Answer: NO

User: "Can you make me a logo for my coffee shop?"
Answer: YES: a logo for a coffee shop

User: "Tell me about image processing"
Answer: NO

User: "Draw a cute cat wearing a hat"
Answer: YES: a cute cat wearing a hat

User: "How do I upload an image?"
Answer: NO

User: "gen me a pic of a dragon"
Answer: YES: a dragon

User: "make a pfp for my discord"
Answer: YES: a profile picture for discord

User: "make a picture of a mouse on a unicycle"
Answer: YES: a mouse on a unicycle"""

        result = await llm_service.simple_completion(
            prompt=text,
            system_prompt=system_prompt,
            max_tokens=100,
        )
        
        result = result.strip()
        logger.info(f"LLM image classification result: {result[:100]}")
        
        if result.upper().startswith("YES"):
            # Extract the prompt after "YES:"
            if ":" in result:
                prompt = result.split(":", 1)[1].strip()
            else:
                prompt = text.strip()
            logger.info(f"LLM confirmed image request, prompt: {prompt[:50]}...")
            return True, prompt
        else:
            logger.info("LLM determined this is NOT an image generation request")
            return False, None
            
    except Exception as e:
        logger.error(f"LLM image confirmation failed: {e}")
        # SAFE DEFAULT: If we can't confirm with LLM, don't generate image
        # This prevents false positives from regex triggering on non-image text
        logger.warning("Falling back to SAFE DEFAULT: not generating image")
        return False, None


def detect_image_request_regex(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if text is an image generation request using regex patterns.
    
    Returns:
        Tuple of (is_image_request, extracted_prompt)
    """
    if not text or len(text.strip()) < 5:
        return False, None
    
    text_lower = text.lower().strip()
    
    # Check negative patterns first (describing/analyzing existing images)
    for pattern in COMPILED_NEGATIVE_PATTERNS:
        if pattern.search(text_lower):
            logger.debug(f"Negative pattern matched, not an image request: {text[:50]}")
            return False, None
    
    # Check positive patterns
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text_lower)
        if match:
            logger.info(f"Image generation pattern matched: {pattern.pattern}")
            return True, text.strip()
    
    return False, None


def detect_image_request(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if text is an image generation request (synchronous, regex-only).
    
    This is the quick pre-filter. For LLM confirmation, use detect_image_request_async.
    
    Returns:
        Tuple of (is_image_request, extracted_prompt)
        - is_image_request: True if this appears to be an image generation request
        - extracted_prompt: The cleaned prompt for image generation, or None
    """
    return detect_image_request_regex(text)


async def detect_image_request_async(text: str, db: "AsyncSession" = None, use_llm: bool = None) -> Tuple[bool, Optional[str]]:
    """
    Detect if text is an image generation request, optionally using LLM confirmation.
    
    Args:
        text: The user's message
        db: Database session (required if use_llm=True)
        use_llm: Whether to use LLM confirmation. If None, uses IMAGE_CONFIRM_WITH_LLM env var.
    
    Returns:
        Tuple of (is_image_request, extracted_prompt)
    """
    if use_llm is None:
        use_llm = IMAGE_CONFIRM_WITH_LLM
    
    # First do quick regex pre-filter
    regex_match, regex_prompt = detect_image_request_regex(text)
    
    if not regex_match:
        # Regex didn't match, definitely not an image request
        return False, None
    
    if not use_llm:
        # LLM confirmation disabled, use regex result
        logger.info("Image request detected via regex (LLM confirmation disabled)")
        return regex_match, regex_prompt
    
    if db is None:
        logger.warning("LLM confirmation requested but no db session provided, falling back to regex")
        return regex_match, regex_prompt
    
    # Regex matched, now confirm with LLM
    logger.info("Regex matched image pattern, confirming with LLM...")
    return await confirm_image_request_with_llm(text, db)


def extract_image_prompt(text: str) -> str:
    """
    Extract and clean the image generation prompt from user text.
    Removes common prefixes like "create an image of", "make a picture showing", etc.
    """
    # Common prefixes to remove (including size specifications)
    prefixes = [
        r"^(please\s+)?(can you\s+)?(create|make|generate|draw|design|produce|render)\s+(me\s+)?(an?\s+)?(\d+\s*x\s*\d+\s+)?(image|picture|photo|illustration|artwork|graphic|logo|icon)\s*(of|showing|depicting|with|for|:)?\s*",
        r"^(please\s+)?(i\s+)?(want|need|would like)\s+(an?\s+)?(\d+\s*x\s*\d+\s+)?(image|picture|photo|illustration|logo|icon)\s*(of|showing|depicting|with|for|:)?\s*",
        r"^(please\s+)?(paint|sketch|illustrate|visualize)\s+(me\s+)?(an?\s+)?",
    ]
    
    cleaned = text.strip()
    for prefix_pattern in prefixes:
        cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE).strip()
    
    return cleaned if cleaned else text.strip()


def extract_size_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract image size from text like "Create a 1280x720 image" or "generate 512x512 picture".
    
    Returns:
        Tuple of (width, height) or (None, None) if no size found
    """
    # Pattern to match WIDTHxHEIGHT or WIDTH×HEIGHT
    size_pattern = r'\b(\d{3,4})\s*[x×]\s*(\d{3,4})\b'
    
    match = re.search(size_pattern, text, re.IGNORECASE)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        
        # Validate reasonable size range
        if 256 <= width <= 2048 and 256 <= height <= 2048:
            logger.info(f"Extracted size from text: {width}x{height}")
            return width, height
    
    return None, None


# Common aspect ratio sizes for quick reference
ASPECT_RATIO_SIZES = {
    '1:1': (1024, 1024),
    '16:9': (1280, 720),
    '9:16': (720, 1280),
    '4:3': (1024, 768),
    '3:4': (768, 1024),
    '3:2': (1200, 800),
    '2:3': (800, 1200),
    '21:9': (1344, 576),
}


def extract_aspect_ratio_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract size based on aspect ratio mention like "16:9 aspect" or "in 4:3".
    
    Returns:
        Tuple of (width, height) or (None, None) if no aspect ratio found
    """
    for ratio, (width, height) in ASPECT_RATIO_SIZES.items():
        # Pattern like "16:9", "16:9 aspect", "in 16:9"
        pattern = rf'\b{ratio.replace(":", r"[:\s]*")}\b'
        if re.search(pattern, text, re.IGNORECASE):
            logger.info(f"Extracted aspect ratio {ratio} -> {width}x{height}")
            return width, height
    
    return None, None


class ImageGenServiceClient:
    """Client for image generation backend service"""
    
    def __init__(self, base_url: str = IMAGE_GEN_SERVICE_URL):
        self.base_url = base_url.rstrip("/")
        self._client = None
    
    @property
    def client(self):
        """Lazy-init HTTP client with long timeout for image generation"""
        if self._client is None:
            import httpx
            # Image generation can take 30-120+ seconds - use very long timeouts
            timeout = httpx.Timeout(
                connect=30.0,   # 30s to connect
                read=600.0,     # 10 min to read response (generation takes time!)
                write=60.0,     # 60s to send request
                pool=30.0,      # 30s to acquire connection
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
            )
            logger.info(f"Created httpx client: base_url={self.base_url}, read_timeout=600s")
        return self._client
    
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check image generation service health"""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Image gen service health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def is_available(self) -> bool:
        """Check if image generation service is available"""
        health = await self.health_check()
        return health.get("status") == "healthy"
    
    async def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: The image description/prompt
            width: Image width (default 1024)
            height: Image height (default 1024)
            seed: Random seed (None for random)
            **kwargs: Additional parameters
        
        Returns:
            Dict with image data and metadata
        """
        try:
            # Clean the prompt
            clean_prompt = extract_image_prompt(prompt)
            
            payload = {
                "prompt": clean_prompt,
                "original_prompt": prompt,
                "width": width,
                "height": height,
            }
            if seed is not None:
                payload["seed"] = seed
            
            # Add any extra params
            payload.update(kwargs)
            
            logger.info(f"Sending image generation request: {clean_prompt[:100]}...")
            logger.info(f"Payload: width={width}, height={height}, seed={seed}")
            
            response = await self.client.post("/generate", json=payload)
            logger.info(f"Response status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Response keys: {list(result.keys())}")
            
            if result.get("success"):
                base64_len = len(result.get("image_base64", "")) if result.get("image_base64") else 0
                logger.info(f"Image generated successfully: job_id={result.get('job_id')}, seed={result.get('seed')}, base64_len={base64_len}")
            else:
                logger.warning(f"Image generation returned error: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            # Reset client on error so it gets recreated with fresh settings
            if "Timeout" in str(type(e).__name__) or "timeout" in str(e).lower():
                logger.warning("Timeout detected, resetting HTTP client")
                await self.close()
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }


# Singleton client instance
_client: Optional[ImageGenServiceClient] = None


def get_image_gen_client() -> ImageGenServiceClient:
    """Get or create the image generation client singleton"""
    global _client
    if _client is None:
        _client = ImageGenServiceClient()
        logger.info("Created new ImageGenServiceClient singleton")
    return _client


async def reset_image_gen_client():
    """Reset the image generation client (useful after timeout errors)"""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("Reset ImageGenServiceClient singleton")


async def handle_image_request(
    prompt: str,
    user_id: str,
    chat_id: str,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Handle an image generation request.
    
    This is the main entry point for processing detected image requests.
    
    Args:
        prompt: The user's message (will be cleaned)
        user_id: The requesting user's ID
        chat_id: The chat context
        width: Image width (default 1024, can be overridden by text)
        height: Image height (default 1024, can be overridden by text)
        seed: Random seed (None for random)
        **kwargs: Additional parameters
    
    Returns:
        Dict with generation result
    """
    client = get_image_gen_client()
    
    # Try to extract size from the prompt text
    extracted_width, extracted_height = extract_size_from_text(prompt)
    if extracted_width and extracted_height:
        width = extracted_width
        height = extracted_height
        logger.info(f"Using extracted size: {width}x{height}")
    else:
        # Try aspect ratio
        aspect_width, aspect_height = extract_aspect_ratio_from_text(prompt)
        if aspect_width and aspect_height:
            width = aspect_width
            height = aspect_height
            logger.info(f"Using aspect ratio size: {width}x{height}")
    
    # Check if service is available
    logger.info(f"Checking image gen service availability...")
    if not await client.is_available():
        logger.warning("Image generation service not available")
        return {
            "success": False,
            "available": False,
            "message": "Image generation service is not currently available. Please try again later.",
            "prompt": prompt
        }
    
    # Generate the image
    logger.info(f"Calling image generation service: prompt={prompt[:50]}..., size={width}x{height}")
    result = await client.generate_image(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        user_id=user_id,
        chat_id=chat_id,
        **kwargs
    )
    
    # Add availability flag
    result["available"] = True
    
    if result.get("success"):
        base64_len = len(result.get("image_base64", ""))
        logger.info(f"Image generation succeeded, base64 length={base64_len}")
    else:
        logger.error(f"Image generation failed: {result.get('error')}")
    
    return result
