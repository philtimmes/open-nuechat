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

logger.info(f"Image gen service URL: {IMAGE_GEN_SERVICE_URL}, timeout: {IMAGE_GEN_TIMEOUT}s")

# Pattern detection for image generation requests
IMAGE_GEN_PATTERNS = [
    # Direct creation requests
    r"\b(create|make|generate|draw|design|produce|render)\b.{0,20}\b(image|picture|photo|illustration|artwork|graphic|visual|icon|logo|banner|poster|avatar|thumbnail)\b",
    r"\b(image|picture|photo|illustration|artwork|graphic|visual|icon|logo|banner|poster|avatar|thumbnail)\b.{0,20}\b(of|for|showing|depicting|with)\b",
    # Specific art requests
    r"\b(paint|sketch|illustrate|visualize)\b.{0,30}\b",
    # Logo and design requests
    r"\b(logo|icon|badge|emblem)\b.{0,20}\b(for|design|create)\b",
    r"\b(design|create)\b.{0,20}\b(logo|icon|badge|emblem)\b",
    # Portrait and character requests
    r"\b(portrait|headshot|character|avatar)\b.{0,20}\b(of|for)\b",
    # Style-based requests
    r"\b(in the style of|like a|as a)\b.{0,30}\b(painting|drawing|photograph|illustration)\b",
    # Explicit image requests
    r"\bgenerate\s+an?\s+(image|picture|photo)\b",
    r"\bcan\s+you\s+(create|make|draw|generate).{0,20}(image|picture|photo|logo)\b",
    r"\bi\s+(want|need|would like).{0,20}(image|picture|photo|logo|illustration)\b",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in IMAGE_GEN_PATTERNS]

# Negative patterns - these indicate NOT an image generation request
NEGATIVE_PATTERNS = [
    r"\b(analyze|describe|explain|what is|tell me about|look at)\b.{0,20}\b(this|the|my)\b.{0,10}\b(image|picture|photo)\b",
    r"\b(upload|attach|send|share)\b.{0,20}\b(image|picture|photo)\b",
    r"\b(image|picture|photo)\b.{0,10}\b(you (sent|shared|uploaded))\b",
]
COMPILED_NEGATIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]


def detect_image_request(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if text is an image generation request.
    
    Returns:
        Tuple of (is_image_request, extracted_prompt)
        - is_image_request: True if this appears to be an image generation request
        - extracted_prompt: The cleaned prompt for image generation, or None
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
            # Extract the prompt - use the full text as the prompt
            # The image generation service can further process it
            return True, text.strip()
    
    return False, None


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
