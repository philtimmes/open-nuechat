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
# Simple patterns: verb + image word anywhere in the text
IMAGE_GEN_PATTERNS = [
    r"\bgenerate\b.*\bimage\b",
    r"\bmake\b.*\bimage\b",
    r"\bmake\b.*\bpicture\b",
    r"\bmake\b.*\bpic\b",
    r"\bgenerate\b.*\bpic\b",
    r"\bcreate\b.*\bimage\b",
    r"\bcreate\b.*\bpic\b",
    r"\bcreate\b.*\bpicture\b",
    # Also match reverse order (image/pic first, then verb)
    r"\bimage\b.*\bgenerate\b",
    r"\bimage\b.*\bmake\b",
    r"\bimage\b.*\bcreate\b",
    r"\bpicture\b.*\bmake\b",
    r"\bpicture\b.*\bcreate\b",
    r"\bpic\b.*\bmake\b",
    r"\bpic\b.*\bcreate\b",
    r"\bpic\b.*\bgenerate\b",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in IMAGE_GEN_PATTERNS]


async def confirm_image_request_with_llm(text: str, db: "AsyncSession") -> Tuple[bool, Optional[str], bool]:
    """
    Use the LLM to confirm if this is an image generation request.
    
    Uses admin-configurable prompt and expected response from system_settings:
    - image_classification_prompt: The system prompt for the classifier
    - image_classification_true_response: What response indicates YES (default: "YES")
    
    Returns:
        Tuple of (is_image_request, extracted_prompt, was_error)
        - was_error: True if there was an error (should fall back to regex)
    """
    from app.services.llm import LLMService
    from app.api.routes.admin import get_system_setting
    
    logger.info(f"[IMAGE_LLM] Starting LLM confirmation for: '{text[:60]}...'")
    
    try:
        # Get admin-configured prompt and expected response
        custom_prompt = await get_system_setting(db, "image_classification_prompt")
        true_response = await get_system_setting(db, "image_classification_true_response")
        
        # Default prompt if not configured
        if not custom_prompt:
            custom_prompt = """You are a classifier that determines if the user wants to generate/create an image.
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

User: "Make an image: 1280x720 A surreal landscape"
Answer: YES: A surreal landscape"""
        
        # Default true response
        if not true_response:
            true_response = "YES"
        
        true_response = true_response.strip().upper()
        
        # Get LLM service with admin panel settings
        llm_service = await LLMService.from_database(db)
        
        result = await llm_service.simple_completion(
            prompt=text,
            system_prompt=custom_prompt,
            max_tokens=200,
        )
        
        result = result.strip()
        logger.info(f"[IMAGE_LLM] LLM response: '{result[:100]}', checking for '{true_response}'")
        
        # Check for empty result (indicates LLM error)
        if not result:
            logger.warning("[IMAGE_LLM] LLM returned empty result")
            return False, None, True  # Error - should fall back
        
        # Check if response starts with the expected true response
        if result.upper().startswith(true_response):
            # Extract the prompt after the true response and colon
            prompt = text.strip()
            if ":" in result:
                # Try to extract prompt from "YES: <prompt>" format
                after_marker = result.split(":", 1)
                if len(after_marker) > 1 and after_marker[1].strip():
                    prompt = after_marker[1].strip()
            
            logger.info(f"[IMAGE_LLM] Confirmed as image request, prompt: '{prompt[:50]}...'")
            return True, prompt, False
        else:
            logger.info(f"[IMAGE_LLM] Not an image request (response didn't start with '{true_response}')")
            return False, None, False  # Genuine NO
            
    except Exception as e:
        logger.error(f"[IMAGE_LLM] Exception: {e}")
        return False, None, True  # Error - should fall back


def detect_image_request_regex(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if text is an image generation request using regex patterns.
    
    Returns:
        Tuple of (is_image_request, extracted_prompt)
    """
    if not text or len(text.strip()) < 5:
        return False, None
    
    text_lower = text.lower().strip()
    
    # Check positive patterns - if any match, it's potentially an image request
    # LLM confirmation (if enabled) handles false positives
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text_lower)
        if match:
            logger.info(f"[IMAGE_REGEX] Pattern matched: '{match.group()}' in text: '{text[:60]}...'")
            return True, text.strip()
    
    logger.debug(f"[IMAGE_REGEX] No pattern matched for: '{text[:60]}...'")
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
    Detect if text is an image generation request.
    
    Order:
    1. Regex pre-filter (must contain verb + image word like "make image", "create pic", etc.)
    2. If regex matches AND LLM confirmation enabled: LLM confirms
    3. If LLM says no, reject. If LLM errors, use regex result.
    
    Args:
        text: The user's message
        db: Database session (required if use_llm=True)
        use_llm: Whether to use LLM confirmation. If None, uses IMAGE_CONFIRM_WITH_LLM env var.
    
    Returns:
        Tuple of (is_image_request, extracted_prompt)
    """
    if use_llm is None:
        use_llm = IMAGE_CONFIRM_WITH_LLM
    
    # Step 1: Regex pre-filter - must contain verb + image word
    regex_match, regex_prompt = detect_image_request_regex(text)
    
    if not regex_match:
        # Regex didn't match - not an image request
        logger.info("[IMAGE_DETECT] Regex did not match, not an image request")
        return False, None
    
    logger.info("[IMAGE_DETECT] Regex matched")
    
    # Step 2: If LLM confirmation enabled, ask LLM to confirm
    if use_llm and db is not None:
        logger.info("[IMAGE_DETECT] Confirming with LLM...")
        llm_result, llm_prompt, was_error = await confirm_image_request_with_llm(text, db)
        
        if was_error:
            # LLM error - fall back to regex result (generate image)
            logger.warning("[IMAGE_DETECT] LLM error, using regex result")
            return True, extract_image_prompt(regex_prompt)
        
        if llm_result:
            logger.info("[IMAGE_DETECT] LLM confirmed")
            return True, llm_prompt if llm_prompt else extract_image_prompt(regex_prompt)
        else:
            logger.info("[IMAGE_DETECT] LLM rejected")
            return False, None
    
    # LLM disabled or no db - use regex result
    logger.info("[IMAGE_DETECT] LLM disabled, using regex result")
    return True, extract_image_prompt(regex_prompt)


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
