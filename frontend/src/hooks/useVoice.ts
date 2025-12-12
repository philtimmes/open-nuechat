import { useState, useRef, useCallback, useEffect } from 'react';
import { useVoiceStore } from '../stores/voiceStore';
import api from '../lib/api';

interface UseVoiceOptions {
  onTranscript?: (text: string) => void;
  onVoiceModeEnd?: () => void;
  silenceThreshold?: number; // dB threshold for silence detection
  silenceDuration?: number; // ms of silence before stopping
  minSpeechDuration?: number; // ms of speech required before considering it valid
}

export function useVoice(options: UseVoiceOptions = {}) {
  const { 
    ttsEnabled, 
    ttsMethod,
    selectedVoice, 
    selectedLocalVoice,
    localVoices,
    availableVoices,
    sttEnabled, 
    selectedLanguage,
    ttsAvailable,
    sttAvailable,
    localTtsAvailable,
    setTalkToMeMode,
    setTtsMethod,
    setSelectedVoice,
    setSelectedLocalVoice,
    fetchVoices,
    fetchLocalVoices,
    talkToMeMode
  } = useVoiceStore();
  
  const [isReading, setIsReading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);
  const [readingMessageId, setReadingMessageId] = useState<string | null>(null);
  const [isMuted, setIsMuted] = useState(false); // Mic muted during TTS
  const [currentParagraph, setCurrentParagraph] = useState<string>(''); // Current text being read
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioQueueRef = useRef<AudioBuffer[]>([]);
  const isPlayingRef = useRef(false);
  const ttsAbortControllerRef = useRef<AbortController | null>(null);
  const audioContextPrimedRef = useRef(false);
  const currentUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const isMutedRef = useRef(false); // Ref for consistent mute state
  const talkToMeModeRef = useRef(talkToMeMode); // Ref to avoid stale closure
  
  // Streaming paragraph TTS refs
  const streamingTextRef = useRef<string>(''); // Full accumulated text
  const lastReadIndexRef = useRef<number>(0);  // Index we've read up to
  const paragraphQueueRef = useRef<string[]>([]); // Paragraphs waiting to be read
  const isStreamingTTSRef = useRef(false); // Whether we're in streaming TTS mode
  const streamingReadInProgressRef = useRef(false); // Whether a paragraph is currently being read
  
  // Silence detection refs
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const speechStartRef = useRef<number | null>(null); // Track when speech started
  const hasSpokenRef = useRef(false);
  const silenceCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  
  // Refs to avoid circular dependencies
  const startListeningRef = useRef<() => void>(() => {});
  const stopVoiceModeRef = useRef<() => void>(() => {});
  const handleVoiceCommandRef = useRef<(transcript: string) => boolean>(() => false);
  const unmuteMicRef = useRef<() => void>(() => {});
  const isReadingRef = useRef(false);
  
  // Keep refs in sync with state
  useEffect(() => {
    isReadingRef.current = isReading;
  }, [isReading]);
  
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);
  
  useEffect(() => {
    talkToMeModeRef.current = talkToMeMode;
  }, [talkToMeMode]);

  // Fetch voices when voice mode is enabled (for voice commands)
  useEffect(() => {
    if (talkToMeMode) {
      if (availableVoices.length === 0) {
        console.log('Voice mode active - fetching natural voices...');
        fetchVoices();
      }
      if (localVoices.length === 0) {
        console.log('Voice mode active - fetching local voices...');
        fetchLocalVoices();
      }
    }
  }, [talkToMeMode, availableVoices.length, localVoices.length, fetchVoices, fetchLocalVoices]);

  // Ref to track if we need to start voice mode after initialization
  const pendingVoiceModeRef = useRef(false);

  // Check for pending voice mode after page reload (from voice change command)
  useEffect(() => {
    const pendingVoiceMode = sessionStorage.getItem('nexus-voice-mode-pending');
    if (pendingVoiceMode === 'true') {
      console.log('Voice mode pending after reload - will start when ready...');
      sessionStorage.removeItem('nexus-voice-mode-pending');
      pendingVoiceModeRef.current = true;
    }
  }, []); // Run once on mount

  // Effect to start voice mode when refs are ready and pending
  useEffect(() => {
    if (pendingVoiceModeRef.current && sttEnabled) {
      console.log('Starting voice mode after reload...');
      pendingVoiceModeRef.current = false;
      
      // Small delay to ensure audio context and refs are ready
      setTimeout(() => {
        setTalkToMeMode(true);
        // Another small delay then start listening
        setTimeout(() => {
          if (startListeningRef.current) {
            console.log('Starting listening after voice change reload');
            startListeningRef.current();
          }
        }, 500);
      }, 500);
    }
  }, [sttEnabled, setTalkToMeMode]);

  // Initialize AudioContext for streaming playback
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      // Create a gain node for consistent volume
      gainNodeRef.current = audioContextRef.current.createGain();
      gainNodeRef.current.gain.value = 1.0;
      gainNodeRef.current.connect(audioContextRef.current.destination);
    }
    return audioContextRef.current;
  }, []);

  // Prime the audio context with a silent buffer to ensure consistent volume
  const primeAudioContext = useCallback(async () => {
    if (audioContextPrimedRef.current) return;
    
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    
    // Create a very short silent buffer and play it
    const silentBuffer = ctx.createBuffer(1, ctx.sampleRate * 0.1, ctx.sampleRate);
    const source = ctx.createBufferSource();
    source.buffer = silentBuffer;
    source.connect(gainNodeRef.current || ctx.destination);
    source.start();
    
    audioContextPrimedRef.current = true;
    console.log('Audio context primed');
  }, [getAudioContext]);

  // Mute function that updates both state and ref immediately
  const muteMic = useCallback(() => {
    console.log('Muting mic for TTS');
    setIsMuted(true);
    isMutedRef.current = true;
  }, []);
  
  const unmuteMic = useCallback(() => {
    console.log('Unmuting mic after TTS');
    setIsMuted(false);
    isMutedRef.current = false;
  }, []);

  // Check if mic should be muted (TTS is playing)
  const isSpeaking = useCallback(() => {
    return isReadingRef.current || isMutedRef.current || isPlayingRef.current;
  }, []);

  // Stop reading but keep mic muted (for transitioning between TTS)
  const stopReadingKeepMuted = useCallback(() => {
    console.log('stopReadingKeepMuted called');
    
    // Abort any in-flight server TTS request
    if (ttsAbortControllerRef.current) {
      ttsAbortControllerRef.current.abort();
      ttsAbortControllerRef.current = null;
    }
    
    // Stop local TTS (Web Speech API)
    if ('speechSynthesis' in window && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
    currentUtteranceRef.current = null;
    
    // Stop HTML5 Audio
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      setCurrentAudio(null);
    }
    
    // Reset streaming TTS state
    isStreamingTTSRef.current = false;
    streamingTextRef.current = '';
    lastReadIndexRef.current = 0;
    paragraphQueueRef.current = [];
    streamingReadInProgressRef.current = false;
    
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setIsReading(false);
    setReadingMessageId(null);
    setCurrentParagraph('');
    // NOTE: intentionally NOT unmuting here
  }, [currentAudio]);

  // Stop any current audio playback (both server and local TTS)
  const stopReading = useCallback(async () => {
    console.log('stopReading called');
    
    // Abort any in-flight server TTS request
    if (ttsAbortControllerRef.current) {
      ttsAbortControllerRef.current.abort();
      ttsAbortControllerRef.current = null;
    }
    
    // Cancel any queued backend TTS jobs
    try {
      await api.post('/tts/cancel-all');
    } catch (e) {
      // Ignore errors - service may not be available
    }
    
    // Stop local TTS (Web Speech API)
    if ('speechSynthesis' in window && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
    currentUtteranceRef.current = null;
    
    // Stop HTML5 Audio
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      setCurrentAudio(null);
    }
    
    // Reset streaming TTS state
    isStreamingTTSRef.current = false;
    streamingTextRef.current = '';
    lastReadIndexRef.current = 0;
    paragraphQueueRef.current = [];
    streamingReadInProgressRef.current = false;
    
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setIsReading(false);
    setReadingMessageId(null);
    setCurrentParagraph('');
    unmuteMic();
  }, [currentAudio, unmuteMic]);

  // Play queued audio buffers sequentially
  const playNextBuffer = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;
    
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    
    // Prime the context on first playback
    await primeAudioContext();
    
    isPlayingRef.current = true;
    
    while (audioQueueRef.current.length > 0) {
      const buffer = audioQueueRef.current.shift()!;
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      
      // Connect through gain node for consistent volume
      if (gainNodeRef.current) {
        source.connect(gainNodeRef.current);
      } else {
        source.connect(ctx.destination);
      }
      
      await new Promise<void>((resolve) => {
        source.onended = () => resolve();
        source.start();
      });
    }
    
    isPlayingRef.current = false;
    
    // Check if we should start listening (all audio done)
    if (audioQueueRef.current.length === 0) {
      setIsReading(false);
      setReadingMessageId(null);
      unmuteMic(); // Unmute mic after TTS finishes
      
      // If in voice mode, start listening after response is read
      if (talkToMeMode && sttEnabled) {
        setTimeout(() => startListeningRef.current(), 300);
      }
    }
  }, [getAudioContext, primeAudioContext, talkToMeMode, sttEnabled, unmuteMic]);

  // Convert LaTeX math to spoken form
  const convertMathToSpeech = useCallback((math: string): string => {
    let result = math.trim();
    
    // Common LaTeX symbols to spoken words
    const mathReplacements: [RegExp, string][] = [
      // Greek letters
      [/\\alpha/g, ' alpha '],
      [/\\beta/g, ' beta '],
      [/\\gamma/g, ' gamma '],
      [/\\delta/g, ' delta '],
      [/\\epsilon/g, ' epsilon '],
      [/\\theta/g, ' theta '],
      [/\\lambda/g, ' lambda '],
      [/\\mu/g, ' mu '],
      [/\\pi/g, ' pi '],
      [/\\sigma/g, ' sigma '],
      [/\\omega/g, ' omega '],
      [/\\phi/g, ' phi '],
      [/\\psi/g, ' psi '],
      [/\\rho/g, ' rho '],
      [/\\tau/g, ' tau '],
      
      // Operations
      [/\\times/g, ' times '],
      [/\\cdot/g, ' times '],
      [/\\div/g, ' divided by '],
      [/\\pm/g, ' plus or minus '],
      [/\\mp/g, ' minus or plus '],
      [/\\neq/g, ' not equal to '],
      [/\\leq/g, ' less than or equal to '],
      [/\\geq/g, ' greater than or equal to '],
      [/\\approx/g, ' approximately equal to '],
      [/\\equiv/g, ' equivalent to '],
      [/\\propto/g, ' proportional to '],
      
      // Fractions
      [/\\frac\{([^}]+)\}\{([^}]+)\}/g, ' $1 over $2 '],
      [/\\dfrac\{([^}]+)\}\{([^}]+)\}/g, ' $1 over $2 '],
      
      // Powers and roots
      [/\\sqrt\{([^}]+)\}/g, ' square root of $1 '],
      [/\\sqrt\[(\d+)\]\{([^}]+)\}/g, ' $1th root of $2 '],
      [/\^2(?![0-9])/g, ' squared '],
      [/\^3(?![0-9])/g, ' cubed '],
      [/\^\{([^}]+)\}/g, ' to the power of $1 '],
      [/\^(\d+)/g, ' to the power of $1 '],
      [/_\{([^}]+)\}/g, ' sub $1 '],
      [/_(\w)/g, ' sub $1 '],
      
      // Sums, integrals, limits
      [/\\sum/g, ' sum of '],
      [/\\prod/g, ' product of '],
      [/\\int/g, ' integral of '],
      [/\\lim/g, ' limit of '],
      [/\\infty/g, ' infinity '],
      
      // Common functions
      [/\\sin/g, ' sine '],
      [/\\cos/g, ' cosine '],
      [/\\tan/g, ' tangent '],
      [/\\log/g, ' log '],
      [/\\ln/g, ' natural log '],
      [/\\exp/g, ' e to the '],
      
      // Brackets
      [/\\left\(/g, ' ('],
      [/\\right\)/g, ') '],
      [/\\left\[/g, ' ['],
      [/\\right\]/g, '] '],
      [/\\left\{/g, ' {'],
      [/\\right\}/g, '} '],
      [/\\left\|/g, ' absolute value of '],
      [/\\right\|/g, ' '],
      
      // Remove remaining LaTeX commands
      [/\\[a-zA-Z]+/g, ' '],
      [/[{}]/g, ''],
    ];
    
    for (const [pattern, replacement] of mathReplacements) {
      result = result.replace(pattern, replacement);
    }
    
    // Clean up
    result = result
      .replace(/\s+/g, ' ')
      .replace(/\(\s+/g, '(')
      .replace(/\s+\)/g, ')')
      .trim();
    
    return result || 'mathematical expression';
  }, []);

  // Clean text for TTS (strip markdown, convert math)
  const cleanTextForTTS = useCallback((text: string): string => {
    let result = text;
    
    // Remove code blocks (fenced with ``` or ~~~)
    result = result.replace(/```[\s\S]*?```/g, ' code block omitted ');
    result = result.replace(/~~~[\s\S]*?~~~/g, ' code block omitted ');
    
    // Remove inline code
    result = result.replace(/`[^`]+`/g, '');
    
    // Convert display math ($$...$$ or \[...\])
    result = result.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => convertMathToSpeech(math));
    result = result.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => convertMathToSpeech(math));
    
    // Convert inline math ($...$ or \(...\))
    result = result.replace(/\$([^$]+)\$/g, (_, math) => convertMathToSpeech(math));
    result = result.replace(/\\\(([^)]+)\\\)/g, (_, math) => convertMathToSpeech(math));
    
    // Convert links [text](url) to just text
    result = result.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    
    // Remove images ![alt](url)
    result = result.replace(/!\[[^\]]*\]\([^)]+\)/g, ' image ');
    
    // Remove HTML tags
    result = result.replace(/<[^>]+>/g, '');
    
    // Convert headers to just text (remove # symbols)
    result = result.replace(/^#{1,6}\s+(.+)$/gm, '$1');
    
    // Remove horizontal rules
    result = result.replace(/^[-*_]{3,}$/gm, '');
    
    // Remove blockquote markers
    result = result.replace(/^>\s*/gm, '');
    
    // Remove list markers but keep text
    result = result.replace(/^[\s]*[-*+]\s+/gm, '');
    result = result.replace(/^[\s]*\d+\.\s+/gm, '');
    
    // Remove bold/italic/strikethrough markers
    result = result.replace(/\*\*\*([^*]+)\*\*\*/g, '$1'); // bold+italic
    result = result.replace(/\*\*([^*]+)\*\*/g, '$1');     // bold
    result = result.replace(/\*([^*]+)\*/g, '$1');         // italic
    result = result.replace(/___([^_]+)___/g, '$1');       // bold+italic
    result = result.replace(/__([^_]+)__/g, '$1');         // bold
    result = result.replace(/_([^_]+)_/g, '$1');           // italic
    result = result.replace(/~~([^~]+)~~/g, '$1');         // strikethrough
    
    // Remove remaining markdown symbols that might have been missed
    result = result.replace(/[#*_~`]/g, '');
    
    // Clean up whitespace
    result = result.replace(/\n+/g, ' ');
    result = result.replace(/\s+/g, ' ');
    
    return result.trim();
  }, [convertMathToSpeech]);

  // Stream TTS audio - plays chunks as they arrive
  const readAloudStreaming = useCallback(async (text: string, messageId?: string) => {
    if (!ttsEnabled || !text) return;
    
    // Mute mic FIRST, then stop any existing TTS
    muteMic();
    stopReadingKeepMuted();
    
    // Create new abort controller for this request
    ttsAbortControllerRef.current = new AbortController();
    
    // Use enhanced text cleaning (strips markdown, converts math, etc.)
    const cleanText = cleanTextForTTS(text);
    
    if (!cleanText) {
      unmuteMic();
      return;
    }
    
    const startTime = Date.now();
    let firstChunkTime: number | null = null;
    let firstPlayTime: number | null = null;
    let chunkCount = 0;
    
    try {
      setIsReading(true);
      if (messageId) setReadingMessageId(messageId);
      
      const ctx = getAudioContext();
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }
      
      // Get auth token
      const authHeader = api.defaults.headers.common['Authorization'];
      
      console.log(`TTS starting for ${cleanText.length} chars...`);
      
      // Fetch streaming TTS
      const response = await fetch(`${api.defaults.baseURL}/tts/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(authHeader ? { 'Authorization': authHeader as string } : {})
        },
        body: JSON.stringify({
          text: cleanText.substring(0, 5000),
          voice: selectedVoice
        }),
        signal: ttsAbortControllerRef.current.signal
      });
      
      if (!response.ok) {
        throw new Error(`TTS streaming failed: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');
      
      let buffer = new Uint8Array(0);
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // Append new data to buffer
        const newBuffer = new Uint8Array(buffer.length + value.length);
        newBuffer.set(buffer);
        newBuffer.set(value, buffer.length);
        buffer = newBuffer;
        
        // Process complete chunks (4-byte length prefix + WAV data)
        while (buffer.length >= 4) {
          const chunkLength = new DataView(buffer.buffer, buffer.byteOffset).getUint32(0, false);
          
          if (buffer.length < 4 + chunkLength) break; // Not enough data yet
          
          // Extract WAV chunk
          const wavData = buffer.slice(4, 4 + chunkLength);
          buffer = buffer.slice(4 + chunkLength);
          
          chunkCount++;
          if (!firstChunkTime) {
            firstChunkTime = Date.now() - startTime;
            console.log(`TTS first chunk received at ${firstChunkTime}ms`);
          }
          
          // Decode and queue audio
          try {
            const arrayBuffer = wavData.buffer.slice(
              wavData.byteOffset, 
              wavData.byteOffset + wavData.byteLength
            );
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            
            if (!firstPlayTime && audioQueueRef.current.length === 0 && !isPlayingRef.current) {
              firstPlayTime = Date.now() - startTime;
              console.log(`TTS first audio will play at ${firstPlayTime}ms`);
            }
            
            audioQueueRef.current.push(audioBuffer);
            playNextBuffer();
          } catch (e) {
            console.warn('Failed to decode audio chunk:', e);
          }
        }
      }
      
      // Wait for all audio to finish playing
      while (audioQueueRef.current.length > 0 || isPlayingRef.current) {
        await new Promise(r => setTimeout(r, 100));
      }
      
      const totalTime = Date.now() - startTime;
      console.log(`TTS complete: ${chunkCount} chunks, first chunk ${firstChunkTime}ms, first play ${firstPlayTime}ms, total ${totalTime}ms`);
      
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('TTS aborted');
      } else {
        console.error('TTS streaming error:', error);
        // Fallback to non-streaming
        readAloudNonStreaming(text, messageId);
      }
    } finally {
      ttsAbortControllerRef.current = null;
    }
  }, [ttsEnabled, selectedVoice, getAudioContext, stopReadingKeepMuted, playNextBuffer, muteMic, unmuteMic, cleanTextForTTS]);

  // Fallback: Read text aloud using non-streaming TTS API
  const readAloudNonStreaming = useCallback(async (text: string, messageId?: string) => {
    if (!ttsEnabled || !text) return;
    
    // Mute mic FIRST, then stop any existing TTS
    muteMic();
    stopReadingKeepMuted();
    
    // Use enhanced text cleaning (strips markdown, converts math, etc.)
    const cleanText = cleanTextForTTS(text);
    
    if (!cleanText) {
      unmuteMic();
      return;
    }
    
    try {
      setIsReading(true);
      if (messageId) setReadingMessageId(messageId);
      
      const response = await api.post('/tts/generate', {
        text: cleanText.substring(0, 5000),
        voice: selectedVoice
      }, {
        responseType: 'blob'
      });
      
      const audioBlob = new Blob([response.data], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.onended = () => {
        setIsReading(false);
        setReadingMessageId(null);
        setCurrentAudio(null);
        unmuteMic(); // Unmute after reading
        URL.revokeObjectURL(audioUrl);
        
        if (talkToMeMode && sttEnabled) {
          setTimeout(() => startListeningRef.current(), 300);
        }
      };
      
      audio.onerror = () => {
        setIsReading(false);
        setReadingMessageId(null);
        setCurrentAudio(null);
        unmuteMic();
        URL.revokeObjectURL(audioUrl);
      };
      
      setCurrentAudio(audio);
      await audio.play();
      
    } catch (error) {
      console.error('TTS error:', error);
      setIsReading(false);
      setReadingMessageId(null);
      unmuteMic();
    }
  }, [ttsEnabled, selectedVoice, stopReadingKeepMuted, muteMic, unmuteMic, talkToMeMode, sttEnabled, cleanTextForTTS]);

  // Local TTS using Web Speech API (works on iOS/Safari, Chrome, Android, etc.)
  // Uses sentence chunking to work around Android Chrome's aggressive timeout
  const readAloudLocal = useCallback(async (text: string, messageId?: string) => {
    if (!ttsEnabled || !text || !('speechSynthesis' in window)) return;
    
    // Mute mic FIRST, then stop any existing TTS
    muteMic();
    stopReadingKeepMuted();
    
    // Use enhanced text cleaning (strips markdown, converts math, etc.)
    const cleanText = cleanTextForTTS(text);
    
    if (!cleanText) {
      unmuteMic();
      return;
    }
    
    try {
      setIsReading(true);
      if (messageId) setReadingMessageId(messageId);
      
      // Cancel any pending speech
      window.speechSynthesis.cancel();
      
      // Split text into sentences/chunks for Android compatibility
      // Android Chrome cancels speech after ~15 seconds, so we chunk it
      const chunks = cleanText
        .replace(/([.!?])\s+/g, '$1|SPLIT|')  // Split on sentence endings
        .split('|SPLIT|')
        .map(s => s.trim())
        .filter(s => s.length > 0);
      
      // If chunks are still too long, split further
      const maxChunkLength = 200; // Characters per chunk
      const finalChunks: string[] = [];
      for (const chunk of chunks) {
        if (chunk.length <= maxChunkLength) {
          finalChunks.push(chunk);
        } else {
          // Split long chunks on commas or spaces
          const subChunks = chunk.match(new RegExp(`.{1,${maxChunkLength}}(\\s|,|$)`, 'g')) || [chunk];
          finalChunks.push(...subChunks.map(s => s.trim()).filter(s => s.length > 0));
        }
      }
      
      console.log(`Local TTS: ${finalChunks.length} chunks to speak`);
      
      // Get the selected voice
      let selectedVoiceObj: SpeechSynthesisVoice | null = null;
      if (selectedLocalVoice) {
        const voiceURI = selectedLocalVoice.replace('local:', '');
        const voices = window.speechSynthesis.getVoices();
        selectedVoiceObj = voices.find(v => v.voiceURI === voiceURI) || null;
      }
      
      let chunkIndex = 0;
      let cancelled = false;
      
      const speakNextChunk = () => {
        if (cancelled || chunkIndex >= finalChunks.length) {
          // All done
          console.log('Local TTS finished all chunks');
          setIsReading(false);
          setReadingMessageId(null);
          currentUtteranceRef.current = null;
          unmuteMic();
          
          // If in voice mode, start listening after response is read
          if (talkToMeMode && sttEnabled) {
            setTimeout(() => startListeningRef.current(), 300);
          }
          return;
        }
        
        const chunk = finalChunks[chunkIndex];
        const utterance = new SpeechSynthesisUtterance(chunk);
        currentUtteranceRef.current = utterance;
        
        if (selectedVoiceObj) {
          utterance.voice = selectedVoiceObj;
          utterance.lang = selectedVoiceObj.lang;
        }
        
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        utterance.onend = () => {
          chunkIndex++;
          // Small delay between chunks for more natural speech
          setTimeout(speakNextChunk, 50);
        };
        
        utterance.onerror = (event) => {
          // 'interrupted' and 'canceled' are expected when stopping
          if (event.error !== 'interrupted' && event.error !== 'canceled') {
            console.error('Local TTS chunk error:', event.error);
          }
          cancelled = true;
          setIsReading(false);
          setReadingMessageId(null);
          currentUtteranceRef.current = null;
          unmuteMic();
        };
        
        // iOS Safari requires resume before speaking
        window.speechSynthesis.resume();
        window.speechSynthesis.speak(utterance);
      };
      
      // Store cancel function for stopReading
      const originalCancel = window.speechSynthesis.cancel.bind(window.speechSynthesis);
      
      // Start speaking
      speakNextChunk();
      
    } catch (error) {
      console.error('Local TTS error:', error);
      setIsReading(false);
      setReadingMessageId(null);
      unmuteMic();
    }
  }, [ttsEnabled, selectedLocalVoice, stopReadingKeepMuted, muteMic, unmuteMic, talkToMeMode, sttEnabled, cleanTextForTTS]);

  // Main read aloud function - picks between local and server TTS based on settings
  const readAloud = useCallback((text: string, messageId?: string) => {
    if (ttsMethod === 'local' && localTtsAvailable) {
      return readAloudLocal(text, messageId);
    } else if (ttsAvailable) {
      return readAloudStreaming(text, messageId);
    } else if (localTtsAvailable) {
      // Fallback to local if server not available
      return readAloudLocal(text, messageId);
    }
    // No TTS available
    console.warn('No TTS method available');
  }, [ttsMethod, localTtsAvailable, ttsAvailable, readAloudLocal, readAloudStreaming]);

  // Fuzzy match a voice name from user input
  const findMatchingVoice = useCallback((input: string, isNatural: boolean): { voice: string; name: string } | null => {
    const inputLower = input.toLowerCase().trim();
    const inputWords = inputLower.split(/\s+/).filter(w => w.length > 1);
    
    const voicesToSearch = isNatural ? availableVoices : localVoices;
    
    console.log(`findMatchingVoice: looking for "${input}" (words: ${JSON.stringify(inputWords)}) in ${isNatural ? 'natural' : 'local'} voices`);
    console.log(`Available ${isNatural ? 'natural' : 'local'} voices (${voicesToSearch.length}):`, voicesToSearch.map(v => v.name));
    
    if (voicesToSearch.length === 0) {
      console.warn(`No ${isNatural ? 'natural' : 'local'} voices available!`);
      return null;
    }
    
    if (isNatural) {
      // Search in Kokoro voices (availableVoices)
      let bestMatch: { voice: string; name: string; score: number } | null = null;
      
      for (const voice of availableVoices) {
        const nameLower = voice.name.toLowerCase();
        let score = 0;
        
        // Check for word matches
        for (const word of inputWords) {
          if (nameLower.includes(word)) {
            score += word.length;
            console.log(`  Match: "${word}" found in "${voice.name}", score now ${score}`);
          }
        }
        
        // Bonus for exact name match
        if (nameLower === inputLower) {
          score += 100;
        }
        
        if (score > 0 && (!bestMatch || score > bestMatch.score)) {
          bestMatch = { voice: voice.id, name: voice.name, score };
        }
      }
      
      console.log(`findMatchingVoice result:`, bestMatch);
      return bestMatch ? { voice: bestMatch.voice, name: bestMatch.name } : null;
    } else {
      // Search in local OS voices
      let bestMatch: { voice: string; name: string; score: number } | null = null;
      
      for (const voice of localVoices) {
        const nameLower = voice.name.toLowerCase();
        let score = 0;
        
        // Check for word matches
        for (const word of inputWords) {
          if (nameLower.includes(word)) {
            score += word.length;
            console.log(`  Match: "${word}" found in "${voice.name}", score now ${score}`);
          }
        }
        
        // Bonus for exact name match
        if (nameLower === inputLower) {
          score += 100;
        }
        
        if (score > 0 && (!bestMatch || score > bestMatch.score)) {
          bestMatch = { voice: voice.id, name: voice.name, score };
        }
      }
      
      console.log(`findMatchingVoice result:`, bestMatch);
      return bestMatch ? { voice: bestMatch.voice, name: bestMatch.name } : null;
    }
  }, [availableVoices, localVoices]);

  // Speak a confirmation message (mutes mic during playback)
  const speakConfirmation = useCallback(async (message: string) => {
    // Mute mic during confirmation to avoid feedback
    muteMic();
    
    // Use current TTS method
    if (ttsMethod === 'local' && localTtsAvailable && 'speechSynthesis' in window) {
      return new Promise<void>((resolve) => {
        const utterance = new SpeechSynthesisUtterance(message);
        
        if (selectedLocalVoice) {
          const voiceURI = selectedLocalVoice.replace('local:', '');
          const voices = window.speechSynthesis.getVoices();
          const voice = voices.find(v => v.voiceURI === voiceURI);
          if (voice) {
            utterance.voice = voice;
            utterance.lang = voice.lang;
          }
        }
        
        utterance.rate = 1.0;
        utterance.volume = 1.0;
        utterance.onend = () => resolve();
        utterance.onerror = () => resolve();
        
        window.speechSynthesis.resume();
        window.speechSynthesis.speak(utterance);
      });
    } else if (ttsAvailable) {
      // Use server TTS
      const authHeader = api.defaults.headers.common['Authorization'];
      
      try {
        const response = await fetch(`${api.defaults.baseURL}/tts/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(authHeader ? { 'Authorization': authHeader as string } : {})
          },
          body: JSON.stringify({
            text: message,
            voice: selectedVoice
          })
        });
        
        if (!response.ok) return;
        
        const ctx = getAudioContext();
        if (ctx.state === 'suspended') await ctx.resume();
        
        const reader = response.body?.getReader();
        if (!reader) return;
        
        let buffer = new Uint8Array(0);
        const audioBuffers: AudioBuffer[] = [];
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const newBuffer = new Uint8Array(buffer.length + value.length);
          newBuffer.set(buffer);
          newBuffer.set(value, buffer.length);
          buffer = newBuffer;
          
          while (buffer.length >= 4) {
            const chunkLength = new DataView(buffer.buffer, buffer.byteOffset).getUint32(0, false);
            if (buffer.length < 4 + chunkLength) break;
            
            const wavData = buffer.slice(4, 4 + chunkLength);
            buffer = buffer.slice(4 + chunkLength);
            
            try {
              const arrayBuffer = wavData.buffer.slice(wavData.byteOffset, wavData.byteOffset + wavData.byteLength);
              const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
              audioBuffers.push(audioBuffer);
            } catch (e) {
              // Skip bad chunk
            }
          }
        }
        
        // Play all audio
        for (const audioBuffer of audioBuffers) {
          await new Promise<void>((res) => {
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            if (gainNodeRef.current) {
              source.connect(gainNodeRef.current);
            } else {
              source.connect(ctx.destination);
            }
            source.onended = () => res();
            source.start();
          });
        }
      } catch (e) {
        console.error('Confirmation TTS error:', e);
      }
    }
  }, [ttsMethod, localTtsAvailable, ttsAvailable, selectedLocalVoice, selectedVoice, getAudioContext, muteMic]);

  // Check if transcript is a voice command and handle it
  // Returns true if it was a command (should not be sent to chat)
  const handleVoiceCommand = useCallback((transcript: string): boolean => {
    // Clean transcript: remove punctuation except commas initially, extra spaces, lowercase
    let clean = transcript
      .toLowerCase()
      .replace(/[.!?;:'"]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    
    console.log(`Checking voice command: "${clean}"`);
    
    // Keywords that indicate a voice change command
    const voiceChangeKeywords = ['change voice', 'switch voice', 'use voice', 'set voice', 'select voice'];
    
    // Check if this looks like a voice change command
    const isVoiceCommand = voiceChangeKeywords.some(keyword => clean.includes(keyword));
    
    if (!isVoiceCommand) {
      return false;
    }
    
    console.log('Voice change command detected');
    
    // Extract the voice name part (everything after the keyword pattern)
    let voicePart = '';
    for (const keyword of voiceChangeKeywords) {
      const idx = clean.indexOf(keyword);
      if (idx !== -1) {
        voicePart = clean.slice(idx + keyword.length).trim();
        break;
      }
    }
    
    // Handle comma-separated format: "change voice, OS, English US" or "change voice, natural, adam"
    // Split by commas and normalize
    const parts = voicePart.split(',').map(p => p.trim()).filter(p => p.length > 0);
    
    let voiceType: 'natural' | 'os' | null = null;
    let voiceName = '';
    
    if (parts.length >= 2) {
      // First part might be "to" or the voice type
      const firstPart = parts[0].replace(/^to\s+/, '').trim();
      
      if (/^(natural|kokoro)$/i.test(firstPart)) {
        voiceType = 'natural';
        voiceName = parts.slice(1).join(' ').trim();
      } else if (/^(os|local|system|device)$/i.test(firstPart)) {
        voiceType = 'os';
        voiceName = parts.slice(1).join(' ').trim();
      } else {
        // No type specified in first part, check second part
        const secondPart = parts[1].trim();
        if (/^(natural|kokoro)$/i.test(secondPart)) {
          voiceType = 'natural';
          voiceName = parts.slice(2).join(' ').trim() || firstPart;
        } else if (/^(os|local|system|device)$/i.test(secondPart)) {
          voiceType = 'os';
          voiceName = parts.slice(2).join(' ').trim() || firstPart;
        } else {
          // Assume format "change voice, voice name" (no type)
          voiceName = parts.join(' ').trim();
        }
      }
    } else {
      // Single part after keyword - parse normally
      voiceName = voicePart
        .replace(/,/g, ' ')
        .replace(/^to\s+/, '')
        .replace(/^the\s+/, '')
        .trim();
      
      // Check if it starts with type indicator
      if (/^(natural|kokoro)\s+/i.test(voiceName)) {
        voiceType = 'natural';
        voiceName = voiceName.replace(/^(natural|kokoro)\s+/i, '').trim();
      } else if (/^(os|local|system|device)\s+/i.test(voiceName)) {
        voiceType = 'os';
        voiceName = voiceName.replace(/^(os|local|system|device)\s+/i, '').trim();
      }
    }
    
    // Remove remaining filler words
    voiceName = voiceName
      .replace(/^to\s+/, '')
      .replace(/^the\s+/, '')
      .trim();
    
    if (!voiceName) {
      speakConfirmation('Please specify a voice name');
      return true;
    }
    
    console.log(`Looking for voice: "${voiceName}", type: ${voiceType || 'auto'}`);
    
    // Helper to apply voice change and reload
    const applyVoiceChangeAndReload = (method: 'natural' | 'local', voiceId: string, voiceName: string) => {
      console.log(`Voice command: Changing to ${method} voice "${voiceName}"`);
      
      // Set the voice in store (persisted)
      if (method === 'natural') {
        setTtsMethod('natural');
        setSelectedVoice(voiceId);
      } else {
        setTtsMethod('local');
        setSelectedLocalVoice(voiceId);
      }
      
      // Set flag to re-enter voice mode after reload
      sessionStorage.setItem('nexus-voice-mode-pending', 'true');
      
      // Speak confirmation then reload
      speakConfirmation('Voice changed').then(() => {
        // Small delay to ensure audio finishes
        setTimeout(() => {
          window.location.reload();
        }, 500);
      });
    };
    
    if (voiceType === 'natural') {
      // Natural voice requested
      const match = findMatchingVoice(voiceName, true);
      
      if (match) {
        applyVoiceChangeAndReload('natural', match.voice, match.name);
        return true;
      } else {
        speakConfirmation(`Could not find a natural voice matching ${voiceName}`);
        return true;
      }
    }
    
    if (voiceType === 'os') {
      // OS/local voice requested
      const match = findMatchingVoice(voiceName, false);
      
      if (match) {
        applyVoiceChangeAndReload('local', match.voice, match.name);
        return true;
      } else {
        speakConfirmation(`Could not find an OS voice matching ${voiceName}`);
        return true;
      }
    }
    
    // No type specified - try natural first, then local
    let match = findMatchingVoice(voiceName, true);
    if (match) {
      applyVoiceChangeAndReload('natural', match.voice, match.name);
      return true;
    }
    
    match = findMatchingVoice(voiceName, false);
    if (match) {
      applyVoiceChangeAndReload('local', match.voice, match.name);
      return true;
    }
    
    speakConfirmation(`Could not find a voice matching ${voiceName}`);
    return true;
  }, [findMatchingVoice, setTtsMethod, setSelectedVoice, setSelectedLocalVoice, speakConfirmation]);

  // Read a single paragraph (used by streaming TTS)
  const readParagraphAsync = useCallback((paragraph: string): Promise<void> => {
    return new Promise((resolve) => {
      if (!paragraph.trim()) {
        resolve();
        return;
      }

      const cleanText = cleanTextForTTS(paragraph);
      if (!cleanText) {
        resolve();
        return;
      }

      // Set current paragraph for UI display
      setCurrentParagraph(paragraph);
      console.log(`Reading paragraph: "${cleanText.substring(0, 50)}..."`);

      if (ttsMethod === 'local' && localTtsAvailable && 'speechSynthesis' in window) {
        // Local TTS - use Web Speech API
        const utterance = new SpeechSynthesisUtterance(cleanText);
        
        if (selectedLocalVoice) {
          const voiceURI = selectedLocalVoice.replace('local:', '');
          const voices = window.speechSynthesis.getVoices();
          const voice = voices.find(v => v.voiceURI === voiceURI);
          if (voice) {
            utterance.voice = voice;
            utterance.lang = voice.lang;
          }
        }
        
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        utterance.onend = () => resolve();
        utterance.onerror = () => resolve();
        
        window.speechSynthesis.resume();
        window.speechSynthesis.speak(utterance);
      } else if (ttsAvailable) {
        // Server TTS - stream and play chunks as they arrive
        const authHeader = api.defaults.headers.common['Authorization'];
        const startTime = Date.now();
        let firstChunkTime: number | null = null;
        let firstPlayTime: number | null = null;
        
        fetch(`${api.defaults.baseURL}/tts/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(authHeader ? { 'Authorization': authHeader as string } : {})
          },
          body: JSON.stringify({
            text: cleanText.substring(0, 2000), // Limit size for paragraphs
            voice: selectedVoice
          })
        })
        .then(async (response) => {
          if (!response.ok) {
            resolve();
            return;
          }
          
          const ctx = getAudioContext();
          if (ctx.state === 'suspended') await ctx.resume();
          
          const reader = response.body?.getReader();
          if (!reader) {
            resolve();
            return;
          }
          
          let buffer = new Uint8Array(0);
          const audioQueue: AudioBuffer[] = [];
          let isPlaying = false;
          let streamDone = false;
          
          // Function to play next chunk from queue
          const playNext = async () => {
            if (isPlaying) return;
            
            if (audioQueue.length === 0) {
              if (streamDone) {
                const totalTime = Date.now() - startTime;
                console.log(`TTS complete: first chunk ${firstChunkTime}ms, first play ${firstPlayTime}ms, total ${totalTime}ms`);
                resolve();
              }
              return;
            }
            
            isPlaying = true;
            const audioBuffer = audioQueue.shift()!;
            
            if (!firstPlayTime) {
              firstPlayTime = Date.now() - startTime;
              console.log(`TTS first audio playing at ${firstPlayTime}ms`);
            }
            
            await new Promise<void>((res) => {
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              if (gainNodeRef.current) {
                source.connect(gainNodeRef.current);
              } else {
                source.connect(ctx.destination);
              }
              source.onended = () => {
                isPlaying = false;
                res();
                // Immediately try to play next
                playNext();
              };
              source.start();
            });
          };
          
          // Read stream and decode chunks, playing as they arrive
          while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
              streamDone = true;
              // If nothing is playing and queue is empty, we're done
              if (!isPlaying && audioQueue.length === 0) {
                const totalTime = Date.now() - startTime;
                console.log(`TTS complete (no audio): total ${totalTime}ms`);
                resolve();
              }
              break;
            }
            
            // Append to buffer
            const newBuffer = new Uint8Array(buffer.length + value.length);
            newBuffer.set(buffer);
            newBuffer.set(value, buffer.length);
            buffer = newBuffer;
            
            // Try to extract and decode complete chunks
            while (buffer.length >= 4) {
              const chunkLength = new DataView(buffer.buffer, buffer.byteOffset).getUint32(0, false);
              if (buffer.length < 4 + chunkLength) break;
              
              const wavData = buffer.slice(4, 4 + chunkLength);
              buffer = buffer.slice(4 + chunkLength);
              
              try {
                if (!firstChunkTime) {
                  firstChunkTime = Date.now() - startTime;
                  console.log(`TTS first chunk received at ${firstChunkTime}ms`);
                }
                const arrayBuffer = wavData.buffer.slice(wavData.byteOffset, wavData.byteOffset + wavData.byteLength);
                const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
                audioQueue.push(audioBuffer);
                
                // Start playback if not already playing
                if (!isPlaying) {
                  playNext();
                }
              } catch (e) {
                // Skip bad chunk
                console.warn('Failed to decode audio chunk:', e);
              }
            }
          }
        })
        .catch((e) => {
          console.error('TTS stream error:', e);
          resolve();
        });
      } else {
        resolve();
      }
    });
  }, [ttsMethod, localTtsAvailable, ttsAvailable, selectedLocalVoice, selectedVoice, getAudioContext, cleanTextForTTS]);

  // For Natural voices, we need to queue more text after each paragraph
  // This is called after a paragraph finishes reading
  const queueNextNaturalParagraph = useCallback(() => {
    if (ttsMethod !== 'natural') return;
    if (!isStreamingTTSRef.current && streamingTextRef.current === '') return;
    
    // Get any remaining unread text
    const fullText = streamingTextRef.current;
    if (!fullText) return;
    
    const unreadText = fullText.slice(lastReadIndexRef.current);
    if (!unreadText.trim()) return;
    
    // Look for the next paragraph break
    const paragraphBreakRegex = /\n\n+/;
    const sentenceEndRegex = /[.!?]\s+(?=[A-Z])/;
    
    let breakIndex = -1;
    let breakLength = 0;
    
    const paragraphMatch = unreadText.match(paragraphBreakRegex);
    if (paragraphMatch && paragraphMatch.index !== undefined) {
      breakIndex = paragraphMatch.index;
      breakLength = paragraphMatch[0].length;
    }
    
    if (breakIndex === -1 && unreadText.length > 100) {
      const sentenceMatch = unreadText.match(sentenceEndRegex);
      if (sentenceMatch && sentenceMatch.index !== undefined && sentenceMatch.index > 50) {
        breakIndex = sentenceMatch.index + 1;
        breakLength = sentenceMatch[0].length - 1;
      }
    }
    
    if (breakIndex > 0) {
      const paragraph = unreadText.slice(0, breakIndex).trim();
      if (paragraph) {
        paragraphQueueRef.current.push(paragraph);
        lastReadIndexRef.current += breakIndex + breakLength;
        console.log(`Natural voice: queued next paragraph (${paragraph.length} chars)`);
      }
    } else if (!isStreamingTTSRef.current) {
      // Streaming is complete, queue any remaining text
      const remaining = unreadText.trim();
      if (remaining) {
        paragraphQueueRef.current.push(remaining);
        lastReadIndexRef.current = fullText.length;
        console.log(`Natural voice: queued final remaining text (${remaining.length} chars)`);
      }
    }
  }, [ttsMethod]);

  // Process the paragraph queue
  const processStreamingQueue = useCallback(async () => {
    if (streamingReadInProgressRef.current) return;
    if (paragraphQueueRef.current.length === 0) {
      // For Natural voices, check if there's more text to queue
      if (ttsMethod === 'natural') {
        queueNextNaturalParagraph();
        if (paragraphQueueRef.current.length > 0) {
          // Found more text, continue processing
          processStreamingQueue();
          return;
        }
      }
      
      // Check if streaming is done and we should start listening
      if (!isStreamingTTSRef.current && talkToMeMode && sttEnabled) {
        setIsReading(false);
        setCurrentParagraph('');
        unmuteMic();
        setTimeout(() => startListeningRef.current(), 300);
      }
      return;
    }
    
    streamingReadInProgressRef.current = true;
    
    // For Natural voices, only process one paragraph at a time
    const isNaturalVoice = ttsMethod === 'natural';
    
    if (isNaturalVoice) {
      // Process just the first paragraph
      const paragraph = paragraphQueueRef.current.shift()!;
      await readParagraphAsync(paragraph);
      
      streamingReadInProgressRef.current = false;
      
      // Queue next paragraph for Natural voice
      queueNextNaturalParagraph();
      
      // Continue if there's more
      if (paragraphQueueRef.current.length > 0) {
        processStreamingQueue();
      } else if (!isStreamingTTSRef.current) {
        // Streaming complete
        setIsReading(false);
        setCurrentParagraph('');
        unmuteMic();
        if (talkToMeMode && sttEnabled) {
          setTimeout(() => startListeningRef.current(), 300);
        }
      }
    } else {
      // For local TTS, process all queued paragraphs
      while (paragraphQueueRef.current.length > 0) {
        const paragraph = paragraphQueueRef.current.shift()!;
        await readParagraphAsync(paragraph);
      }
      
      streamingReadInProgressRef.current = false;
      
      // Check if more paragraphs arrived while reading
      if (paragraphQueueRef.current.length > 0) {
        processStreamingQueue();
      } else if (!isStreamingTTSRef.current) {
        // Streaming complete, no more paragraphs
        setIsReading(false);
        setCurrentParagraph('');
        unmuteMic();
        if (talkToMeMode && sttEnabled) {
          setTimeout(() => startListeningRef.current(), 300);
        }
      }
    }
  }, [readParagraphAsync, talkToMeMode, sttEnabled, unmuteMic, ttsMethod, queueNextNaturalParagraph]);

  // Handle streaming text chunk - call this as streaming content updates
  const handleStreamingText = useCallback((fullText: string, isComplete: boolean = false) => {
    if (!talkToMeMode || !ttsEnabled) return;
    
    // Initialize streaming TTS on first chunk (or if complete and never initialized)
    if (!isStreamingTTSRef.current && fullText.length > 0) {
      isStreamingTTSRef.current = true;
      streamingTextRef.current = '';
      lastReadIndexRef.current = 0;
      paragraphQueueRef.current = [];
      muteMic();
      setIsReading(true);
      console.log('Starting streaming TTS');
    }
    
    // If we still haven't initialized (empty text), skip
    if (!isStreamingTTSRef.current && !isComplete) return;
    
    streamingTextRef.current = fullText;
    
    // For Natural voices (Kokoro), only queue one paragraph at a time
    // to reduce latency - wait for completion before processing rest
    const isNaturalVoice = ttsMethod === 'natural';
    
    // If Natural voice and already processing, don't queue more until complete
    if (isNaturalVoice && streamingReadInProgressRef.current && !isComplete) {
      return;
    }
    
    // Find paragraph boundaries in unread text
    const unreadText = fullText.slice(lastReadIndexRef.current);
    
    // Look for paragraph breaks (double newline) or long sentences
    const paragraphBreakRegex = /\n\n+/;
    const sentenceEndRegex = /[.!?]\s+(?=[A-Z])/;
    
    let breakIndex = -1;
    let breakLength = 0;
    
    // Check for paragraph break first
    const paragraphMatch = unreadText.match(paragraphBreakRegex);
    if (paragraphMatch && paragraphMatch.index !== undefined) {
      breakIndex = paragraphMatch.index;
      breakLength = paragraphMatch[0].length;
    }
    
    // If no paragraph break, look for sentence end (but only if we have enough text)
    if (breakIndex === -1 && unreadText.length > 100) {
      const sentenceMatch = unreadText.match(sentenceEndRegex);
      if (sentenceMatch && sentenceMatch.index !== undefined && sentenceMatch.index > 50) {
        breakIndex = sentenceMatch.index + 1; // Include the punctuation
        breakLength = sentenceMatch[0].length - 1;
      }
    }
    
    // If we found a break point, queue the paragraph
    if (breakIndex > 0) {
      const paragraph = unreadText.slice(0, breakIndex).trim();
      if (paragraph) {
        // For Natural voices, only queue if not already processing
        if (isNaturalVoice && paragraphQueueRef.current.length > 0) {
          // Don't queue more for Natural - will be handled on completion
          return;
        }
        
        paragraphQueueRef.current.push(paragraph);
        lastReadIndexRef.current += breakIndex + breakLength;
        console.log(`Queued paragraph (${paragraph.length} chars), queue size: ${paragraphQueueRef.current.length}, isNatural: ${isNaturalVoice}`);
        processStreamingQueue();
      }
    }
    
    // If streaming is complete, queue any remaining text
    if (isComplete) {
      const remaining = fullText.slice(lastReadIndexRef.current).trim();
      if (remaining) {
        paragraphQueueRef.current.push(remaining);
        console.log(`Queued final paragraph (${remaining.length} chars)`);
      } else {
        console.log('No remaining text to queue');
      }
      isStreamingTTSRef.current = false;
      lastReadIndexRef.current = 0;
      streamingTextRef.current = '';
      
      // Ensure queue processing starts even if queue was empty before
      processStreamingQueue();
    }
  }, [talkToMeMode, ttsEnabled, ttsMethod, muteMic, processStreamingQueue]);

  // Reset streaming TTS state
  const resetStreamingTTS = useCallback(() => {
    isStreamingTTSRef.current = false;
    streamingTextRef.current = '';
    lastReadIndexRef.current = 0;
    paragraphQueueRef.current = [];
    streamingReadInProgressRef.current = false;
  }, []);

  // Check if audio level is silence
  const checkSilence = useCallback(() => {
    if (!analyserRef.current) return;
    
    const dataArray = new Uint8Array(analyserRef.current.fftSize);
    analyserRef.current.getByteTimeDomainData(dataArray);
    
    // Calculate RMS (root mean square) for volume level
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const normalized = (dataArray[i] - 128) / 128;
      sum += normalized * normalized;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    const db = 20 * Math.log10(rms + 0.0001);
    
    // Less sensitive threshold - only detect actual speech
    const threshold = options.silenceThreshold ?? -35; // dB threshold (was -45)
    const silenceDurationMs = options.silenceDuration ?? 1500; // 1.5 seconds of silence (was 2)
    const minSpeechDurationMs = options.minSpeechDuration ?? 500; // Require at least 500ms of speech
    
    if (db < threshold) {
      // Silence detected
      if (silenceStartRef.current === null) {
        silenceStartRef.current = Date.now();
      }
      
      // Check if we have valid speech (spoke for minimum duration) and then silence
      const speechDuration = speechStartRef.current ? (silenceStartRef.current - speechStartRef.current) : 0;
      const silenceDuration = Date.now() - silenceStartRef.current;
      
      if (hasSpokenRef.current && speechDuration >= minSpeechDurationMs && silenceDuration > silenceDurationMs) {
        // User spoke for enough time and then was silent - stop recording
        console.log(`Silence detected after ${speechDuration}ms of speech, stopping recording`);
        if (mediaRecorderRef.current?.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
        if (silenceCheckIntervalRef.current) {
          clearInterval(silenceCheckIntervalRef.current);
          silenceCheckIntervalRef.current = null;
        }
      }
    } else {
      // Sound detected above threshold
      if (!hasSpokenRef.current) {
        // First time detecting speech
        speechStartRef.current = Date.now();
      }
      silenceStartRef.current = null;
      hasSpokenRef.current = true;
    }
  }, [options.silenceThreshold, options.silenceDuration, options.minSpeechDuration]);

  // Start listening for voice input (non-streaming fallback)
  const startListening = useCallback(async () => {
    // Don't start listening if TTS is playing (mic is muted)
    const speaking = isReadingRef.current || isMutedRef.current || isPlayingRef.current;
    if (!sttEnabled || isListening || speaking) {
      console.log('Cannot start listening:', { 
        sttEnabled, 
        isListening, 
        isReading: isReadingRef.current, 
        isMuted: isMutedRef.current,
        isPlaying: isPlayingRef.current
      });
      return;
    }
    
    console.log('Starting to listen...');
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Set up audio analysis for silence detection
      const ctx = getAudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      
      // Reset silence detection state
      silenceStartRef.current = null;
      speechStartRef.current = null;
      hasSpokenRef.current = false;
      recordingStartTimeRef.current = Date.now();
      
      // Start checking for silence
      silenceCheckIntervalRef.current = setInterval(checkSilence, 100);
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        // Clear silence check interval
        if (silenceCheckIntervalRef.current) {
          clearInterval(silenceCheckIntervalRef.current);
          silenceCheckIntervalRef.current = null;
        }
        
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        audioChunksRef.current = [];
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        analyserRef.current = null;
        
        // Calculate actual speech duration
        const speechDuration = speechStartRef.current 
          ? (Date.now() - speechStartRef.current)
          : 0;
        const minSpeechDurationMs = options.minSpeechDuration ?? 500;
        
        // Only process if user actually spoke for minimum duration
        const hasValidSpeech = hasSpokenRef.current && speechDuration >= minSpeechDurationMs;
        
        console.log(`Recording stopped: hasSpoken=${hasSpokenRef.current}, speechDuration=${speechDuration}ms, valid=${hasValidSpeech}`);
        
        if (audioBlob.size > 0 && hasValidSpeech) {
          setIsProcessing(true);
          try {
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.webm');
            if (selectedLanguage) {
              formData.append('language', selectedLanguage);
            }
            formData.append('task', 'transcribe');
            
            const response = await api.post('/stt/transcribe', formData, {
              headers: { 'Content-Type': 'multipart/form-data' }
            });
            
            let transcript = response.data?.text?.trim() || '';
            
            // Filter out common STT hallucinations that occur with silence/noise
            const hallucinations = [
              "we'll be right back",
              "thanks for watching",
              "thank you for watching", 
              "please subscribe",
              "like and subscribe",
              "see you next time",
              "bye",
              "goodbye",
              "...",
              "you"
            ];
            
            const lowerTranscript = transcript.toLowerCase();
            const isHallucination = hallucinations.some(h => 
              lowerTranscript === h || 
              lowerTranscript.startsWith(h + '.') ||
              lowerTranscript.startsWith(h + '!')
            );
            
            if (isHallucination) {
              console.log(`Filtered STT hallucination: "${transcript}"`);
              transcript = '';
            }
            
            if (transcript) {
              console.log(`STT Transcript: "${transcript}", talkToMeMode: ${talkToMeModeRef.current}`);
              
              // Check for stop command
              if (transcript.toLowerCase().includes('stop')) {
                console.log('Stop command detected');
                stopVoiceModeRef.current();
                return;
              }
              
              // Check for voice commands (change voice, etc.)
              // Use ref to avoid stale closure
              if (talkToMeModeRef.current) {
                console.log('Checking for voice commands...');
                const wasCommand = handleVoiceCommandRef.current(transcript);
                console.log(`Voice command result: ${wasCommand}`);
                if (wasCommand) {
                  // It was a voice command, restart listening after confirmation plays
                  // Give extra time for TTS confirmation to complete
                  setTimeout(() => {
                    console.log('Voice command complete, restarting listening...');
                    // Make sure mic is unmuted before listening
                    unmuteMicRef.current();
                    // Small delay after unmute to ensure audio context is ready
                    setTimeout(() => {
                      if (talkToMeModeRef.current) {
                        startListeningRef.current();
                      }
                    }, 100);
                  }, 2000); // 2 seconds should be enough for "Voice changed" to play
                  return;
                }
              }
              
              if (options.onTranscript) {
                console.log('Sending transcript to chat');
                options.onTranscript(transcript);
              }
            } else if (talkToMeModeRef.current) {
              // No valid transcript but in voice mode - restart listening
              setTimeout(() => startListeningRef.current(), 300);
            }
          } catch (error) {
            console.error('STT error:', error);
            // In voice mode, keep listening even on error
            if (talkToMeModeRef.current) {
              setTimeout(() => startListeningRef.current(), 500);
            }
          } finally {
            setIsProcessing(false);
          }
        } else if (talkToMeModeRef.current) {
          // No valid speech detected, restart listening in voice mode
          console.log('No valid speech detected, restarting listening');
          setTimeout(() => startListeningRef.current(), 300);
        }
        
        setIsListening(false);
      };
      
      mediaRecorder.start();
      setIsListening(true);
      
      // Auto-stop after 30 seconds max
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
      }, 30000);
      
    } catch (error) {
      console.error('Microphone error:', error);
      setIsListening(false);
      
      // Clear interval on error
      if (silenceCheckIntervalRef.current) {
        clearInterval(silenceCheckIntervalRef.current);
        silenceCheckIntervalRef.current = null;
      }
    }
  }, [sttEnabled, isListening, selectedLanguage, options, getAudioContext, checkSilence, talkToMeMode, handleVoiceCommand]);

  // Stop listening
  const stopListening = useCallback(() => {
    // Clear silence check interval
    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current);
      silenceCheckIntervalRef.current = null;
    }
    
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'end' }));
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }
    
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    analyserRef.current = null;
    setIsListening(false);
  }, []);

  // Start voice mode
  const startVoiceMode = useCallback(() => {
    // Ensure voices are loaded for voice commands
    if (availableVoices.length === 0) {
      console.log('Fetching natural voices for voice mode...');
      fetchVoices();
    }
    if (localVoices.length === 0) {
      console.log('Fetching local voices for voice mode...');
      fetchLocalVoices();
    }
    
    setTalkToMeMode(true);
    startListeningRef.current();
  }, [setTalkToMeMode, availableVoices.length, localVoices.length, fetchVoices, fetchLocalVoices]);

  // Stop voice mode
  const stopVoiceMode = useCallback(() => {
    stopReading();
    stopListening();
    setTalkToMeMode(false);
    unmuteMic();
    if (options.onVoiceModeEnd) {
      options.onVoiceModeEnd();
    }
  }, [stopReading, stopListening, setTalkToMeMode, unmuteMic, options]);

  // Toggle voice mode
  const toggleVoiceMode = useCallback(() => {
    if (talkToMeMode) {
      stopVoiceMode();
    } else {
      startVoiceMode();
    }
  }, [talkToMeMode, stopVoiceMode, startVoiceMode]);

  // Update refs when functions change
  useEffect(() => {
    startListeningRef.current = startListening;
    stopVoiceModeRef.current = stopVoiceMode;
    unmuteMicRef.current = unmuteMic;
  }, [startListening, stopVoiceMode, unmuteMic]);

  // Keep handleVoiceCommandRef in sync
  useEffect(() => {
    handleVoiceCommandRef.current = handleVoiceCommand;
  }, [handleVoiceCommand]);

  // Handle new assistant response in voice mode
  const handleAssistantResponse = useCallback((content: string) => {
    if (talkToMeMode && ttsEnabled) {
      readAloudStreaming(content);
    }
  }, [talkToMeMode, ttsEnabled, readAloudStreaming]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (silenceCheckIntervalRef.current) {
        clearInterval(silenceCheckIntervalRef.current);
      }
      if (ttsAbortControllerRef.current) {
        ttsAbortControllerRef.current.abort();
      }
      if (currentAudio) {
        currentAudio.pause();
      }
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    // State
    isReading,
    isListening,
    isProcessing,
    isVoiceMode: talkToMeMode,
    readingMessageId,
    isMuted, // Exposed for UI indication
    currentParagraph, // Text currently being read (for UI display)
    
    // TTS
    readAloud,
    readAloudNonStreaming,
    readAloudLocal,
    stopReading,
    
    // Streaming TTS (for voice mode)
    handleStreamingText,
    resetStreamingTTS,
    
    // STT
    startListening,
    stopListening,
    
    // Voice mode
    toggleVoiceMode,
    startVoiceMode,
    stopVoiceMode,
    handleAssistantResponse,
    
    // Status
    ttsEnabled,
    ttsMethod,
    sttEnabled,
    ttsAvailable,
    sttAvailable,
    localTtsAvailable,
  };
}
