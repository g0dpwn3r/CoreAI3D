'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { useGlobeBridge, GlobeCommands } from '@/lib/gep-bridge';
import { cn } from '@/lib/utils';
import { Maximize, Minimize, Mic, Globe, AlertCircle, RefreshCw, Settings } from 'lucide-react';
import { PermissionManager, useMicrophonePermission, type PermissionStatus } from '@/lib/permissions';

interface GlobeFrameProps {
  className?: string;
  url?: string;
  onEvent?: (event: any) => void;
}

export function GlobeFrame({ className, url, onEvent }: GlobeFrameProps) {
  const globeUrl =
    url ||
    (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_GLOBE_URL
      ? process.env.NEXT_PUBLIC_GLOBE_URL
      : 'http://localhost:4173?embed=1');
  const [fullscreen, setFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const bridge = useGlobeBridge({
    origin: new URL(globeUrl).origin,
    onEvent: (message) => {
      onEvent?.(message);
    },
  });

  const toggleFullscreen = () => {
    const el = containerRef.current;
    if (!el) return;
    if (!fullscreen) {
      void el.requestFullscreen?.().catch(() => {});
    } else {
      void document.exitFullscreen?.().catch(() => {});
    }
  };

  useEffect(() => {
    const onChange = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative bg-black rounded-xl border border-core-border overflow-hidden aspect-video',
        fullscreen
          ? 'fixed inset-0 z-50 w-screen h-screen bg-black'
          : 'aspect-video',
        className,
      )}
    >
      <iframe
        ref={bridge.iframeRef}
        src={`${globeUrl}&bridge_origin=${encodeURIComponent(
          typeof location !== 'undefined' ? location.origin : '',
        )}`}
        title="God's Eye View"
        className="w-full h-full border-0 bg-black"
        allow="fullscreen; autoplay; microphone; camera"
        loading="eager"
      />
      {!bridge.ready && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 text-text-muted">
          <Globe className="w-12 h-12 mb-2 animate-pulse" />
          <div className="text-sm font-medium">Loading God's Eye View…</div>
          <div className="text-xs text-text-muted mt-2">
            {bridge.error ? bridge.error : 'Ensure GEV is running on port 4173'}
          </div>
        </div>
      )}
      <div className="absolute top-2 right-2 flex gap-1 z-20">
        <button
          onClick={toggleFullscreen}
          className="p-2 rounded bg-black/60 text-white hover:bg-black/80 transition-colors"
          aria-label={fullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
        >
          {fullscreen ? <Minimize className="w-5 h-5" /> : <Maximize className="w-5 h-5" />}
        </button>
      </div>
    </div>
  );
}

interface GlobeControlProps {
  send: (message: any) => void;
  sendAndAwait: (message: any, timeoutMs?: number) => Promise<any>;
  ready: boolean;
}

export function GlobeControl({ send, sendAndAwait, ready }: GlobeControlProps) {
  const [last, setLast] = useState<string>('');

  const invoke = async (name: keyof typeof GlobeCommands, args: any) => {
    if (!ready) return;
    try {
      const cmdFn = GlobeCommands[name] as (a: any) => any;
      const reply = await sendAndAwait(cmdFn(args), 5000);
      setLast(
        `ok: ${String((reply as any)?.payload?.ok ?? (reply as any)?.ok)}`,
      );
    } catch (err) {
      setLast('error: ' + (err as Error)?.message);
    }
  };

  return (
    <div className="bg-core-card border border-core-border rounded-xl p-4 space-y-3">
      <h2 className="text-sm font-bold text-text-primary flex items-center gap-2">
        <Globe className="w-4 h-4" /> Globe Control
      </h2>
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => invoke('flyTo', { locationId: 'tokyo', viewMode: 'close' })}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Fly to Tokyo
        </button>
        <button
          onClick={() => invoke('enterCockpit', { targetLayer: 'flights' })}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Enter Cockpit
        </button>
        <button
          onClick={() => invoke('exitCockpit', {})}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Exit Cockpit
        </button>
        <button
          onClick={() => invoke('applySensorStyle', { style: 'thermal' })}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Thermal
        </button>
        <button
          onClick={() => invoke('setLayer', { layerId: 'flights', enabled: true })}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Layer: Flights On
        </button>
        <button
          onClick={() => invoke('setLayer', { layerId: 'flights', enabled: false })}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Layer: Flights Off
        </button>
        <button
          onClick={() => invoke('runDirectorScene', { action: 'list' })}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Director List
        </button>
        <button
          onClick={() => invoke('captureScreenshot', {})}
          disabled={!ready}
          className="px-3 py-2 rounded bg-electric-blue/10 text-electric-blue hover:bg-electric-blue/20 disabled:opacity-50 text-sm transition-colors"
        >
          Screenshot
        </button>
      </div>
      {last && (
        <div className="text-xs text-text-muted border-t border-core-border/50 pt-2">
          {last}
        </div>
      )}
    </div>
  );
}

interface GlobeMICProps {
  send: (message: any) => void;
  ready: boolean;
}

export function GlobeMIC({ send, ready }: GlobeMICProps) {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [permissionStatus, setPermissionStatus] = useState<PermissionStatus>({
    granted: false,
    denied: false,
    prompt: false,
    unavailable: false,
  });
  const [error, setError] = useState<string | null>(null);
  const [showPermissionGuide, setShowPermissionGuide] = useState(false);
  const permissionManager = PermissionManager.getInstance();
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);

  // Check permission status on mount
  useEffect(() => {
    if (!ready) return;
    const checkStatus = async () => {
      try {
        const status = await permissionManager.checkMicrophonePermission();
        setPermissionStatus(status);
      } catch (e) {
        console.error('Permission check failed:', e);
      }
    };
    checkStatus();
  }, [ready]);

  const stopTracks = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  const handleMicClick = async () => {
    if (!ready) return;

    // If already denied, show guide
    if (permissionStatus.denied && !permissionStatus.prompt) {
      setShowPermissionGuide(true);
      setError('Microphone access denied. Please enable it in browser settings.');
      return;
    }

    try {
      setError(null);
      
      // Request microphone access with user gesture
      const stream = await permissionManager.requestMicrophoneWithUserGesture();
      mediaStreamRef.current = stream;
      
      setPermissionStatus({ granted: true, denied: false, prompt: false, unavailable: false });
      
      // Start speech recognition
      startSpeechRecognition();
    } catch (err) {
      const errorName = (err as Error)?.name || '';
      const errorMessage = (err as Error)?.message || 'Unknown error';
      
      let userMessage = '';
      let showGuide = false;

      switch (errorName) {
        case 'NotAllowedError':
          userMessage = 'Microphone permission denied. Please allow microphone access and try again.';
          showGuide = true;
          break;
        case 'NotFoundError':
          userMessage = 'No microphone found. Please connect a microphone and try again.';
          break;
        case 'NotReadableError':
          userMessage = 'Microphone is in use by another application. Please close other apps using the microphone.';
          break;
        case 'SecurityError':
          userMessage = 'Microphone access blocked by security policy. Please use HTTPS and try again.';
          showGuide = true;
          break;
        default:
          if (errorMessage.includes('timeout')) {
            userMessage = 'Permission request timed out. Please tap the microphone button again to retry.';
          } else if (errorMessage.includes('secure context') || errorMessage.includes('HTTPS')) {
            userMessage = 'Microphone requires HTTPS. Please access the site via HTTPS.';
            showGuide = true;
          } else {
            userMessage = `Microphone error: ${errorMessage}`;
          }
      }

      setError(userMessage);
      if (showGuide) setShowPermissionGuide(true);
      
      // Update permission status
      if (errorName === 'NotAllowedError' || errorName === 'SecurityError') {
        setPermissionStatus({ granted: false, denied: true, prompt: true, unavailable: false });
      } else if (errorName === 'NotFoundError') {
        setPermissionStatus({ granted: false, denied: false, prompt: false, unavailable: true });
      } else if (errorName === 'NotReadableError') {
        setPermissionStatus({ granted: false, denied: true, prompt: false, unavailable: false });
      }
    }
  };

  const startSpeechRecognition = () => {
    const SpeechRecognition =
      typeof window !== 'undefined'
        ? (window as any).SpeechRecognition ||
          (window as any).webkitSpeechRecognition
        : null;
    
    if (!SpeechRecognition) {
      setError('Speech recognition not supported in this browser');
      setPermissionStatus({ granted: false, denied: false, prompt: false, unavailable: true });
      return;
    }

    const rec = new SpeechRecognition();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = 'en-US';
    
    rec.onresult = (event: any) => {
      const result = event.results[0][0].transcript;
      setTranscript(result);
    };
    
    rec.onend = () => {
      setListening(false);
      if (transcript && transcript !== 'Listening…') {
        send(GlobeCommands.voiceCommand({ transcript }));
        setTranscript('');
      }
      stopTracks();
    };
    
    rec.onerror = (event: any) => {
      const errorName = event.error || 'unknown';
      setListening(false);
      
      switch (errorName) {
        case 'not-allowed':
          setError('Speech recognition permission denied. Please check browser settings.');
          setShowPermissionGuide(true);
          break;
        case 'no-speech':
          setTranscript('No speech detected. Try speaking louder or closer to the microphone.');
          break;
        case 'audio-capture':
          setError('Audio capture failed. Please check your microphone.');
          break;
        case 'network':
          setError('Network error during speech recognition.');
          break;
        default:
          setError(`Voice error: ${errorName}`);
      }
      stopTracks();
    };
    
    recognitionRef.current = rec;
    setListening(true);
    setTranscript('Listening…');
    rec.start();
  };

  const handleRetry = async () => {
    setShowPermissionGuide(false);
    setError(null);
    await handleMicClick();
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopTracks();
      if (recognitionRef.current) {
        try { recognitionRef.current.abort(); } catch {}
      }
    };
  }, [stopTracks]);

  // Determine button state
  const isMobile = permissionManager.isMobileDevice();
  const canRecord = ready && (permissionStatus.granted || permissionStatus.prompt);
  const isDenied = permissionStatus.denied && !permissionStatus.prompt;
  const isUnavailable = permissionStatus.unavailable;

  if (isUnavailable) {
    return (
      <div className="bg-core-card border border-core-border rounded-xl p-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-full bg-gray-700 text-gray-400">
            <Mic className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium text-text-primary">Microphone unavailable</div>
            <div className="text-[10px] text-text-muted">
              {isMobile ? 'Mobile browser does not support microphone access' : 'Microphone API not supported'}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-core-card border border-core-border rounded-xl p-3">
      <div className="flex items-center gap-3">
        <button
          onClick={handleMicClick}
          disabled={!canRecord || listening}
          className={cn(
            'p-2 rounded-full transition-colors flex items-center justify-center',
            listening && 'bg-red-500 animate-pulse',
            !listening && canRecord && 'bg-electric-blue hover:bg-electric-blue/80',
            !listening && !canRecord && 'bg-gray-700 text-gray-500 cursor-not-allowed',
            isDenied && 'bg-red-500/20 text-red-400',
          )}
          aria-label={listening ? 'Listening' : isDenied ? 'Microphone access denied - click for help' : 'Tap to speak'}
          title={listening 
            ? 'Listening... release to send' 
            : isDenied 
              ? 'Microphone access denied. Click for help.'
              : 'Tap to speak to the globe'}
        >
          <Mic className="w-5 h-5 text-white" />
        </button>
        
        <div className="flex-1 min-w-0">
          <div className="text-xs font-medium text-text-primary truncate">
            {error || transcript || (listening ? 'Listening…' : isDenied ? 'Microphone access denied' : 'Tap mic to speak to the globe')}
          </div>
          <div className="text-[10px] text-text-muted flex items-center gap-2">
            <span>{listening ? 'Listening…' : isDenied ? 'Blocked' : canRecord ? 'Ready' : 'Standby'}</span>
            {isMobile && !permissionStatus.granted && !listening && (
              <span className="px-1.5 py-0.5 text-[9px] bg-amber-500/20 text-amber-400 rounded">
                Mobile: tap to allow
              </span>
            )}
          </div>
        </div>

        {isDenied && (
          <button
            onClick={() => setShowPermissionGuide(true)}
            className="p-1.5 text-text-muted hover:text-amber-400"
            aria-label="Permission help"
          >
            <AlertCircle className="w-4 h-4" />
          </button>
        )}

        {showPermissionGuide && (
          <div className="absolute bottom-full left-0 right-0 mb-2 p-3 bg-gray-900 border border-red-500/50 rounded-lg text-xs z-10 animate-slide-up">
            <div className="flex items-start gap-2 text-red-300">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="font-medium mb-2">Microphone Access Blocked</p>
                <p className="mb-2 text-text-muted">
                  {permissionStatus.denied && 'Permission was denied. To fix this:'}
                  {permissionStatus.prompt && 'Permission is needed. To grant it:'}
                </p>
                <ul className="space-y-1 text-[11px] text-text-muted">
                  <li>• Chrome/Brave: Click the lock icon 🔒 in address bar → Microphone → Allow</li>
                  <li>• Firefox: Click the shield icon → Permissions → Microphone → Allow</li>
                  <li>• Safari: Safari menu → Settings → Websites → Microphone → Allow</li>
                  <li>• Android: Settings → Apps → Browser → Permissions → Microphone → Allow</li>
                  <li>• Close any overlay apps (screen recorders, etc.) that may block permission</li>
                </ul>
                <div className="flex gap-2 mt-3">
                  <button
                    onClick={handleRetry}
                    className="flex-1 px-2 py-1.5 bg-electric-blue text-white rounded text-xs hover:bg-electric-blue/80"
                  >
                    <RefreshCw className="w-3 h-3 inline mr-1" /> Retry
                  </button>
                  <button
                    onClick={() => setShowPermissionGuide(false)}
                    className="px-2 py-1.5 bg-gray-700 text-white rounded text-xs hover:bg-gray-600"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {error && !showPermissionGuide && !isDenied && (
          <div className="absolute bottom-full left-0 right-0 mb-2 p-2 bg-red-500/20 border border-red-500/50 rounded-lg text-xs text-red-300 z-10 animate-slide-up">
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-2 text-red-400 hover:text-red-300"
            >
              ✕
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
