import { AnimatePresence, motion, useAnimation } from 'motion/react';
import React, { useEffect, useRef, useState } from 'react';

type AppState = 'idle' | 'running' | 'interrupted' | 'completed';

export default function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [progress, setProgress] = useState(0); // 0 to 60 seconds
  const [isInhale, setIsInhale] = useState(true);

  const videoRef = useRef<HTMLVideoElement>(null);
  
  // Animation controls
  const plantControls = useAnimation();
  const auraControls = useAnimation();
  const leftLeafControls = useAnimation();
  const rightLeafControls = useAnimation();

  // Initialize camera
  useEffect(() => {
    let isMounted = true;
    let stream: MediaStream | null = null;

    const initCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 1280 } }
        });
        if (isMounted && videoRef.current) {
          videoRef.current.srcObject = stream;
        } else if (stream) {
          stream.getTracks().forEach(track => track.stop());
        }
      } catch (err) {
        console.error('Camera access denied:', err);
      }
    };

    initCamera();
      
    return () => {
      isMounted = false;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, []);

  // Timer logic
  useEffect(() => {
    if (appState === 'running') {
      const interval = setInterval(() => {
        setProgress((p) => {
          if (p >= 60) {
            setAppState('completed');
            return 60;
          }
          return p + 0.1;
        });
      }, 100);
      return () => clearInterval(interval);
    }
  }, [appState]);

  // Breathing animation loop
  useEffect(() => {
    let isCancelled = false;

    const runSequence = async () => {
      while (!isCancelled && appState === 'running') {
        setIsInhale(true);
        // Inhale 4s
        await Promise.all([
          plantControls.start({ scaleY: 1.15, scaleX: 1.05, transition: { duration: 4, ease: 'easeOut' } }),
          auraControls.start({ opacity: 1, scale: 1.4, transition: { duration: 4, ease: 'easeOut' } }),
          leftLeafControls.start({ rotate: -25, opacity: 1, filter: 'brightness(1.5)', transition: { duration: 4, ease: 'easeOut' } }),
          rightLeafControls.start({ rotate: 25, opacity: 1, filter: 'brightness(1.5)', transition: { duration: 4, ease: 'easeOut' } }),
        ]);

        if (isCancelled || appState !== 'running') break;
        setIsInhale(false);

        // Exhale 6s
        await Promise.all([
          plantControls.start({ scaleY: 1, scaleX: 1, transition: { duration: 6, ease: 'easeInOut' } }),
          auraControls.start({ opacity: 0.3, scale: 1, transition: { duration: 6, ease: 'easeInOut' } }),
          leftLeafControls.start({ rotate: 10, opacity: 0.6, filter: 'brightness(0.9)', transition: { duration: 6, ease: 'easeInOut' } }),
          rightLeafControls.start({ rotate: -10, opacity: 0.6, filter: 'brightness(0.9)', transition: { duration: 6, ease: 'easeInOut' } }),
        ]);
      }
    };

    if (appState === 'running') {
      runSequence();
    } else if (appState === 'interrupted') {
      plantControls.stop();
      auraControls.stop();
      leftLeafControls.stop();
      rightLeafControls.stop();
      
      // Error flutter effect
      plantControls.start({
        opacity: [1, 0.4, 0.9, 0.5, 1],
        filter: ['brightness(1)', 'hue-rotate(90deg)', 'brightness(1)'],
        transition: { repeat: Infinity, duration: 1.5 }
      });
    } else if (appState === 'idle') {
      plantControls.set({ scaleY: 1, scaleX: 1, opacity: 1, filter: 'brightness(1)' });
      auraControls.set({ opacity: 0.3, scale: 1 });
      leftLeafControls.set({ rotate: 10, opacity: 0.6, filter: 'brightness(0.9)' });
      rightLeafControls.set({ rotate: -10, opacity: 0.6, filter: 'brightness(0.9)' });
      setProgress(0);
    } else if (appState === 'completed') {
       plantControls.start({
         scale: 1.2, opacity: 1,
         transition: { duration: 2, ease: 'easeOut'}
       });
       auraControls.start({
         opacity: 1, scale: 2,
         transition: { duration: 2, ease: 'easeOut'}
       });
    }

    return () => {
      isCancelled = true;
    };
  }, [appState, plantControls, auraControls, leftLeafControls, rightLeafControls]);

  const handlePointerDown = (e: React.PointerEvent) => {
    if (appState === 'completed') return;
    setAppState('running');
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (appState === 'completed') return;
    if (progress > 0) {
      setAppState('interrupted');
    }
  };

  return (
    <div className="w-full h-[100dvh] bg-[#050b14] flex justify-center overflow-hidden font-sans text-teal-50 selection:bg-teal-500/30">
      {/* Mobile constraint container */}
      <div className="w-full max-w-[480px] h-full flex flex-col relative shadow-2xl">
        
        {/* Top 40%: Camera Area (水镜) */}
        <div className="relative h-[40%] w-full overflow-hidden flex-shrink-0 z-10">
          <video
            ref={videoRef}
            className="absolute inset-0 w-full h-full object-cover transform -scale-x-100"
            playsInline
            muted
            onLoadedMetadata={() => {
              videoRef.current?.play().catch((e) => {
                if (e.name !== 'AbortError') {
                  console.error('Video play error:', e);
                }
              });
            }}
          />
          
          {/* Soft mask edge for storytelling effect */}
          <div 
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'radial-gradient(ellipse at center, transparent 30%, #050b14 90%)'
            }}
          />
          
          {/* Fog overlay for errors/interruptions */}
          <AnimatePresence>
            {appState === 'interrupted' && (
              <motion.div
                initial={{ opacity: 0, backdropFilter: 'blur(0px)' }}
                animate={{ opacity: 1, backdropFilter: 'blur(12px)' }}
                exit={{ opacity: 0, backdropFilter: 'blur(0px)' }}
                transition={{ duration: 0.5 }}
                className="absolute inset-0 bg-teal-950/60 pointer-events-none"
              />
            )}
          </AnimatePresence>

          {/* Intro Text Overlay in Camera frame */}
          <AnimatePresence>
             {appState === 'idle' && (
                <motion.div 
                   initial={{ opacity: 0, y: 10 }}
                   animate={{ opacity: 1, y: 0 }}
                   exit={{ opacity: 0 }}
                   className="absolute top-12 left-0 right-0 text-center"
                >
                   <h1 className="text-2xl tracking-widest font-light text-teal-100 drop-shadow-md">灵泉生木</h1>
                   <p className="mt-2 text-sm text-teal-300/80 tracking-wider">静息心率与协同呼吸基线测算</p>
                </motion.div>
             )}
          </AnimatePresence>
        </div>

        {/* Bottom 60%: Interaction Area (幽谷) */}
        <div className="relative h-[60%] w-full flex-grow bg-gradient-to-b from-[#050b14] via-[#091a24] to-[#010609] overflow-hidden flex flex-col items-center justify-end pb-12 z-20">
          
          {/* Background particles/decorations */}
          <div className="absolute inset-0 pointer-events-none opacity-30 bg-[radial-gradient(circle_at_50%_100%,_#0f3e3e_0%,_transparent_70%)]" />

          {/* Feedback Messages */}
          <div className="absolute top-8 left-0 w-full flex justify-center px-6 min-h-[3rem] items-center text-center">
            <AnimatePresence mode="wait">
              {appState === 'running' && (
                <motion.div
                  key={isInhale ? 'inhale' : 'exhale'}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className="tracking-[0.2em] text-teal-100/90 font-light text-lg drop-shadow-md"
                >
                  {isInhale ? '吸气... (4s)' : '呼气... (6s)'}
                </motion.div>
              )}
              {appState === 'interrupted' && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-amber-200/90 tracking-wide font-medium bg-amber-950/40 px-4 py-2 rounded-full border border-amber-500/20 shadow-lg backdrop-blur-md"
                >
                  灵气紊乱，请保持正对水面并稳住指尖
                </motion.div>
              )}
              {appState === 'idle' && (
                 <motion.div
                  key="idle-msg"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-teal-300/60 tracking-wider font-light text-sm"
                 >
                   稳定面部于水镜中央，长按下方灵根开始
                 </motion.div>
              )}
              {appState === 'completed' && (
                <motion.div
                  key="completed"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-emerald-300 tracking-widest text-xl font-medium drop-shadow-lg"
                >
                  测算完成
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Plant Animation Area */}
          <div className="flex-grow flex items-end justify-center mb-8 pointer-events-none">
            <motion.div 
              className="relative flex flex-col items-center justify-end w-40 h-56 origin-bottom z-10" 
              animate={plantControls}
            >
              {/* Central Glow Aura */}
              <motion.div 
                className="absolute inset-x-0 bottom-4 h-48 bg-teal-400/30 blur-3xl rounded-t-full mix-blend-screen"
                initial={{ opacity: 0.3 }}
                animate={auraControls}
              />
              
              {/* Left Leaf */}
               <motion.div 
                className="absolute bottom-16 right-1/2 w-16 h-28 bg-gradient-to-tr from-emerald-600/90 to-teal-200/50 rounded-tl-[100%] rounded-br-[100%] rounded-tr-sm rounded-bl-sm origin-bottom-right shadow-[0_0_15px_rgba(20,184,166,0.3)] backdrop-blur-sm"
                initial={{ rotate: 10, opacity: 0.6 }}
                animate={leftLeafControls}
              />
              {/* Right Leaf */}
              <motion.div 
                className="absolute bottom-16 left-1/2 w-16 h-28 bg-gradient-to-tl from-emerald-600/90 to-teal-200/50 rounded-tr-[100%] rounded-bl-[100%] rounded-tl-sm rounded-br-sm origin-bottom-left shadow-[0_0_15px_rgba(20,184,166,0.3)] backdrop-blur-sm"
                initial={{ rotate: -10, opacity: 0.6 }}
                animate={rightLeafControls}
              />

              {/* Stem */}
              <motion.div 
                className="w-1.5 h-32 bg-gradient-to-t from-emerald-900 via-emerald-500 to-teal-300 rounded-full shadow-[0_0_10px_rgba(52,211,153,0.5)] z-20"
              />
              
              {/* Inner glowing pulse traveling up */}
              {appState === 'running' && (
                 <motion.div 
                    className="absolute bottom-0 w-2 h-8 bg-white/70 blur-[2px] rounded-full z-30 mix-blend-overlay"
                    animate={{ y: [0, -100], opacity: [0, 1, 0] }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'easeOut' }}
                 />
              )}
            </motion.div>
          </div>

          {/* Root Interactive Button */}
          <div className="relative flex items-center justify-center pt-8 pb-4 z-30">
            {/* Progress Ring */}
            <svg className="absolute w-[110px] h-[110px] pointer-events-none -rotate-90">
              <circle 
                cx="55" cy="55" r="50" 
                fill="none" 
                stroke="rgba(20, 184, 166, 0.15)" 
                strokeWidth="2" 
              />
              <motion.circle 
                cx="55" cy="55" r="50" 
                fill="none" 
                stroke="#4fd1c5" 
                strokeWidth="4" 
                strokeLinecap="round"
                strokeDasharray={2 * Math.PI * 50}
                strokeDashoffset={2 * Math.PI * 50 * (1 - progress / 60)}
                className="transition-all duration-100 ease-linear shadow-[0_0_10px_#4fd1c5]"
              />
            </svg>

            {/* Long Press Target */}
            <motion.div 
              className="w-20 h-20 rounded-full bg-teal-950/60 border border-teal-500/40 relative flex items-center justify-center backdrop-blur-md cursor-pointer select-none touch-none group"
              onPointerDown={handlePointerDown}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              onPointerLeave={handlePointerUp}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="absolute inset-0 rounded-full bg-teal-500/20 group-active:bg-teal-400/40 transition-colors duration-300" />
              
              {/* Core Light */}
              <motion.div 
                className="w-8 h-8 rounded-full bg-teal-300 shadow-[0_0_20px_#5eead4]"
                animate={{ 
                  scale: appState === 'running' ? [1, 1.2, 1] : 1,
                  opacity: appState === 'completed' ? 0.3 : 1
                }}
                transition={{ duration: 1.5, repeat: appState === 'running' ? Infinity : 0 }}
              />
            </motion.div>
          </div>

          {/* Time text below button */}
          <div className="absolute bottom-4 text-teal-500/60 font-mono text-xs tracking-widest">
            {progress > 0 ? `T-${Math.max(0, Math.ceil(60 - progress))}s` : '60s'}
          </div>

        </div>
      </div>
    </div>
  );
}
