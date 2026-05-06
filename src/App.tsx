import React, { useRef, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Eraser, Search, RefreshCw, Cpu, BrainCircuit, Activity } from 'lucide-react';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);

  useEffect(() => {
    // Check if backend is up and model is loaded
    const checkStatus = async () => {
      try {
        const response = await fetch('/health');
        const data = await response.json();
        if (data.status === 'ok' && data.model_loaded) {
          setIsModelReady(true);
        } else {
          // If model is not loaded, it might be training
          setTimeout(checkStatus, 2000);
        }
      } catch (err) {
        // Backend might still be starting
        setTimeout(checkStatus, 2000);
      }
    };
    checkStatus();

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 15;
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, []);

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.beginPath(); // Reset path
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    let x, y;
    if ('touches' in e) {
      const rect = canvas.getBoundingClientRect();
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.nativeEvent.offsetX;
      y = e.nativeEvent.offsetY;
    }

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) {
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      setPrediction(null);
      setConfidence(null);
    }
  };

  const predictDigit = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setIsLoading(true);
    const imageData = canvas.toDataURL('image/png');

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) throw new Error('Prediction failed');

      const data = await response.json();
      setPrediction(data.prediction);
      setConfidence(data.confidence);
    } catch (err) {
      console.error(err);
      alert('Error connecting to AI service. Make sure the model is trained.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-50 font-sans text-neutral-900 selection:bg-blue-100">
      {/* Header */}
      <header className="border-b border-neutral-200 bg-white px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-200">
              <BrainCircuit size={24} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">NeuroGraph</h1>
              <p className="text-xs text-neutral-500 font-mono">MNIST CLASSIFIER v1.0</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden items-center gap-2 md:flex">
               <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
               <span className="text-xs font-medium text-neutral-500 uppercase tracking-widest leading-none">Model Status: {isModelReady ? 'Ready' : 'Initializing...'}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-12">
        <div className="grid gap-12 lg:grid-cols-2">
          {/* Drawing Area */}
          <section className="space-y-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-semibold tracking-tight text-neutral-800">Handwriting Input</h2>
              <p className="text-neutral-500">Draw a single digit (0-9) inside the box below. Use your mouse or touch screen.</p>
            </div>

            <div className="relative aspect-square overflow-hidden rounded-2xl border-2 border-neutral-200 bg-white shadow-xl shadow-neutral-100 ring-1 ring-black/5 transition-all focus-within:border-blue-500 focus-within:ring-4 focus-within:ring-blue-100">
              <canvas
                ref={canvasRef}
                width={400}
                height={400}
                onMouseDown={startDrawing}
                onMouseUp={stopDrawing}
                onMouseMove={draw}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchEnd={stopDrawing}
                onTouchMove={draw}
                className="h-full w-full cursor-crosshair touch-none"
                id="drawing-canvas"
              />
              
              {!isModelReady && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 backdrop-blur-sm">
                   <RefreshCw className="mb-4 animate-spin text-blue-600" size={32} />
                   <p className="font-medium text-neutral-700">Training Intelligence...</p>
                   <p className="text-xs text-neutral-500 mt-1">Downloading MNIST dataset</p>
                </div>
              )}
            </div>

            <div className="flex gap-3">
              <button
                onClick={clearCanvas}
                className="flex items-center gap-2 rounded-xl border border-neutral-200 bg-white px-5 py-2.5 text-sm font-semibold text-neutral-700 transition hover:bg-neutral-50 hover:border-neutral-300 disabled:opacity-50"
                id="clear-btn"
              >
                <Eraser size={18} />
                Clear
              </button>
              <button
                onClick={predictDigit}
                disabled={isLoading || !isModelReady}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-bold text-white shadow-lg shadow-blue-200 transition hover:bg-blue-700 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100"
                id="predict-btn"
              >
                {isLoading ? (
                  <RefreshCw className="animate-spin" size={18} />
                ) : (
                  <Search size={18} />
                )}
                Analyze Digit
              </button>
            </div>
          </section>

          {/* Results Area */}
          <section className="flex flex-col gap-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-semibold tracking-tight text-neutral-800">Inference Results</h2>
              <p className="text-neutral-500">Real-time classification based on a Random Forest ensemble model.</p>
            </div>

            <div className="flex-1 space-y-6">
              <div className="rounded-3xl border border-neutral-200 bg-white p-8 shadow-xl shadow-neutral-100">
                <AnimatePresence mode="wait">
                  {prediction !== null ? (
                    <motion.div
                      key="result"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="flex flex-col items-center justify-center space-y-4"
                    >
                      <div className="text-neutral-400 font-mono text-xs uppercase tracking-widest">Prediction</div>
                      <div className="text-[120px] font-black leading-none text-blue-600 tabular-nums">
                        {prediction}
                      </div>
                      <div className="w-full space-y-2">
                        <div className="flex justify-between text-xs font-bold text-neutral-500">
                          <span>CONFIDENCE LEVEL</span>
                          <span>{(confidence! * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-100">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${confidence! * 100}%` }}
                            className="h-full bg-blue-600"
                          />
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <div className="flex h-full min-h-[250px] flex-col items-center justify-center space-y-4 text-center text-neutral-400">
                      <div className="rounded-full bg-neutral-50 p-4">
                        <Activity size={40} />
                      </div>
                      <p className="text-sm">Draw a digit and click analyze to see results</p>
                    </div>
                  )}
                </AnimatePresence>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-2xl border border-neutral-200 bg-white p-4">
                  <div className="flex items-center gap-2 text-neutral-500 mb-1">
                    <Cpu size={14} />
                    <span className="text-[10px] font-bold uppercase tracking-widest leading-none">Architecture</span>
                  </div>
                  <div className="text-sm font-semibold truncate leading-none">Random Forest</div>
                </div>
                <div className="rounded-2xl border border-neutral-200 bg-white p-4">
                   <div className="flex items-center gap-2 text-neutral-500 mb-1">
                    <Activity size={14} />
                    <span className="text-[10px] font-bold uppercase tracking-widest leading-none">Dataset</span>
                  </div>
                  <div className="text-sm font-semibold truncate leading-none">MNIST-784</div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>

      <footer className="mx-auto max-w-5xl px-6 py-8 text-center text-[10px] font-bold uppercase tracking-[0.2em] text-neutral-400">
        Powered by Scikit-Learn & FastAPI • Distributed via AI Studio
      </footer>
    </div>
  );
}
