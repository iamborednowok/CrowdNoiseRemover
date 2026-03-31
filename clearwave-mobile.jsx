import React, { useState, useRef } from "react";

// ── FFT ──
function fft(re, im) {
  const N = re.length;
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i],re[j]]=[re[j],re[i]]; [im[i],im[j]]=[im[j],im[i]]; }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = -2*Math.PI/len, wRe=Math.cos(ang), wIm=Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let uRe=1, uIm=0;
      for (let k = 0; k < (len>>1); k++) {
        const e=i+k, o=i+k+(len>>1);
        const tRe=uRe*re[o]-uIm*im[o], tIm=uRe*im[o]+uIm*re[o];
        re[o]=re[e]-tRe; im[o]=im[e]-tIm; re[e]+=tRe; im[e]+=tIm;
        const nu=uRe*wRe-uIm*wIm; uIm=uRe*wIm+uIm*wRe; uRe=nu;
      }
    }
  }
}
function ifft(re, im) {
  const N=re.length;
  for (let i=0;i<N;i++) im[i]=-im[i];
  fft(re,im);
  for (let i=0;i<N;i++){re[i]/=N;im[i]=-im[i]/N;}
}
function hannWindow(N) {
  const w=new Float32Array(N);
  for(let i=0;i<N;i++) w[i]=0.5*(1-Math.cos(2*Math.PI*i/(N-1)));
  return w;
}

// ── DSP ──
async function processChannel(samples, strength, sensitivity, profile, sampleRate, onProgress) {
  const N=2048, hop=N>>1, halfN=N>>1;
  const win=hannWindow(N), len=samples.length;
  if(len<N) return samples.slice();
  const totalFrames=Math.max(1,Math.floor((len-N)/hop)+1);
  const binHz=k=>k*(sampleRate||44100)/N;

  const noiseEst=new Float32Array(halfN).fill(Infinity);
  let fi=0;
  for(let pos=0;pos+N<=len;pos+=hop){
    const re=new Float32Array(N),im=new Float32Array(N);
    for(let i=0;i<N;i++) re[i]=samples[pos+i]*win[i];
    fft(re,im);
    for(let k=0;k<halfN;k++){const m=Math.hypot(re[k],im[k]);if(m<noiseEst[k])noiseEst[k]=m;}
    fi++;
    if(fi%80===0){onProgress(fi/totalFrames*0.35);await new Promise(r=>setTimeout(r,0));}
  }
  const R=profile==='live'?9:6;
  const smoothed=new Float32Array(halfN);
  for(let k=0;k<halfN;k++){
    let s=0,c=0;
    for(let j=Math.max(0,k-R);j<=Math.min(halfN-1,k+R);j++){s+=noiseEst[j];c++;}
    smoothed[k]=(profile==='live'?1.4:1.0)*s/c;
  }

  const output=new Float32Array(len+N),normBuf=new Float32Array(len+N);
  fi=0;
  for(let pos=0;pos+N<=len;pos+=hop){
    const re=new Float32Array(N),im=new Float32Array(N);
    for(let i=0;i<N;i++) re[i]=samples[pos+i]*win[i];
    fft(re,im);
    for(let k=1;k<halfN;k++){
      const mag=Math.hypot(re[k],im[k]);
      const phase=Math.atan2(im[k],re[k]);
      const hz=binHz(k);
      let sc=strength,fl=0.01;
      if(profile==='music'){
        if(hz<200){sc=strength*0.15;fl=0.60;}
        else if(hz<900){sc=strength*0.35;fl=0.40;}
        else if(hz<3500){sc=strength*0.65;fl=0.25;}
        else if(hz<8000){sc=strength*1.60;fl=0.08;}
        else{sc=strength*1.30;fl=0.06;}
      } else if(profile==='live'){
        if(hz<120){sc=strength*0.30;fl=0.45;}
        else if(hz<400){sc=strength*0.60;fl=0.30;}
        else if(hz<800){sc=strength*0.80;fl=0.22;}
        else if(hz<3500){sc=strength*1.20;fl=0.18;}
        else if(hz<8000){sc=strength*1.70;fl=0.07;}
        else{sc=strength*1.40;fl=0.05;}
      }
      const sub=sensitivity*smoothed[k]*sc;
      const nm=Math.max(mag-sub,fl*mag);
      re[k]=nm*Math.cos(phase); im[k]=nm*Math.sin(phase);
      re[N-k]=nm*Math.cos(-phase); im[N-k]=nm*Math.sin(-phase);
    }
    ifft(re,im);
    for(let i=0;i<N;i++){output[pos+i]+=re[i]*win[i];normBuf[pos+i]+=win[i]*win[i];}
    fi++;
    if(fi%80===0){onProgress(0.35+fi/totalFrames*0.65);await new Promise(r=>setTimeout(r,0));}
  }
  const result=new Float32Array(len);
  for(let i=0;i<len;i++) result[i]=normBuf[i]>1e-9?output[i]/normBuf[i]:0;
  return result;
}

// ── WAV encoder ──
function encodeWAV(buf){
  const nCh=buf.numberOfChannels,len=buf.length,sr=buf.sampleRate;
  const ab=new ArrayBuffer(44+len*nCh*2),v=new DataView(ab);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,'RIFF');v.setUint32(4,36+len*nCh*2,true);ws(8,'WAVE');ws(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,nCh,true);
  v.setUint32(24,sr,true);v.setUint32(28,sr*nCh*2,true);
  v.setUint16(32,nCh*2,true);v.setUint16(34,16,true);
  ws(36,'data');v.setUint32(40,len*nCh*2,true);
  let off=44;
  for(let i=0;i<len;i++)for(let c=0;c<nCh;c++){
    const s=Math.max(-1,Math.min(1,buf.getChannelData(c)[i]));
    v.setInt16(off,s<0?s*32768:s*32767,true);off+=2;
  }
  return ab;
}

// ── App ──
export default function App() {
  const [file, setFile]       = useState(null);
  const [origBuf, setOrigBuf] = useState(null);
  const [procBuf, setProcBuf] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress]     = useState(0);
  const [strength, setStrength]     = useState(0.70);
  const [sensitivity, setSensitivity] = useState(1.5);
  const [profile, setProfile]       = useState('standard');
  const [error, setError]           = useState(null);
  const [playing, setPlaying]       = useState(null);

  const audioCtx = useRef(null);
  const srcRef   = useRef(null);

  const getCtx = () => {
    if (!audioCtx.current || audioCtx.current.state === 'closed')
      audioCtx.current = new AudioContext();
    return audioCtx.current;
  };

  const stopPlay = () => {
    if (srcRef.current) { try { srcRef.current.stop(); } catch {} srcRef.current = null; }
    setPlaying(null);
  };

  const loadFile = async (f) => {
    setError(null); setFile(f); setProcBuf(null); stopPlay();
    try {
      const ab = await f.arrayBuffer();
      const buf = await getCtx().decodeAudioData(ab);
      setOrigBuf(buf);
    } catch {
      setError('Could not decode file. Try MP3, WAV, OGG, or M4A.');
      setFile(null);
    }
  };

  const handleProcess = async () => {
    if (!origBuf || processing) return;
    setProcessing(true); setProgress(0); setProcBuf(null); stopPlay();
    try {
      const ctx = getCtx();
      const nCh = origBuf.numberOfChannels;
      const newB = ctx.createBuffer(nCh, origBuf.length, origBuf.sampleRate);
      for (let ch = 0; ch < nCh; ch++) {
        const result = await processChannel(
          origBuf.getChannelData(ch), strength, sensitivity, profile, origBuf.sampleRate,
          p => setProgress(((ch + p) / nCh) * 100)
        );
        newB.getChannelData(ch).set(result);
      }
      setProcBuf(newB); setProgress(100);
    } catch (e) { setError('Processing failed: ' + e.message); }
    setProcessing(false);
  };

  const playBuf = (buf, which) => {
    if (playing === which) { stopPlay(); return; }
    stopPlay();
    const ctx = getCtx();
    if (ctx.state === 'suspended') ctx.resume();
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination); src.start();
    src.onended = () => setPlaying(null);
    srcRef.current = src; setPlaying(which);
  };

  const download = () => {
    if (!procBuf) return;
    const blob = new Blob([encodeWAV(procBuf)], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'cleaned_' + (file?.name.replace(/\.[^.]+$/, '') || 'audio') + '.wav';
    a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const profiles = [
    { id: 'standard', label: '⚙ Standard',   desc: 'Flat subtraction, works for any audio' },
    { id: 'music',    label: '🎵 Studio',     desc: 'Protects bass, targets 3.5–8 kHz chatter' },
    { id: 'live',     label: '🎤 Live',       desc: 'Aggressive mids for venue/crowd bleed' },
  ];

  const fmt = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;

  return (
    <div style={{minHeight:'100vh',background:'#0a0c14',color:'#c8d4e8',fontFamily:'system-ui,sans-serif',padding:'20px 16px',boxSizing:'border-box'}}>

      {/* Header */}
      <div style={{marginBottom:24}}>
        <div style={{fontSize:20,fontWeight:700,color:'#3de8ff',letterSpacing:2}}>CLEARWAVE</div>
        <div style={{fontSize:12,color:'#ffffff40',marginTop:2}}>Crowd Noise Remover · Browser-local</div>
      </div>

      {/* Upload */}
      <div
        onClick={() => document.getElementById('fi').click()}
        style={{border:'1px dashed #ffffff18',borderRadius:12,padding:'24px 16px',textAlign:'center',cursor:'pointer',marginBottom:16,background:'rgba(61,232,255,0.03)'}}>
        <input id="fi" type="file" accept="audio/*" style={{display:'none'}}
          onChange={e => e.target.files[0] && loadFile(e.target.files[0])} />
        {file ? (
          <div>
            <div style={{color:'#3de8ff',fontSize:14,marginBottom:4,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{file.name}</div>
            <div style={{fontSize:11,color:'#ffffff40'}}>{(file.size/1048576).toFixed(1)} MB{origBuf ? ` · ${fmt(origBuf.duration)}` : ''}</div>
          </div>
        ) : (
          <div>
            <div style={{fontSize:28,marginBottom:8}}>🎵</div>
            <div style={{fontSize:14,color:'#ffffff50'}}>Tap to choose audio file</div>
            <div style={{fontSize:11,color:'#ffffff25',marginTop:4}}>MP3 · WAV · OGG · M4A</div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div style={{background:'#1a0808',border:'1px solid #ff444440',borderRadius:8,padding:'10px 14px',color:'#ff6060',fontSize:13,marginBottom:16}}>
          {error}
        </div>
      )}

      {origBuf && (
        <>
          {/* Profile selector */}
          <div style={{marginBottom:16}}>
            <div style={{fontSize:11,color:'#ffffff40',letterSpacing:1,marginBottom:8}}>FILTER PROFILE</div>
            {profiles.map(p => (
              <div key={p.id}
                onClick={() => setProfile(p.id)}
                style={{
                  border: `1px solid ${profile===p.id ? '#3de8ff50' : '#ffffff10'}`,
                  borderRadius:8, padding:'10px 14px', marginBottom:8, cursor:'pointer',
                  background: profile===p.id ? 'rgba(61,232,255,0.06)' : 'transparent',
                  display:'flex', alignItems:'center', gap:10
                }}>
                <div style={{width:10,height:10,borderRadius:'50%',background:profile===p.id?'#3de8ff':'#ffffff20',flexShrink:0}}/>
                <div>
                  <div style={{fontSize:13,color:profile===p.id?'#3de8ff':'#8899aa'}}>{p.label}</div>
                  <div style={{fontSize:11,color:'#ffffff30',marginTop:2}}>{p.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Sliders */}
          <div style={{marginBottom:16}}>
            <div style={{fontSize:11,color:'#ffffff40',letterSpacing:1,marginBottom:12}}>CONTROLS</div>

            <div style={{marginBottom:16}}>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}>
                <span style={{fontSize:12,color:'#8899aa'}}>Reduction Strength</span>
                <span style={{fontSize:13,color:'#3de8ff',fontWeight:600}}>{Math.round(strength*100)}%</span>
              </div>
              <input type="range" min={0.1} max={1.0} step={0.05} value={strength}
                onChange={e=>setStrength(parseFloat(e.target.value))}
                style={{width:'100%',accentColor:'#3de8ff'}}/>
            </div>

            <div>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}>
                <span style={{fontSize:12,color:'#8899aa'}}>Sensitivity</span>
                <span style={{fontSize:13,color:'#3de8ff',fontWeight:600}}>{sensitivity.toFixed(1)}×</span>
              </div>
              <input type="range" min={0.5} max={3.0} step={0.1} value={sensitivity}
                onChange={e=>setSensitivity(parseFloat(e.target.value))}
                style={{width:'100%',accentColor:'#3de8ff'}}/>
            </div>
          </div>

          {/* Process button */}
          <button
            onClick={handleProcess}
            disabled={processing}
            style={{
              width:'100%', padding:'14px', borderRadius:10, border:'none', cursor:processing?'not-allowed':'pointer',
              background: processing ? '#1a2535' : 'linear-gradient(135deg,#3de8ff22,#0070ff22)',
              border: '1px solid ' + (processing ? '#ffffff10' : '#3de8ff50'),
              color: processing ? '#ffffff40' : '#3de8ff',
              fontSize:14, fontWeight:600, letterSpacing:1, marginBottom:16
            }}>
            {processing ? `Processing… ${Math.round(progress)}%` : '▶ Remove Crowd Noise'}
          </button>

          {/* Progress bar */}
          {processing && (
            <div style={{height:3,background:'#0e1825',borderRadius:2,marginBottom:16,overflow:'hidden'}}>
              <div style={{height:'100%',width:`${progress}%`,background:'linear-gradient(90deg,#3de8ff,#0070ff)',transition:'width 0.25s'}}/>
            </div>
          )}

          {/* Playback */}
          <div style={{display:'flex',flexDirection:'column',gap:10,marginBottom:16}}>
            <button onClick={() => playBuf(origBuf, 'orig')} style={btnStyle(playing==='orig','#3de8ff')}>
              {playing==='orig' ? '⏹ Stop Original' : '▶ Play Original'}
            </button>
            {procBuf && (
              <>
                <button onClick={() => playBuf(procBuf, 'proc')} style={btnStyle(playing==='proc','#ff7d45')}>
                  {playing==='proc' ? '⏹ Stop Cleaned' : '▶ Play Cleaned'}
                </button>
                <button onClick={download} style={btnStyle(false,'#50dd90')}>
                  ⬇ Download WAV
                </button>
              </>
            )}
          </div>
        </>
      )}

      <div style={{fontSize:11,color:'#ffffff15',textAlign:'center',marginTop:8}}>
        All processing happens in your browser — no uploads
      </div>
    </div>
  );
}

function btnStyle(active, color) {
  return {
    width:'100%', padding:'12px', borderRadius:8, cursor:'pointer',
    background: active ? color+'15' : 'transparent',
    border: `1px solid ${active ? color+'60' : '#ffffff15'}`,
    color: active ? color : '#8899aa',
    fontSize:13, fontWeight:500
  };
}
