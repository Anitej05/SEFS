// sounds.js — Web Audio API procedural synthesis for SEFS ambient feedback
// Pure oscillator-based: no external audio files needed

let audioCtx = null;

function getCtx() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioCtx;
}

/** Soft chime when a new cluster forms */
export function playClusterChime() {
    try {
        const ctx = getCtx();
        const now = ctx.currentTime;

        [523.25, 659.25, 783.99].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now + i * 0.12);
            gain.gain.setValueAtTime(0, now + i * 0.12);
            gain.gain.linearRampToValueAtTime(0.08, now + i * 0.12 + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.12 + 0.6);
            osc.connect(gain).connect(ctx.destination);
            osc.start(now + i * 0.12);
            osc.stop(now + i * 0.12 + 0.7);
        });
    } catch (e) { /* audio not supported */ }
}

/** Swoosh when a file is moved between clusters */
export function playFileMoveSwish() {
    try {
        const ctx = getCtx();
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        const filter = ctx.createBiquadFilter();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.exponentialRampToValueAtTime(200, now + 0.3);

        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(2000, now);
        filter.frequency.exponentialRampToValueAtTime(400, now + 0.3);

        gain.gain.setValueAtTime(0.06, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

        osc.connect(filter).connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.4);
    } catch (e) { /* audio not supported */ }
}

/** Soft ping when a file is uploaded */
export function playUploadPing() {
    try {
        const ctx = getCtx();
        const now = ctx.currentTime;

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(1046.5, now); // C6
        osc.frequency.exponentialRampToValueAtTime(1318.5, now + 0.1); // E6

        gain.gain.setValueAtTime(0.1, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);

        osc.connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.45);
    } catch (e) { /* audio not supported */ }
}

/** Subtle notification for NL command response */
export function playCommandResponse() {
    try {
        const ctx = getCtx();
        const now = ctx.currentTime;

        [880, 1108.73].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now + i * 0.08);
            gain.gain.setValueAtTime(0, now + i * 0.08);
            gain.gain.linearRampToValueAtTime(0.06, now + i * 0.08 + 0.03);
            gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.3);
            osc.connect(gain).connect(ctx.destination);
            osc.start(now + i * 0.08);
            osc.stop(now + i * 0.08 + 0.35);
        });
    } catch (e) { /* audio not supported */ }
}

/** Warning sound for duplicate detection */
export function playDuplicateAlert() {
    try {
        const ctx = getCtx();
        const now = ctx.currentTime;

        [440, 415.3].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(freq, now + i * 0.15);
            gain.gain.setValueAtTime(0.07, now + i * 0.15);
            gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.15 + 0.25);
            osc.connect(gain).connect(ctx.destination);
            osc.start(now + i * 0.15);
            osc.stop(now + i * 0.15 + 0.3);
        });
    } catch (e) { /* audio not supported */ }
}
