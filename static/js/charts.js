/**
 * GYMSENSE AI - Hospital Vital-Style Waveforms & ML Analytics Visualizer
 * Student: Ranjitha B K | USN: 1SB24AI041
 * Branch: AIML | Event: MACHINE SPECTRA 1.0
 * 
 * High-performance 60 FPS Canvas Oscilloscopes for Accelerometer (X, Y, Z, |A|)
 * and Gyroscope (Alpha, Beta, Gamma, |G|), plus Confusion Matrix & Feature Importance renderers.
 */

class WaveformVisualizer {
  constructor(canvasId, maxPoints = 160, signalType = 'accel') {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.maxPoints = maxPoints;
    this.signalType = signalType; // 'accel' | 'gyro'
    this.dataBuffer = []; // array of { v1, v2, v3, mag }
    this.sweepX = 0;

    if (this.canvas) {
      this.handleResize();
      window.addEventListener('resize', () => this.handleResize());
    }
  }

  handleResize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx = this.canvas.getContext('2d');
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height;
  }

  pushSample(sample) {
    if (this.signalType === 'accel') {
      this.dataBuffer.push({
        v1: sample.accel_x || 0,
        v2: sample.accel_y || 0,
        v3: sample.accel_z || 9.81,
        mag: sample.accel_mag || 9.81
      });
    } else {
      this.dataBuffer.push({
        v1: sample.gyro_alpha || 0,
        v2: sample.gyro_beta || 0,
        v3: sample.gyro_gamma || 0,
        mag: sample.gyro_mag || 0
      });
    }

    if (this.dataBuffer.length > this.maxPoints) {
      this.dataBuffer.shift();
    }
  }

  render() {
    if (!this.ctx || !this.width || !this.height) return;
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    ctx.clearRect(0, 0, w, h);

    // 1. Draw Oscilloscope Dark Grid
    this.drawGrid(ctx, w, h);

    if (this.dataBuffer.length < 2) return;

    // 2. Compute dynamic scale range
    const centerY = h / 2;
    const scale = this.signalType === 'accel' ? (h * 0.4) / 25 : (h * 0.4) / 140;

    // Colors
    const c1 = this.signalType === 'accel' ? '#00f2fe' : '#c084fc'; // X / Alpha
    const c2 = this.signalType === 'accel' ? '#ff2e93' : '#fb923c'; // Y / Beta
    const c3 = this.signalType === 'accel' ? '#fbbf24' : '#38bdf8'; // Z / Gamma
    const cMag = this.signalType === 'accel' ? '#10b981' : '#34d399'; // Magnitude

    // 3. Draw Waveform Lines
    this.drawLineTrace(ctx, 'v1', c1, centerY, scale, 1.6);
    this.drawLineTrace(ctx, 'v2', c2, centerY, scale, 1.6);
    this.drawLineTrace(ctx, 'v3', c3, centerY, scale, 1.6);
    this.drawLineTrace(ctx, 'mag', cMag, centerY, scale, 1.2, true);

    // 4. Sweep Line Effect
    const sweepPointX = (this.dataBuffer.length / this.maxPoints) * w;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sweepPointX, 0);
    ctx.lineTo(sweepPointX, h);
    ctx.stroke();
  }

  drawGrid(ctx, w, h) {
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.07)';
    ctx.lineWidth = 1;

    // Horizontal gridlines
    const hDivisions = 4;
    for (let i = 1; i < hDivisions; i++) {
      const y = (h / hDivisions) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Vertical gridlines
    const vDivisions = 8;
    for (let j = 1; j < vDivisions; j++) {
      const x = (w / vDivisions) * j;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Center Baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawLineTrace(ctx, key, color, centerY, scale, lineWidth = 1.5, isDashed = false) {
    const w = this.width;
    const n = this.dataBuffer.length;
    const stepX = w / (this.maxPoints - 1);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    if (isDashed) ctx.setLineDash([2, 2]);

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = i * stepX;
      const val = this.dataBuffer[i][key];
      const y = centerY - (val * scale);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }
}

/* ============================================================
   ML MODEL INSPECTION & BENCHMARK RENDERER
   ============================================================ */
class MLInspectionRenderer {
  static async loadAndRender() {
    try {
      const resp = await fetch('/model-info');
      if (!resp.ok) return;
      const data = await resp.json();

      this.renderBenchmarkTable(data.model_comparison, data.selected_model);
      this.renderConfusionMatrix(data.confusion_matrix);
      this.renderFeatureImportances(data.top_feature_importances);
      this.renderDatasetStats(data);
    } catch (e) {
      console.warn('Failed to load ML inspection data:', e);
    }
  }

  static renderBenchmarkTable(comparison, winner) {
    const tableBody = document.getElementById('benchmark-table-body');
    if (!tableBody || !comparison) return;

    let html = '';
    for (const [modelName, metrics] of Object.entries(comparison)) {
      const isWinner = modelName === winner;
      html += `
        <tr class="${isWinner ? 'winner-row' : ''}">
          <td>
            <strong>${modelName}</strong>
            ${isWinner ? '<span class="winner-badge">★ BEST MODEL</span>' : ''}
          </td>
          <td><strong>${metrics.accuracy}%</strong></td>
          <td>${metrics.precision}%</td>
          <td>${metrics.recall}%</td>
          <td>${metrics.f1_score}%</td>
        </tr>
      `;
    }
    tableBody.innerHTML = html;
  }

  static renderConfusionMatrix(cmData) {
    const matrixContainer = document.getElementById('confusion-matrix-grid');
    if (!matrixContainer || !cmData) return;

    const matrix = cmData.matrix;
    const labels = cmData.labels;
    const n = labels.length;

    matrixContainer.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    let html = '';

    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const val = matrix[r][c];
        const isDiagonal = r === c;
        // Heatmap color intensity
        const bg = isDiagonal 
          ? `rgba(255, 46, 147, ${Math.min(1.0, 0.3 + (val / 70) * 0.7)})`
          : (val > 0 ? 'rgba(239, 68, 68, 0.4)' : 'rgba(15, 23, 42, 0.5)');

        html += `
          <div class="matrix-cell" style="background: ${bg}" title="Actual: ${labels[r]} | Pred: ${labels[c]} -> ${val}">
            ${val}
          </div>
        `;
      }
    }
    matrixContainer.innerHTML = html;
  }

  static renderFeatureImportances(topFeatures) {
    const container = document.getElementById('feature-importance-list');
    if (!container || !topFeatures) return;

    const maxImp = topFeatures[0] ? topFeatures[0].importance : 1.0;
    let html = '';

    for (const feat of topFeatures) {
      const pct = feat.importance;
      const widthPct = (pct / maxImp) * 100;
      html += `
        <div class="feat-bar-item">
          <div class="feat-bar-label">
            <span>${feat.feature}</span>
            <strong>${pct.toFixed(2)}%</strong>
          </div>
          <div class="feat-progress-track">
            <div class="feat-progress-fill" style="width: ${widthPct}%"></div>
          </div>
        </div>
      `;
    }
    container.innerHTML = html;
  }

  static renderDatasetStats(data) {
    const statsContainer = document.getElementById('model-stats-summary');
    if (!statsContainer || !data) return;

    statsContainer.innerHTML = `
      <div class="summary-exercise-row">
        <span>Trained Algorithm</span>
        <strong style="color: var(--primary-pink);">${data.selected_model}</strong>
      </div>
      <div class="summary-exercise-row">
        <span>Total Training Windows</span>
        <strong>${data.total_training_windows} windows (80% split)</strong>
      </div>
      <div class="summary-exercise-row">
        <span>Total Testing Windows</span>
        <strong>${data.total_testing_windows} windows (20% split)</strong>
      </div>
      <div class="summary-exercise-row">
        <span>Features Extracted</span>
        <strong>${data.total_features} features / window</strong>
      </div>
      <div class="summary-exercise-row">
        <span>Window Duration</span>
        <strong>${data.window_duration_seconds}s (${data.window_size_samples} samples @ ${data.sampling_rate_hz}Hz)</strong>
      </div>
      <div class="summary-exercise-row">
        <span>Classes Trained</span>
        <strong style="color: var(--neon-cyan);">${data.classes.length} classes</strong>
      </div>
    `;
  }
}

window.WaveformVisualizer = WaveformVisualizer;
window.MLInspectionRenderer = MLInspectionRenderer;
