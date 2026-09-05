/**
 * GYMSENSE AI - Workout Session, Rep Counter & ML Prediction Engine
 * Student: Ranjitha B K | USN: 1SB24AI041
 * Branch: AIML | Event: MACHINE SPECTRA 1.0
 * 
 * Manages live workout state, ML inference polling, prediction smoothing,
 * peak-detection rep counter, set tracking, and activity segmentation.
 */

class WorkoutManager {
  constructor() {
    this.isActive = false;
    this.startTime = null;
    this.elapsedSeconds = 0;
    this.activeExerciseSeconds = 0;
    this.timerInterval = null;
    this.predictInterval = null;

    // Current State
    this.currentActivity = 'REST';
    this.currentConfidence = 0.95;
    this.currentStability = 'Excellent';
    this.currentReps = 0;
    this.currentSet = 1;
    this.activityStartTime = null;

    // Smoothing & Anti-Flickering Buffer
    this.predictionBuffer = [];
    this.bufferSize = 3;

    // Peak Detection Rep Counter State
    this.lastPeakTime = 0;
    this.repSignalHistory = [];
    this.repThresholdHigh = 2.4;
    this.repThresholdLow = -1.2;
    this.repArmArmed = false;

    // Set & Segment Tracking
    this.activeSegments = [];
    this.exerciseSetHistory = {}; // { 'BICEP CURL': [12, 10], 'SQUAT': [15] }
    this.confidenceHistory = [];

    // Repetition-based exercises that support counting
    this.repCountableExercises = [
      'BICEP CURL',
      'HAMMER CURL',
      'SQUAT',
      'LUNGE',
      'JUMPING JACK',
      'SHOULDER PRESS',
      'FRONT RAISE',
      'LATERAL RAISE'
    ];
  }

  startWorkout() {
    if (this.isActive) return;
    this.isActive = true;
    this.startTime = Date.now();
    this.elapsedSeconds = 0;
    this.activeExerciseSeconds = 0;
    this.currentActivity = 'REST';
    this.currentConfidence = 0.98;
    this.currentStability = 'Excellent';
    this.currentReps = 0;
    this.currentSet = 1;
    this.activityStartTime = Date.now();
    this.activeSegments = [];
    this.exerciseSetHistory = {};
    this.predictionBuffer = [];
    this.confidenceHistory = [];

    // Timer Loop (1s tick)
    this.timerInterval = setInterval(() => {
      this.elapsedSeconds = Math.floor((Date.now() - this.startTime) / 1000);
      if (this.currentActivity !== 'REST') {
        this.activeExerciseSeconds += 1;
      }
      this.updateUI();
    }, 1000);

    // Real ML Inference Loop (~1.1s polling window)
    this.predictInterval = setInterval(() => {
      this.performInferenceStep();
    }, 1100);

    // Listen to live sensor stream for real-time rep peak detection
    window.sensorManager.onReading((reading) => {
      if (this.isActive) {
        this.processRepSignal(reading);
      }
    });

    this.updateUI();
  }

  stopWorkout() {
    if (!this.isActive) return null;
    this.isActive = false;

    if (this.timerInterval) clearInterval(this.timerInterval);
    if (this.predictInterval) clearInterval(this.predictInterval);

    // Finalize ongoing segment
    this.finalizeCurrentActivitySegment();

    // Compile comprehensive workout summary
    const summary = this.generateWorkoutSummary();

    // Save to LocalStorage History
    if (window.historyManager) {
      window.historyManager.saveWorkoutSession(summary);
    }

    return summary;
  }

  async performInferenceStep() {
    if (!this.isActive) return;

    const windowData = window.sensorManager.getRecentWindow();
    if (windowData.length < 10) return;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sensor_window: windowData })
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const result = await response.json();
      this.handlePredictionResult(result);
    } catch (err) {
      console.warn('ML Prediction inference fallback:', err);
    }
  }

  handlePredictionResult(result) {
    const rawActivity = result.activity || 'REST';
    const rawConf = result.confidence || 0.90;

    // Prediction Smoothing Buffer (Majority Voting)
    this.predictionBuffer.push(rawActivity);
    if (this.predictionBuffer.length > this.bufferSize) {
      this.predictionBuffer.shift();
    }

    // Count occurrences in buffer
    const counts = {};
    for (const act of this.predictionBuffer) {
      counts[act] = (counts[act] || 0) + 1;
    }

    let smoothedActivity = rawActivity;
    let maxCount = 0;
    for (const [act, cnt] of Object.entries(counts)) {
      if (cnt > maxCount) {
        maxCount = cnt;
        smoothedActivity = act;
      }
    }

    this.currentConfidence = rawConf;
    this.currentStability = result.stability || 'Excellent';
    this.confidenceHistory.push(rawConf);

    // Check for activity transition
    if (smoothedActivity !== this.currentActivity) {
      this.transitionToActivity(smoothedActivity);
    }

    this.updateUI();
  }

  transitionToActivity(newActivity) {
    // 1. Finalize previous segment
    this.finalizeCurrentActivitySegment();

    const prevActivity = this.currentActivity;
    this.currentActivity = newActivity;
    this.activityStartTime = Date.now();

    // 2. Set Tracking Logic
    if (newActivity !== 'REST') {
      if (!this.exerciseSetHistory[newActivity]) {
        this.exerciseSetHistory[newActivity] = [];
      }
      this.currentSet = this.exerciseSetHistory[newActivity].length + 1;
      this.currentReps = 0;
    } else {
      // Transitioning to Rest -> Save completed set reps for previous exercise
      if (prevActivity !== 'REST' && this.currentReps > 0) {
        if (!this.exerciseSetHistory[prevActivity]) {
          this.exerciseSetHistory[prevActivity] = [];
        }
        this.exerciseSetHistory[prevActivity].push(this.currentReps);
      }
    }

    // 3. Update Avatar Engine Animation automatically!
    if (window.avatarEngine) {
      window.avatarEngine.setActivity(newActivity);
    }
  }

  finalizeCurrentActivitySegment() {
    if (!this.activityStartTime) return;
    const duration = Math.max(1, Math.floor((Date.now() - this.activityStartTime) / 1000));
    
    // Only record segments longer than 1 second
    if (duration >= 1) {
      this.activeSegments.push({
        activity: this.currentActivity,
        startTime: this.activityStartTime,
        endTime: Date.now(),
        durationSeconds: duration,
        reps: this.repCountableExercises.includes(this.currentActivity) ? this.currentReps : 0,
        set: this.currentSet,
        confidence: Math.round(this.currentConfidence * 100)
      });
    }
  }

  /* ==========================================================
     REAL-TIME PEAK-DETECTION REP COUNTER
     ========================================================== */
  processRepSignal(reading) {
    if (!this.repCountableExercises.includes(this.currentActivity)) {
      return;
    }

    // Select primary signal axis based on current kinematic movement
    let signal = reading.accel_y - 9.81; // Default vertical displacement
    if (['LATERAL RAISE', 'JUMPING JACK'].includes(this.currentActivity)) {
      signal = reading.accel_x; // Lateral axis
    } else if (['FRONT RAISE', 'BICEP CURL', 'HAMMER CURL'].includes(this.currentActivity)) {
      signal = reading.accel_z - 4.0; // Sagittal/pitch axis
    }

    // Smoothing filter
    this.repSignalHistory.push(signal);
    if (this.repSignalHistory.length > 5) this.repSignalHistory.shift();
    const smoothedSignal = this.repSignalHistory.reduce((a, b) => a + b, 0) / this.repSignalHistory.length;

    const now = Date.now();
    const minRepDurationMs = 900; // Refractory period (~0.9s per repetition)

    // State Machine for Peak & Trough Detection
    if (!this.repArmArmed && smoothedSignal > this.repThresholdHigh) {
      this.repArmArmed = true; // Armed at peak of contraction
    } else if (this.repArmArmed && smoothedSignal < this.repThresholdLow) {
      if (now - this.lastPeakTime > minRepDurationMs) {
        this.currentReps += 1;
        this.lastPeakTime = now;
        this.repArmArmed = false;

        // Trigger visual pulse on avatar
        if (window.avatarEngine) {
          window.avatarEngine.triggerRepEffect();
        }

        this.updateUI();
      }
    }
  }

  generateWorkoutSummary() {
    const totalDuration = this.elapsedSeconds;
    const activeDuration = this.activeExerciseSeconds;
    
    // Group exercise segments by type
    const exerciseSummary = {};
    let totalReps = 0;
    let totalSets = 0;

    for (const seg of this.activeSegments) {
      const act = seg.activity;
      if (!exerciseSummary[act]) {
        exerciseSummary[act] = {
          activity: act,
          totalDuration: 0,
          totalReps: 0,
          sets: 0,
          avgConfidence: 0,
          count: 0
        };
      }
      exerciseSummary[act].totalDuration += seg.durationSeconds;
      exerciseSummary[act].totalReps += seg.reps;
      exerciseSummary[act].sets = Math.max(exerciseSummary[act].sets, seg.set || 1);
      exerciseSummary[act].avgConfidence += seg.confidence;
      exerciseSummary[act].count += 1;
    }

    for (const k in exerciseSummary) {
      const item = exerciseSummary[k];
      item.avgConfidence = Math.round(item.avgConfidence / item.count);
      totalReps += item.totalReps;
      if (k !== 'REST') {
        totalSets += item.sets;
      }
    }

    return {
      id: 'ws_' + Date.now(),
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      totalDurationSeconds: totalDuration,
      activeDurationSeconds: activeDuration,
      totalReps: totalReps,
      totalSets: totalSets,
      exercises: Object.values(exerciseSummary),
      segments: this.activeSegments,
      avatar: window.currentAvatarGender || 'female'
    };
  }

  formatDuration(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  updateUI() {
    // Workout Screen elements
    const timerElem = document.getElementById('live-workout-timer');
    const exerciseTitleElem = document.getElementById('detected-exercise-name');
    const confValElem = document.getElementById('metric-confidence-val');
    const repsValElem = document.getElementById('metric-reps-val');
    const setValElem = document.getElementById('metric-set-val');
    const stabilityValElem = document.getElementById('metric-stability-val');
    const activePillElem = document.getElementById('detected-exercise-pill');

    if (timerElem) timerElem.textContent = this.formatDuration(this.elapsedSeconds);
    if (exerciseTitleElem) exerciseTitleElem.textContent = this.currentActivity;
    if (confValElem) confValElem.textContent = `${Math.round(this.currentConfidence * 100)}%`;
    if (repsValElem) repsValElem.textContent = this.currentReps;
    if (setValElem) setValElem.textContent = `${this.currentSet} / 3`;
    if (stabilityValElem) stabilityValElem.textContent = this.currentStability;

    if (activePillElem) {
      activePillElem.textContent = this.currentActivity === 'REST' ? 'RESTING' : 'EXERCISE DETECTED';
    }
  }
}

window.workoutManager = new WorkoutManager();
