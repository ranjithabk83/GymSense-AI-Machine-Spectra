/**
 * GYMSENSE AI - Workout History & Analytics Manager
 * Student: Ranjitha B K | USN: 1SB24AI041
 * Branch: AIML | Event: MACHINE SPECTRA 1.0
 * 
 * Manages LocalStorage workout persistence, daily totals across repeated exercises,
 * session detail views, and real analytics computations.
 */

class HistoryManager {
  constructor() {
    this.storageKey = 'gymsense_workout_history_v1';
    this.sessions = this.loadSessions();
  }

  loadSessions() {
    try {
      const data = localStorage.getItem(this.storageKey);
      return data ? JSON.parse(data) : [];
    } catch (e) {
      console.warn('Failed to load history:', e);
      return [];
    }
  }

  saveWorkoutSession(session) {
    if (!session) return;
    this.sessions.unshift(session);
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.sessions));
    } catch (e) {
      console.warn('Failed to persist workout session:', e);
    }
    this.renderHistoryScreen();
    this.renderAnalyticsScreen();
    this.updateHomeStats();
  }

  getTodaySessions() {
    const todayStr = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    return this.sessions.filter(s => s.date === todayStr);
  }

  getDailyExerciseTotals() {
    const todaySessions = this.getTodaySessions();
    const totals = {};

    for (const sess of todaySessions) {
      if (sess.exercises) {
        for (const ex of sess.exercises) {
          const act = ex.activity;
          if (!totals[act]) {
            totals[act] = {
              activity: act,
              totalDurationSeconds: 0,
              totalReps: 0,
              totalSets: 0
            };
          }
          totals[act].totalDurationSeconds += ex.totalDuration || 0;
          totals[act].totalReps += ex.totalReps || 0;
          totals[act].totalSets += ex.sets || 0;
        }
      }
    }
    return Object.values(totals);
  }

  renderHistoryScreen() {
    const container = document.getElementById('history-sessions-list');
    if (!container) return;

    if (this.sessions.length === 0) {
      container.innerHTML = `
        <div class="gym-card" style="text-align: center; padding: 30px 20px;">
          <p style="color: var(--text-muted); margin-bottom: 8px;">No workout sessions recorded yet.</p>
          <p style="font-size: 12px; color: var(--text-secondary);">Complete a live workout to view automated activity segments and set breakdowns.</p>
        </div>
      `;
      return;
    }

    let html = '';
    this.sessions.forEach((sess, idx) => {
      const durationFormatted = workoutManager.formatDuration(sess.totalDurationSeconds || 0);
      const activeFormatted = workoutManager.formatDuration(sess.activeDurationSeconds || 0);

      html += `
        <div class="gym-card" style="cursor: pointer;" onclick="historyManager.toggleSessionDetails(${idx})">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
              <strong style="font-size: 15px; color: #fff;">${sess.date}</strong>
              <span style="font-size: 11px; color: var(--text-muted); margin-left: 6px;">${sess.time || ''}</span>
            </div>
            <span class="sensor-badge" style="background: rgba(255, 46, 147, 0.12); color: #ff66b2; border-color: rgba(255, 46, 147, 0.3);">
              ${sess.avatar.toUpperCase()} AVATAR
            </span>
          </div>

          <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); margin-bottom: 12px;">
            <span>Duration: <strong style="color: #fff;">${durationFormatted}</strong></span>
            <span>Active: <strong style="color: var(--neon-cyan);">${activeFormatted}</strong></span>
            <span>Reps: <strong style="color: var(--primary-pink);">${sess.totalReps || 0}</strong></span>
            <span>Sets: <strong style="color: var(--neon-green);">${sess.totalSets || 0}</strong></span>
          </div>

          <!-- Exercises list chips -->
          <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            ${(sess.exercises || []).map(e => `
              <span style="font-size: 10px; background: rgba(255, 255, 255, 0.06); padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border-subtle);">
                ${e.activity}: ${workoutManager.formatDuration(e.totalDuration)} (${e.totalReps} reps)
              </span>
            `).join('')}
          </div>

          <!-- Expandable Detail Segments -->
          <div id="session-detail-${idx}" style="display: none; margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.06);">
            <p style="font-size: 11px; text-transform: uppercase; color: var(--neon-cyan); letter-spacing: 0.5px; margin-bottom: 8px;">Individual Activity Timeline</p>
            ${(sess.segments || []).map(seg => `
              <div style="display: flex; justify-content: space-between; font-size: 11px; padding: 4px 0; border-bottom: 1px dashed rgba(255,255,255,0.04);">
                <span>${seg.activity}</span>
                <span style="color: var(--text-muted);">${workoutManager.formatDuration(seg.durationSeconds)} | ${seg.reps} reps | Conf: ${seg.confidence}%</span>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    });

    container.innerHTML = html;
  }

  toggleSessionDetails(idx) {
    const el = document.getElementById(`session-detail-${idx}`);
    if (el) {
      el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }
  }

  renderAnalyticsScreen() {
    const todaySessions = this.getTodaySessions();
    let totalSec = 0;
    let totalReps = 0;
    let totalSets = 0;
    const actCounts = {};

    this.sessions.forEach(s => {
      totalSec += s.activeDurationSeconds || 0;
      totalReps += s.totalReps || 0;
      totalSets += s.totalSets || 0;
      if (s.exercises) {
        s.exercises.forEach(e => {
          actCounts[e.activity] = (actCounts[e.activity] || 0) + (e.totalDuration || 0);
        });
      }
    });

    // Find most frequent exercise
    let mostFreq = 'None';
    let maxDur = 0;
    for (const [act, dur] of Object.entries(actCounts)) {
      if (act !== 'REST' && dur > maxDur) {
        maxDur = dur;
        mostFreq = act;
      }
    }

    // Update Analytics UI metrics
    const actTimeElem = document.getElementById('analytics-active-time');
    const totalRepsElem = document.getElementById('analytics-total-reps');
    const totalSetsElem = document.getElementById('analytics-total-sets');
    const mostFreqElem = document.getElementById('analytics-most-frequent');
    const exerciseDistElem = document.getElementById('analytics-exercise-distribution');

    if (actTimeElem) actTimeElem.textContent = workoutManager.formatDuration(totalSec);
    if (totalRepsElem) totalRepsElem.textContent = totalReps;
    if (totalSetsElem) totalSetsElem.textContent = totalSets;
    if (mostFreqElem) mostFreqElem.textContent = mostFreq;

    // Render Distribution Bars
    if (exerciseDistElem) {
      if (Object.keys(actCounts).length === 0) {
        exerciseDistElem.innerHTML = '<p style="font-size: 12px; color: var(--text-muted);">No activity data available yet.</p>';
      } else {
        let html = '';
        for (const [act, dur] of Object.entries(actCounts)) {
          const pct = totalSec > 0 ? Math.round((dur / totalSec) * 100) : 0;
          html += `
            <div class="feat-bar-item">
              <div class="feat-bar-label">
                <span>${act}</span>
                <span>${workoutManager.formatDuration(dur)} (${pct}%)</span>
              </div>
              <div class="feat-progress-track">
                <div class="feat-progress-fill" style="width: ${Math.max(4, pct)}%"></div>
              </div>
            </div>
          `;
        }
        exerciseDistElem.innerHTML = html;
      }
    }
  }

  updateHomeStats() {
    const todaySessions = this.getTodaySessions();
    let todaySec = 0;
    let todayReps = 0;
    const todayActs = new Set();

    todaySessions.forEach(s => {
      todaySec += s.activeDurationSeconds || 0;
      todayReps += s.totalReps || 0;
      if (s.exercises) {
        s.exercises.forEach(e => {
          if (e.activity !== 'REST') todayActs.add(e.activity);
        });
      }
    });

    const homeActiveElem = document.getElementById('home-today-active');
    const homeExElem = document.getElementById('home-today-exercises');
    const homeRepsElem = document.getElementById('home-today-reps');

    if (homeActiveElem) homeActiveElem.textContent = workoutManager.formatDuration(todaySec);
    if (homeExElem) homeExElem.textContent = todayActs.size;
    if (homeRepsElem) homeRepsElem.textContent = todayReps;
  }
}

window.historyManager = new HistoryManager();
