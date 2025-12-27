// Funscript filters - ported from web app
// These filters process arrays of FunscriptPoints { at: number, pos: number }

import type { FunscriptPoint } from '../api/types';

/**
 * Smooth filter - applies moving average smoothing
 */
export function smooth(points: FunscriptPoint[], strength: number = 0.5): FunscriptPoint[] {
  if (points.length < 3) return points;

  const windowSize = Math.max(3, Math.floor(strength * 10) | 1); // Ensure odd number
  const halfWindow = Math.floor(windowSize / 2);

  return points.map((point, i) => {
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(points.length - 1, i + halfWindow);
    const window = points.slice(start, end + 1);
    const avgPos = window.reduce((sum, p) => sum + p.pos, 0) / window.length;

    return { at: point.at, pos: Math.round(avgPos) };
  });
}

/**
 * Savitzky-Golay filter - polynomial smoothing
 */
export function savitzkyGolay(points: FunscriptPoint[], windowSize: number = 5): FunscriptPoint[] {
  if (points.length < windowSize) return points;

  const halfWindow = Math.floor(windowSize / 2);

  return points.map((point, i) => {
    if (i < halfWindow || i >= points.length - halfWindow) {
      return point;
    }

    // Simplified quadratic fit
    let sum = 0;
    for (let j = -halfWindow; j <= halfWindow; j++) {
      const weight = 1 - Math.abs(j) / (halfWindow + 1);
      sum += points[i + j].pos * weight;
    }

    const weightSum = Array.from({ length: windowSize }, (_, j) =>
      1 - Math.abs(j - halfWindow) / (halfWindow + 1)
    ).reduce((a, b) => a + b, 0);

    return { at: point.at, pos: Math.round(sum / weightSum) };
  });
}

/**
 * Anti-jerk filter - removes sudden direction changes
 */
export function antiJerk(points: FunscriptPoint[], threshold: number = 50): FunscriptPoint[] {
  if (points.length < 3) return points;

  return points.map((point, i) => {
    if (i === 0 || i === points.length - 1) return point;

    const prev = points[i - 1];
    const next = points[i + 1];
    const prevDiff = point.pos - prev.pos;
    const nextDiff = next.pos - point.pos;

    // Check for direction reversal with large magnitude
    if (prevDiff * nextDiff < 0 && Math.abs(prevDiff) > threshold && Math.abs(nextDiff) > threshold) {
      // Smooth out the spike
      return { at: point.at, pos: Math.round((prev.pos + next.pos) / 2) };
    }

    return point;
  });
}

/**
 * Amplify filter - increases movement range
 */
export function amplify(points: FunscriptPoint[], amount: number = 1.5): FunscriptPoint[] {
  const center = 50;

  return points.map(point => {
    const deviation = point.pos - center;
    const amplified = center + deviation * amount;
    return { at: point.at, pos: Math.round(Math.max(0, Math.min(100, amplified))) };
  });
}

/**
 * Reduce filter - decreases movement range
 */
export function reduce(points: FunscriptPoint[], amount: number = 0.5): FunscriptPoint[] {
  return amplify(points, amount);
}

/**
 * Invert filter - flips positions (0 becomes 100, 100 becomes 0)
 */
export function invert(points: FunscriptPoint[]): FunscriptPoint[] {
  return points.map(point => ({
    at: point.at,
    pos: 100 - point.pos,
  }));
}

/**
 * Offset filter - shifts all positions by a fixed amount
 */
export function offset(points: FunscriptPoint[], amount: number): FunscriptPoint[] {
  return points.map(point => ({
    at: point.at,
    pos: Math.max(0, Math.min(100, point.pos + amount)),
  }));
}

/**
 * Clamp/Limit Range filter - constrains positions to min/max
 */
export function clamp(points: FunscriptPoint[], min: number = 0, max: number = 100): FunscriptPoint[] {
  const range = max - min;

  return points.map(point => ({
    at: point.at,
    pos: Math.round(min + (point.pos / 100) * range),
  }));
}

/**
 * Time shift filter - moves all points forward or backward in time
 */
export function timeShift(points: FunscriptPoint[], offsetMs: number): FunscriptPoint[] {
  return points
    .map(point => ({
      at: point.at + offsetMs,
      pos: point.pos,
    }))
    .filter(point => point.at >= 0);
}

/**
 * Remove duplicates filter - removes consecutive points with same position
 */
export function removeDuplicates(points: FunscriptPoint[], tolerance: number = 0): FunscriptPoint[] {
  if (points.length < 2) return points;

  return points.filter((point, i) => {
    if (i === 0) return true;
    return Math.abs(point.pos - points[i - 1].pos) > tolerance;
  });
}

/**
 * RDP Simplify filter - reduces number of points using Ramer-Douglas-Peucker algorithm
 */
export function rdpSimplify(points: FunscriptPoint[], epsilon: number = 5): FunscriptPoint[] {
  if (points.length < 3) return points;

  // Normalize time to position scale for distance calculation
  const maxTime = Math.max(...points.map(p => p.at));

  function perpendicularDistance(
    point: FunscriptPoint,
    lineStart: FunscriptPoint,
    lineEnd: FunscriptPoint
  ): number {
    const dx = (lineEnd.at - lineStart.at) / maxTime * 100;
    const dy = lineEnd.pos - lineStart.pos;
    const mag = Math.sqrt(dx * dx + dy * dy);

    if (mag === 0) {
      const pdx = (point.at - lineStart.at) / maxTime * 100;
      const pdy = point.pos - lineStart.pos;
      return Math.sqrt(pdx * pdx + pdy * pdy);
    }

    const u = ((point.at - lineStart.at) / maxTime * 100 * dx + (point.pos - lineStart.pos) * dy) / (mag * mag);
    const closestX = (lineStart.at / maxTime * 100) + u * dx;
    const closestY = lineStart.pos + u * dy;
    const distX = (point.at / maxTime * 100) - closestX;
    const distY = point.pos - closestY;

    return Math.sqrt(distX * distX + distY * distY);
  }

  function rdp(start: number, end: number): FunscriptPoint[] {
    if (end <= start + 1) {
      return [points[start]];
    }

    let maxDistance = 0;
    let maxIndex = start;

    for (let i = start + 1; i < end; i++) {
      const distance = perpendicularDistance(points[i], points[start], points[end]);
      if (distance > maxDistance) {
        maxDistance = distance;
        maxIndex = i;
      }
    }

    if (maxDistance > epsilon) {
      const left = rdp(start, maxIndex);
      const right = rdp(maxIndex, end);
      return [...left, ...right];
    }

    return [points[start]];
  }

  const result = rdp(0, points.length - 1);
  result.push(points[points.length - 1]);
  return result;
}

/**
 * Keyframes filter - keeps only points where significant changes occur
 */
export function keyframes(points: FunscriptPoint[], minChange: number = 10): FunscriptPoint[] {
  if (points.length < 2) return points;

  const result: FunscriptPoint[] = [points[0]];

  for (let i = 1; i < points.length; i++) {
    const lastKept = result[result.length - 1];
    if (Math.abs(points[i].pos - lastKept.pos) >= minChange) {
      result.push(points[i]);
    }
  }

  // Always keep the last point
  if (result[result.length - 1].at !== points[points.length - 1].at) {
    result.push(points[points.length - 1]);
  }

  return result;
}

/**
 * Speed limit filter - ensures movements don't exceed a maximum speed
 */
export function speedLimit(points: FunscriptPoint[], maxSpeed: number = 400): FunscriptPoint[] {
  if (points.length < 2) return points;

  const result: FunscriptPoint[] = [points[0]];

  for (let i = 1; i < points.length; i++) {
    const prev = result[result.length - 1];
    const current = points[i];
    const timeDiff = (current.at - prev.at) / 1000; // Convert to seconds

    if (timeDiff <= 0) {
      result.push(current);
      continue;
    }

    const posDiff = Math.abs(current.pos - prev.pos);
    const speed = posDiff / timeDiff;

    if (speed > maxSpeed) {
      // Limit the position change
      const maxPosDiff = maxSpeed * timeDiff;
      const direction = current.pos > prev.pos ? 1 : -1;
      const limitedPos = Math.round(prev.pos + direction * maxPosDiff);
      result.push({ at: current.at, pos: Math.max(0, Math.min(100, limitedPos)) });
    } else {
      result.push(current);
    }
  }

  return result;
}

/**
 * Interpolate filter - adds points between existing points for smoother motion
 */
export function interpolate(points: FunscriptPoint[], interval: number = 50): FunscriptPoint[] {
  if (points.length < 2) return points;

  const result: FunscriptPoint[] = [];

  for (let i = 0; i < points.length - 1; i++) {
    const current = points[i];
    const next = points[i + 1];
    result.push(current);

    const timeDiff = next.at - current.at;
    const steps = Math.floor(timeDiff / interval);

    for (let j = 1; j < steps; j++) {
      const t = j / steps;
      const interpolatedAt = current.at + t * timeDiff;
      const interpolatedPos = current.pos + t * (next.pos - current.pos);
      result.push({ at: Math.round(interpolatedAt), pos: Math.round(interpolatedPos) });
    }
  }

  result.push(points[points.length - 1]);
  return result;
}

/**
 * Noise filter - adds random variation to positions
 */
export function noise(points: FunscriptPoint[], amount: number = 5): FunscriptPoint[] {
  return points.map(point => ({
    at: point.at,
    pos: Math.max(0, Math.min(100, Math.round(point.pos + (Math.random() - 0.5) * 2 * amount))),
  }));
}

// Export all filters as a collection
export const filters = {
  smooth,
  savitzkyGolay,
  antiJerk,
  amplify,
  reduce,
  invert,
  offset,
  clamp,
  timeShift,
  removeDuplicates,
  rdpSimplify,
  keyframes,
  speedLimit,
  interpolate,
  noise,
};

export default filters;
