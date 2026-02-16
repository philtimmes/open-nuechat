/**
 * GpuTab - GPU VRAM monitoring + model→GPU assignment for Admin panel.
 * Uses rocm-smi data, shows HIP↔ROCm device mapping, persists assignments.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import api from '../../lib/api';

interface GpuInfo {
  rocm_id: number;
  name: string;
  total_gb: number;
  used_gb: number;
  free_gb: number;
  vram_pct: number;
  mem_activity_pct: number;
  avg_bandwidth: number;
  pci_bus?: string;
}

interface GpuStatus {
  available: boolean;
  count: number;
  rocm: boolean;
  rocm_version?: string;
  gpus: GpuInfo[];
  device_map: { hip_to_rocm: Record<string, number>; rocm_to_hip: Record<string, number>; gpu_details?: Record<string, any> };
  assignments: Record<string, { rocm_id: number; label: string } | number>;
  model_keys: string[];
  model_labels: Record<string, string>;
  error?: string;
}

function barColor(pct: number): string {
  if (pct < 50) return '#22c55e';
  if (pct < 75) return '#eab308';
  if (pct < 90) return '#f97316';
  return '#ef4444';
}

// ─── GPU Card ───────────────────────────────────────────────────────────────

const GpuCard: React.FC<{ gpu: GpuInfo; hipId?: number; assignedModels: string[] }> = ({ gpu, hipId, assignedModels }) => {
  const pct = gpu.vram_pct;
  const color = barColor(pct);

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center justify-center w-7 h-7 rounded-md text-xs font-bold"
                style={{ background: `${color}18`, color }}>
            {gpu.rocm_id}
          </span>
          <div>
            <div className="text-xs font-medium text-[var(--color-text)] truncate max-w-[160px]">{gpu.name}</div>
            <div className="text-[9px] text-[var(--color-text-secondary)]">
              ROCm {gpu.rocm_id}{hipId !== undefined && hipId !== gpu.rocm_id ? ` · HIP ${hipId}` : ''}
              {gpu.pci_bus ? ` · ${gpu.pci_bus}` : ''}
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-base font-bold tabular-nums" style={{ color }}>{pct}%</div>
        </div>
      </div>

      {/* VRAM bar */}
      <div className="relative h-5 rounded overflow-hidden mb-2" style={{ background: 'var(--color-background)' }}>
        <div className="absolute inset-y-0 left-0 rounded transition-all duration-500"
             style={{ width: `${Math.min(pct, 100)}%`, background: color, opacity: 0.85 }} />
        <div className="absolute inset-0 flex items-center justify-center text-[10px] font-mono font-bold"
             style={{ color: pct > 40 ? '#fff' : 'var(--color-text)', textShadow: pct > 40 ? '0 1px 2px rgba(0,0,0,0.5)' : 'none' }}>
          {gpu.total_gb > 0 ? `${gpu.used_gb.toFixed(1)} / ${gpu.total_gb.toFixed(1)} GB` : `${pct}% used`}
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-2 text-[10px]">
        <span className="text-[var(--color-text-secondary)]">Free: <span className="font-mono">{gpu.free_gb.toFixed(1)}G</span></span>
        {gpu.mem_activity_pct > 0 && (
          <span className="text-[var(--color-text-secondary)]">Activity: <span className="font-mono">{gpu.mem_activity_pct}%</span></span>
        )}
      </div>

      {/* Assigned models */}
      {assignedModels.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {assignedModels.map(m => (
            <span key={m} className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-blue-500/15 text-blue-400">{m}</span>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Model Assignment Row ───────────────────────────────────────────────────

const AssignmentRow: React.FC<{
  modelKey: string;
  label: string;
  currentRocmId: number | null;
  gpus: GpuInfo[];
  rocmToHip: Record<string, number>;
  onAssign: (modelKey: string, rocmId: number | null) => void;
  saving: boolean;
}> = ({ modelKey, label, currentRocmId, gpus, rocmToHip, onAssign, saving }) => {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-[var(--color-border)] last:border-0">
      <div>
        <div className="text-sm font-medium text-[var(--color-text)]">{label}</div>
        <div className="text-[10px] text-[var(--color-text-secondary)] font-mono">{modelKey}</div>
      </div>
      <select
        value={currentRocmId ?? 'none'}
        onChange={e => {
          const val = e.target.value;
          onAssign(modelKey, val === 'none' ? null : parseInt(val));
        }}
        disabled={saving}
        className="px-3 py-1.5 rounded-md text-sm bg-[var(--color-background)] border border-[var(--color-border)] text-[var(--color-text)] cursor-pointer min-w-[260px]"
      >
        <option value="none">Auto / CPU</option>
        {gpus.map(g => {
          const hipId = rocmToHip[String(g.rocm_id)];
          const hipLabel = hipId !== undefined && hipId !== g.rocm_id ? ` (HIP ${hipId})` : '';
          return (
            <option key={g.rocm_id} value={g.rocm_id}>
              GPU {g.rocm_id}{hipLabel} — {g.vram_pct}% used, {g.free_gb.toFixed(1)}G free
            </option>
          );
        })}
      </select>
    </div>
  );
};

// ─── Main Component ─────────────────────────────────────────────────────────

export default function GpuTab() {
  const [status, setStatus] = useState<GpuStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval>>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await api.get('/admin/gpu/status');
      setStatus(res.data);
      setError(null);
      setLastUpdate(new Date());
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchStatus, 5000);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, fetchStatus]);

  const handleAssign = useCallback(async (modelKey: string, rocmId: number | null) => {
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await api.post('/admin/gpu/assign', { model_key: modelKey, rocm_id: rocmId });
      setSaveMsg(res.data.message);
      setStatus(prev => {
        if (!prev) return prev;
        const newAssignments = { ...prev.assignments };
        if (rocmId !== null) {
          newAssignments[modelKey] = { rocm_id: rocmId, label: `GPU ${rocmId}` };
        } else {
          delete newAssignments[modelKey];
        }
        return { ...prev, assignments: newAssignments };
      });
      setTimeout(() => setSaveMsg(null), 4000);
    } catch (e: any) {
      setSaveMsg(`Error: ${e.response?.data?.detail || e.message}`);
    } finally {
      setSaving(false);
    }
  }, []);

  if (loading) {
    return <div className="text-center py-12 text-[var(--color-text-secondary)]">Loading GPU status...</div>;
  }

  if (error) {
    return (
      <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
        Failed to fetch GPU status: {error}
        <button onClick={fetchStatus} className="ml-3 underline">Retry</button>
      </div>
    );
  }

  if (!status || !status.available) {
    return (
      <div className="p-6 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] text-center">
        <div className="text-4xl mb-3">🖥️</div>
        <div className="text-[var(--color-text)] font-medium">No GPUs Detected</div>
        <div className="text-sm text-[var(--color-text-secondary)] mt-1">
          {status?.error || 'No CUDA/ROCm GPUs available. rocm-smi not found.'}
        </div>
      </div>
    );
  }

  // Build per-GPU assignment labels
  const gpuAssignments: Record<number, string[]> = {};
  const modelLabels = status.model_labels || {};
  for (const [key, val] of Object.entries(status.assignments)) {
    const rid = typeof val === 'object' && val !== null ? (val as any).rocm_id : val;
    if (rid == null) continue;
    if (!gpuAssignments[rid]) gpuAssignments[rid] = [];
    gpuAssignments[rid].push(modelLabels[key] || key);
  }

  const rocmToHip = status.device_map?.rocm_to_hip || {};
  const totalVram = status.gpus.reduce((s, g) => s + g.total_gb, 0);
  const totalUsed = status.gpus.reduce((s, g) => s + g.used_gb, 0);
  const avgPct = totalVram > 0
    ? (totalUsed / totalVram) * 100
    : status.gpus.length > 0
      ? status.gpus.reduce((s, g) => s + g.vram_pct, 0) / status.gpus.length
      : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-lg font-semibold text-[var(--color-text)]">GPU Status</h2>
          <div className="text-xs text-[var(--color-text-secondary)] mt-0.5">
            {status.count} GPU{status.count !== 1 ? 's' : ''}
            {status.rocm ? ` · ROCm ${status.rocm_version || ''}` : ''}
            {' · via rocm-smi'}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdate && (
            <span className="text-[10px] text-[var(--color-text-secondary)]">{lastUpdate.toLocaleTimeString()}</span>
          )}
          <label className="flex items-center gap-1.5 text-xs text-[var(--color-text-secondary)] cursor-pointer">
            <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} className="rounded" />
            Auto 5s
          </label>
          <button onClick={fetchStatus}
                  className="px-3 py-1.5 text-xs rounded-md bg-[var(--color-button)] text-[var(--color-button-text)] hover:opacity-90">
            Refresh
          </button>
        </div>
      </div>

      {/* Summary bar */}
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
        <div className="flex items-center justify-between mb-1.5 text-sm">
          <span className="font-medium text-[var(--color-text)]">Total VRAM</span>
          <span className="font-mono" style={{ color: barColor(avgPct) }}>
            {totalVram > 0 ? `${totalUsed.toFixed(1)} / ${totalVram.toFixed(1)} GB` : `avg ${avgPct.toFixed(0)}%`}
          </span>
        </div>
        <div className="relative h-3 rounded overflow-hidden" style={{ background: 'var(--color-background)' }}>
          {(() => {
            let offset = 0;
            const slotSize = totalVram > 0 ? totalVram : status.gpus.length;
            return status.gpus.map(gpu => {
              const slotW = totalVram > 0 ? (gpu.total_gb / slotSize) * 100 : (100 / status.gpus.length);
              const usedW = totalVram > 0 ? (gpu.used_gb / slotSize) * 100 : (gpu.vram_pct / status.gpus.length);
              const seg = (
                <div key={gpu.rocm_id}
                     className="absolute inset-y-0 transition-all duration-500"
                     style={{ left: `${offset}%`, width: `${usedW}%`, background: barColor(gpu.vram_pct), opacity: 0.85 }}
                     title={`GPU ${gpu.rocm_id}: ${gpu.vram_pct}% used`} />
              );
              offset += slotW;
              return seg;
            });
          })()}
        </div>
      </div>

      {/* GPU Cards Grid */}
      <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        {status.gpus.map(gpu => {
          const hipId = rocmToHip[String(gpu.rocm_id)];
          return (
            <GpuCard
              key={gpu.rocm_id}
              gpu={gpu}
              hipId={hipId !== undefined ? Number(hipId) : undefined}
              assignedModels={gpuAssignments[gpu.rocm_id] || []}
            />
          );
        })}
      </div>

      {/* Device Map — only show if HIP != ROCm for any device */}
      {Object.keys(rocmToHip).length > 0 && Object.entries(rocmToHip).some(([k, v]) => String(k) !== String(v)) && (
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <h3 className="text-sm font-semibold text-[var(--color-text)] mb-2">HIP ↔ ROCm Device Map</h3>
          <div className="text-[10px] text-[var(--color-text-secondary)] mb-2">
            HIP_VISIBLE_DEVICES remaps ROCm device IDs to process-local cuda indices. May change between reboots.
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(rocmToHip).map(([rocm, hip]) => (
              <span key={rocm} className="px-2 py-1 rounded text-[10px] font-mono bg-[var(--color-background)] text-[var(--color-text)]">
                ROCm {rocm} → HIP {String(hip)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Model → GPU Assignments */}
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <h3 className="text-sm font-semibold text-[var(--color-text)] mb-1">Model → GPU Assignment</h3>
        <div className="text-[10px] text-[var(--color-text-secondary)] mb-3">
          Assign models to specific GPUs. Sets HIP_VISIBLE_DEVICES before model load. Changes take effect on next restart.
          Saved to <span className="font-mono">/app/data/gpuSelector/</span>.
        </div>

        {saveMsg && (
          <div className={`mb-3 p-2 rounded text-xs ${saveMsg.startsWith('Error') ? 'bg-red-500/10 text-red-400' : 'bg-green-500/10 text-green-400'}`}>
            {saveMsg}
          </div>
        )}

        <div>
          {(status.model_keys || []).map(key => {
            const assignment = status.assignments[key];
            const currentId = assignment
              ? (typeof assignment === 'object' && assignment !== null ? (assignment as any).rocm_id : assignment)
              : null;
            return (
              <AssignmentRow
                key={key}
                modelKey={key}
                label={modelLabels[key] || key}
                currentRocmId={currentId as number | null}
                gpus={status.gpus}
                rocmToHip={rocmToHip}
                onAssign={handleAssign}
                saving={saving}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
