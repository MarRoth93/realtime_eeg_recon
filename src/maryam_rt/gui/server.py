from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from maryam_rt.gui.monitor import RuntimeMonitorState


class GuiController(Protocol):
    monitor: RuntimeMonitorState

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]: ...


def _html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Realtime Monitor</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #f5f1e8; color: #1f2521; }
    header { padding: 16px 20px; background: #1f3b2d; color: #f7f4ec; display: flex; justify-content: space-between; align-items: center; }
    .wrap { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; }
    .panel { background: white; border-radius: 14px; padding: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
    .side { display: grid; grid-template-rows: auto auto auto; gap: 16px; }
    .images { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #ede8de; border-radius: 10px; }
    .meta { font-size: 13px; line-height: 1.5; }
    .status { display: flex; gap: 10px; flex-wrap: wrap; font-size: 13px; }
    .pill { padding: 6px 10px; border-radius: 999px; background: #e6efe7; }
    .pill.off { background: #f3d9d7; }
    ul { margin: 0; padding-left: 18px; max-height: 240px; overflow: auto; }
    li { margin-bottom: 6px; font-size: 13px; }
  </style>
</head>
<body>
  <header>
    <div>
      <div style="font-size: 20px; font-weight: 700;">Realtime Monitor</div>
      <div id="modeLine" style="font-size: 13px; opacity: 0.85;"></div>
    </div>
    <div class="status">
      <div id="enginePill" class="pill">engine</div>
      <div id="eegPill" class="pill">EEG</div>
      <div id="markerPill" class="pill">markers</div>
      <div id="highPill" class="pill">high-level</div>
    </div>
  </header>
  <div class="wrap">
    <div class="panel">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3 style="margin:6px 0 10px;">EEG Stream</h3>
        <div id="message" class="meta"></div>
      </div>
      <div id="eegPlot" style="height:560px;"></div>
    </div>
    <div class="side">
      <div class="panel">
        <h3 style="margin:6px 0 10px;">Target vs Low-Level</h3>
        <div class="images">
          <div>
            <div class="meta" style="margin-bottom:6px; font-weight:600;">Target</div>
            <img id="targetImage" alt="Target image">
          </div>
          <div>
            <div class="meta" style="margin-bottom:6px; font-weight:600;">Low-level</div>
            <img id="lowImage" alt="Low-level reconstruction">
          </div>
        </div>
      </div>
      <div class="panel">
        <h3 style="margin:6px 0 10px;">Latest Event</h3>
        <div id="latestMeta" class="meta">Waiting for data.</div>
      </div>
      <div class="panel">
        <h3 style="margin:6px 0 10px;">Recent Markers</h3>
        <ul id="markerList"></ul>
      </div>
    </div>
  </div>
  <script>
    async function fetchJson(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }

    function setPill(id, on, label) {
      const el = document.getElementById(id);
      el.textContent = label;
      el.className = on ? 'pill' : 'pill off';
    }

    async function refreshState() {
      const data = await fetchJson('/api/state');
      const status = data.status;
      document.getElementById('modeLine').textContent = `mode: ${status.mode}`;
      document.getElementById('message').textContent = status.error ? `error: ${status.error}` : status.message;
      setPill('enginePill', status.engine_running, status.engine_running ? 'engine running' : 'engine stopped');
      setPill('eegPill', status.eeg_connected, status.eeg_connected ? 'EEG connected' : 'EEG disconnected');
      setPill('markerPill', status.marker_connected, status.marker_connected ? 'markers connected' : 'markers disconnected');
      setPill('highPill', status.high_level_enabled, status.high_level_enabled ? 'high-level on' : 'high-level off');

      const low = data.latest_low_level;
      const target = data.latest_target;
      if (low.path) document.getElementById('lowImage').src = `/api/image/latest-low?v=${data.updated_at}`;
      if (target.path) document.getElementById('targetImage').src = `/api/image/latest-target?v=${data.updated_at}`;

      const latestInfo = [];
      if (low.info) {
        latestInfo.push(`marker: ${low.info.marker_label || low.info.event_code || '-'}`);
        latestInfo.push(`time: ${low.info.event_time || low.info.marker_lsl_timestamp || '-'}`);
      }
      if (target.info && target.info.label) latestInfo.push(`target: ${target.info.label}`);
      document.getElementById('latestMeta').innerHTML = latestInfo.join('<br>') || 'Waiting for data.';

      const markerList = document.getElementById('markerList');
      markerList.innerHTML = '';
      (data.recent_markers || []).slice().reverse().slice(0, 12).forEach((marker) => {
        const item = document.createElement('li');
        item.textContent = `${marker.label || marker.event_name || marker.raw_value} · ${marker.status || 'seen'} · ${marker.timestamp.toFixed ? marker.timestamp.toFixed(3) : marker.timestamp}`;
        markerList.appendChild(item);
      });
    }

    async function refreshPlot() {
      const data = await fetchJson('/api/eeg');
      const traces = (data.traces || []).map((trace) => ({
        x: trace.x,
        y: trace.y,
        type: 'scattergl',
        mode: 'lines',
        name: trace.name,
        line: { width: 1.25 }
      }));
      const shapes = (data.markers || []).map((marker) => ({
        type: 'line',
        x0: marker.x, x1: marker.x,
        y0: 0, y1: 1,
        yref: 'paper',
        line: { color: marker.status === 'rejected' ? '#c0392b' : '#156f4a', width: 2, dash: 'dot' }
      }));
      Plotly.react('eegPlot', traces, {
        margin: { t: 10, r: 10, b: 40, l: 50 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: { title: 'Seconds', zeroline: false },
        yaxis: { showticklabels: false, zeroline: false },
        shapes
      }, {responsive: true, displayModeBar: false});
    }

    async function tick() {
      try {
        await refreshState();
        await refreshPlot();
      } catch (err) {
        document.getElementById('message').textContent = String(err);
      }
    }
    tick();
    setInterval(tick, 750);
  </script>
</body>
</html>"""


def create_app(controller: GuiController) -> FastAPI:
    app = FastAPI(title="Realtime Monitor")

    @app.on_event("startup")
    def _startup() -> None:
        controller.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        controller.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html()

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return controller.monitor.snapshot()

    @app.get("/api/eeg")
    def eeg(seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]:
        return controller.eeg_plot_data(seconds=seconds, max_channels=max_channels)

    def _image_response(which: str) -> FileResponse:
        snapshot = controller.monitor.snapshot()
        record = snapshot.get(which, {})
        path = record.get("path")
        if not path:
            raise HTTPException(status_code=404, detail=f"No image available for {which}.")
        image_path = Path(path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=str(image_path))
        return FileResponse(image_path)

    @app.get("/api/image/latest-low")
    def latest_low() -> FileResponse:
        return _image_response("latest_low_level")

    @app.get("/api/image/latest-target")
    def latest_target() -> FileResponse:
        return _image_response("latest_target")

    @app.get("/api/image/latest-high")
    def latest_high() -> FileResponse:
        return _image_response("latest_high_level")

    return app


class BaseController:
    def __init__(self, monitor: RuntimeMonitorState) -> None:
        self.monitor = monitor
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]:
        raise NotImplementedError


class LiveController(BaseController):
    def __init__(self, runner: Any, monitor: RuntimeMonitorState) -> None:
        super().__init__(monitor)
        self.runner = runner

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, name="RealtimeGuiLiveRunner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.monitor.set_status(engine_running=True, message="Starting live runner.")
        try:
            self.runner.run()
        except Exception as exc:
            self.monitor.set_status(engine_running=False, error=str(exc), message="Runner stopped with error.")
        else:
            self.monitor.set_status(engine_running=False, message="Runner stopped.")

    def stop(self) -> None:
        self.runner.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.monitor.set_status(engine_running=False, message="Live controller stopped.")

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]:
        inlet = self.runner.eeg_inlet
        sampling_rate = inlet.sampling_rate
        if sampling_rate is None or inlet.available_samples <= 0:
            return {"sampling_rate": None, "window_seconds": None, "traces": [], "markers": []}

        n_samples = min(inlet.available_samples, max(int(round(seconds * sampling_rate)), 1))
        window = inlet.get_window(n_samples, timeout=0.0)
        data = window[:max_channels]
        times = np.linspace(-n_samples / sampling_rate, 0.0, n_samples, endpoint=False, dtype=np.float32)
        scale = max(float(np.nanmax(np.abs(data))) * 2.5, 1.0)
        traces = []
        for idx, channel in enumerate(data):
            traces.append(
                {
                    "name": f"Ch {idx + 1}",
                    "x": times.tolist(),
                    "y": (channel + idx * scale).astype(np.float32).tolist(),
                }
            )

        snapshot = self.monitor.snapshot()
        last_lsl = inlet.last_lsl_timestamp
        markers = []
        if last_lsl is not None:
            for marker in snapshot["recent_markers"]:
                marker_x = float(marker["timestamp"]) - float(last_lsl)
                if marker_x >= times[0]:
                    markers.append({"x": marker_x, "label": marker["label"], "status": marker["status"]})

        return {
            "sampling_rate": sampling_rate,
            "window_seconds": n_samples / sampling_rate,
            "traces": traces,
            "markers": markers,
        }


class ReplayController(BaseController):
    def __init__(self, runner: Any, monitor: RuntimeMonitorState) -> None:
        super().__init__(monitor)
        self.runner = runner

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, name="RealtimeGuiReplayRunner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.monitor.set_status(engine_running=True, message="Starting replay runner.")
        try:
            self.runner.run()
        except Exception as exc:
            self.monitor.set_status(engine_running=False, error=str(exc), message="Replay stopped with error.")
        else:
            self.monitor.set_status(engine_running=False, message="Replay complete.")

    def stop(self) -> None:
        self.runner.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.monitor.set_status(engine_running=False, message="Replay controller stopped.")

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]:
        return self.runner.eeg_plot_data(seconds=seconds, max_channels=max_channels)
