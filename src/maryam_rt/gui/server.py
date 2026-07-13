from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Protocol

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from maryam_rt.gui.monitor import RuntimeMonitorState


class GuiController(Protocol):
    monitor: RuntimeMonitorState

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]: ...

    def state_snapshot(self) -> dict[str, Any]: ...


def _html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Maryam Live Demo</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { --green:#1f3b2d; --ok:#16734b; --warn:#9a6415; --bad:#a83c35; --paper:#fff; --bg:#f5f1e8; }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: var(--bg); color: #1f2521; }
    header { padding: 16px 20px; background: var(--green); color: #f7f4ec; display: flex; justify-content: space-between; align-items: center; }
    .wrap { display: grid; grid-template-columns: minmax(300px, 0.8fr) minmax(520px, 2fr); gap: 16px; padding: 16px; }
    .column { display:grid; gap:16px; align-content:start; }
    .panel { background: var(--paper); border-radius: 14px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
    .images { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #ede8de; border-radius: 10px; }
    .meta { font-size: 13px; line-height: 1.5; }
    .status { display: flex; gap: 10px; flex-wrap: wrap; font-size: 13px; }
    .pill { padding: 6px 10px; border-radius: 999px; background: #e6efe7; }
    .pill.off { background: #f3d9d7; }
    .phase { font-size:14px; font-weight:800; letter-spacing:.06em; padding:8px 12px; border-radius:999px; background:#e6efe7; color:#163d2a; }
    label { display:block; font-size:13px; font-weight:650; margin:10px 0 4px; }
    input, select { width:100%; padding:9px 10px; border:1px solid #b9c0b9; border-radius:8px; background:white; }
    button { border:0; border-radius:9px; padding:10px 13px; font-weight:700; cursor:pointer; background:#dce8df; color:#173728; }
    button.primary { background:var(--green); color:white; }
    button.stop { background:#f0d4d1; color:#70251f; }
    button:disabled { opacity:.38; cursor:not-allowed; }
    .buttons { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .check { list-style:none; padding:7px 0; border-bottom:1px solid #eceeea; }
    .check.ok::before { content:'✓'; color:var(--ok); font-weight:900; margin-right:8px; }
    .check.bad::before { content:'○'; color:var(--bad); font-weight:900; margin-right:8px; }
    .notice { padding:10px 12px; border-radius:9px; background:#f0eee4; margin-top:10px; }
    .error { background:#f7dddd; color:#742a24; }
    ul { margin: 0; padding-left: 18px; max-height: 240px; overflow: auto; }
    li { margin-bottom: 6px; font-size: 13px; }
    h3 { margin:2px 0 10px; }
    @media (max-width: 1000px) { .wrap { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div>
      <div style="font-size: 20px; font-weight: 700;">Maryam Live Demo</div>
      <div id="modeLine" style="font-size: 13px; opacity: 0.85;"></div>
    </div>
    <div id="phasePill" class="phase">LOADING MODELS</div>
  </header>
  <div class="wrap">
    <div class="column">
      <div id="setupPanel" class="panel">
        <h3>1. Demo session</h3>
        <label for="participantInput">Anonymous participant code</label>
        <input id="participantInput" value="unseen-demo" autocomplete="off">
        <div id="participantMode" class="notice meta">New/unseen participant</div>

        <h3 style="margin-top:18px;">2. Equipment</h3>
        <label for="eegSelect">EEG stream</label>
        <select id="eegSelect"></select>
        <label for="markerSelect">Marker stream</label>
        <select id="markerSelect"></select>
        <div class="buttons">
          <button id="refreshButton" onclick="loadStreams()">Refresh devices</button>
          <button id="connectButton" class="primary" onclick="connectEquipment()">Connect and check</button>
          <button id="retryButton" onclick="control('retry')">Retry</button>
          <button id="disconnectButton" onclick="control('disconnect')">Disconnect</button>
        </div>
        <div id="discoveryMessage" class="meta notice">Press Refresh devices after NIC2 and PsychoPy are open.</div>

        <h3 style="margin-top:18px;">3. Readiness</h3>
        <ul id="readinessList" style="padding:0"></ul>
        <div class="buttons">
          <button id="channelButton" onclick="control('confirm-channel-order')">Confirm NIC2 channel order</button>
          <button id="armButton" class="primary" onclick="control('arm')">Arm session</button>
          <button id="disarmButton" onclick="control('disarm')">Disarm</button>
          <button id="stopButton" class="stop" onclick="control('stop')">Stop session</button>
        </div>
        <div id="message" class="notice meta"></div>
      </div>
      <div class="panel">
        <h3>Connection</h3>
        <div class="status">
          <div id="enginePill" class="pill">engine</div>
          <div id="eegPill" class="pill">EEG</div>
          <div id="markerPill" class="pill">markers</div>
          <div id="highPill" class="pill">high-level</div>
        </div>
        <div id="streamMeta" class="meta notice"></div>
        <h3 style="margin-top:16px;">Latest event</h3>
        <div id="latestMeta" class="meta">Waiting for data.</div>
        <h3 style="margin-top:16px;">Recent markers</h3>
        <ul id="markerList"></ul>
      </div>
    </div>
    <div class="column">
      <div class="panel">
        <h3>EEG stream</h3>
        <div id="eegPlot" style="height:430px;"></div>
      </div>
      <div class="panel">
        <h3>Current reconstruction</h3>
        <div class="images">
          <div><div class="meta" style="font-weight:700;">Target</div><img id="targetImage" alt="Target image"></div>
          <div><div class="meta" style="font-weight:700;">Low-level</div><img id="lowImage" alt="Low-level reconstruction"></div>
          <div><div class="meta" style="font-weight:700;">High-level</div><img id="highImage" alt="High-level reconstruction"></div>
        </div>
      </div>
    </div>
  </div>
  <script>
    let streamCache = [];
    let lastState = null;

    async function fetchJson(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }

    async function sendJson(path, method='POST', payload=null) {
      const options = {method, headers:{'Content-Type':'application/json'}};
      if (payload !== null) options.body = JSON.stringify(payload);
      const response = await fetch(path, options);
      if (!response.ok) {
        let detail = await response.text();
        try { detail = JSON.parse(detail).detail || detail; } catch (_) {}
        throw new Error(detail);
      }
      return await response.json();
    }

    function setPill(id, on, label) {
      const el = document.getElementById(id);
      el.textContent = label;
      el.className = on ? 'pill' : 'pill off';
    }

    function optionLabel(stream) {
      const rate = stream.nominal_sampling_rate ? ` · ${stream.nominal_sampling_rate} Hz` : '';
      return `${stream.name} · ${stream.type} · ${stream.channel_count} ch${rate}`;
    }

    function populateStreams(selectId, streams, recommended) {
      const select = document.getElementById(selectId);
      select.innerHTML = '';
      streams.forEach((stream) => {
        const option = document.createElement('option');
        option.value = String(streamCache.indexOf(stream));
        option.textContent = optionLabel(stream);
        if (recommended && stream.name === recommended.name && stream.source_id === recommended.source_id) option.selected = true;
        select.appendChild(option);
      });
      if (!streams.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No matching stream visible';
        select.appendChild(option);
      }
    }

    async function loadStreams() {
      const el = document.getElementById('discoveryMessage');
      el.textContent = 'Searching for LSL streams…';
      try {
        const data = await fetchJson('/api/streams?wait_time=0.75');
        streamCache = data.streams || [];
        let eegStreams = streamCache.filter(s => String(s.type).toLowerCase() === 'eeg');
        let markerStreams = streamCache.filter(s => ['marker','markers'].includes(String(s.type).toLowerCase()));
        if (!eegStreams.length && data.configured && data.configured.eeg && data.configured.eeg.name) {
          streamCache.push(data.configured.eeg); eegStreams = [data.configured.eeg];
        }
        if (!markerStreams.length && data.configured && data.configured.markers && data.configured.markers.name) {
          streamCache.push(data.configured.markers); markerStreams = [data.configured.markers];
        }
        populateStreams('eegSelect', eegStreams, (data.recommended && data.recommended.eeg) || (data.configured && data.configured.eeg));
        populateStreams('markerSelect', markerStreams, (data.recommended && data.recommended.markers) || (data.configured && data.configured.markers));
        el.textContent = data.error || `${streamCache.length} LSL stream(s) found.`;
      } catch (err) {
        el.textContent = String(err);
      }
    }

    function selectedStream(id) {
      const value = document.getElementById(id).value;
      return value === '' ? null : streamCache[Number(value)];
    }

    async function connectEquipment() {
      try {
        await sendJson('/api/session', 'PUT', {participant_code: document.getElementById('participantInput').value});
        const eeg = selectedStream('eegSelect');
        const markers = selectedStream('markerSelect');
        if (!eeg || !markers) throw new Error('Select both EEG and marker streams.');
        await sendJson('/api/control/connect', 'POST', {eeg, markers});
        await refreshState();
      } catch (err) {
        showLocalError(err);
      }
    }

    async function control(action) {
      try {
        await sendJson(`/api/control/${action}`);
        await refreshState();
      } catch (err) {
        showLocalError(err);
      }
    }

    function showLocalError(err) {
      const el = document.getElementById('message');
      el.className = 'notice meta error';
      el.textContent = String(err);
    }

    function setButton(id, enabled) { document.getElementById(id).disabled = !enabled; }

    async function refreshState() {
      const data = await fetchJson('/api/state');
      lastState = data;
      const status = data.status;
      document.getElementById('modeLine').textContent = `mode: ${status.mode} · session: ${status.session_id || '-'}`;
      document.getElementById('phasePill').textContent = String(status.phase || 'unknown').replaceAll('_',' ').toUpperCase();
      const participantInput = document.getElementById('participantInput');
      if (!participantInput.dataset.initialized) {
        participantInput.value = status.participant_code || 'unseen-demo';
        participantInput.dataset.initialized = 'true';
      }
      const message = document.getElementById('message');
      message.className = status.error ? 'notice meta error' : 'notice meta';
      message.textContent = status.error ? `Error: ${status.error}` : status.message;
      document.getElementById('setupPanel').style.display = (data.interactive_setup || status.phase === 'error') ? 'block' : 'none';
      document.getElementById('participantMode').textContent = `${status.participant_mode === 'unseen' ? 'New/unseen participant' : 'Known participant'} · ${status.high_level_reason || 'standard model path'}`;
      setPill('enginePill', status.engine_running, status.engine_running ? 'engine running' : 'engine stopped');
      setPill('eegPill', status.eeg_connected, status.eeg_connected ? 'EEG connected' : 'EEG disconnected');
      setPill('markerPill', status.marker_connected, status.marker_connected ? 'markers connected' : 'markers disconnected');
      setPill('highPill', status.high_level_enabled, status.high_level_enabled ? 'high-level on' : 'high-level off');

      const actions = data.actions || {};
      setButton('connectButton', !!actions.can_connect);
      setButton('retryButton', !!actions.can_retry);
      setButton('channelButton', !!actions.can_confirm_channel_order);
      setButton('armButton', !!actions.can_arm);
      setButton('disarmButton', !!actions.can_disarm);
      setButton('stopButton', !!actions.can_stop);
      setButton('disconnectButton', !!actions.can_disconnect);

      const readiness = document.getElementById('readinessList');
      readiness.innerHTML = '';
      Object.values(data.readiness || {}).forEach((item) => {
        const li = document.createElement('li');
        li.className = `check ${item.ok ? 'ok' : 'bad'}`;
        li.textContent = item.label;
        readiness.appendChild(li);
      });

      document.getElementById('streamMeta').textContent =
        `Output: ${status.output_root || '-'}\n` +
        `EEG: ${status.eeg_channel_count || '-'} channels · ${status.eeg_sampling_rate || '-'} Hz · buffer ${(status.buffer_seconds || 0).toFixed(2)} s\n` +
        `Trials: ${status.trials_received} received · ${status.trials_processed} processed · ${status.trials_dropped} dropped · ` +
        `stream gaps: ${status.eeg_timestamp_gaps} · marker overflow: ${status.marker_events_dropped}\n` +
        `Signal check: ${status.signal_quality_status}` +
        (status.signal_quality_warnings && status.signal_quality_warnings.length ? ` · ${status.signal_quality_warnings.join('; ')}` : '');

      const low = data.latest_low_level;
      const target = data.latest_target;
      const high = data.latest_high_level;
      if (low.path) document.getElementById('lowImage').src = `/api/image/latest-low?v=${data.updated_at}`;
      if (target.path) document.getElementById('targetImage').src = `/api/image/latest-target?v=${data.updated_at}`;
      if (high.path) document.getElementById('highImage').src = `/api/image/latest-high?v=${data.updated_at}`;

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
        const reason = marker.reason ? ` · ${marker.reason}` : '';
        item.textContent = `${marker.label || marker.event_name || marker.raw_value} · ${marker.status || 'seen'}${reason} · ${marker.timestamp.toFixed ? marker.timestamp.toFixed(3) : marker.timestamp}`;
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
      if (!window.Plotly) {
        document.getElementById('eegPlot').textContent = traces.length ? 'EEG is streaming. Plotly is unavailable offline.' : 'Waiting for EEG.';
        return;
      }
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
    loadStreams();
    setInterval(tick, 750);
  </script>
</body>
</html>"""


def create_app(controller: GuiController) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        controller.start()
        try:
            yield
        finally:
            controller.stop()

    app = FastAPI(title="Realtime Monitor", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html()

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return controller.state_snapshot()

    @app.get("/api/streams")
    def streams(wait_time: float = 0.5) -> dict[str, Any]:
        discover = getattr(controller, "discover_streams", None)
        if discover is None:
            return {"streams": [], "recommended": {}, "interactive_setup": False}
        try:
            return discover(wait_time=max(min(wait_time, 3.0), 0.0))
        except Exception as exc:
            return {
                "streams": [],
                "recommended": {},
                "interactive_setup": True,
                "error": str(exc),
            }

    @app.put("/api/session")
    def configure_session(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        configure = getattr(controller, "configure_session", None)
        if configure is None:
            raise HTTPException(status_code=409, detail="Session setup is unavailable in replay mode.")
        try:
            configure(str(payload.get("participant_code") or ""))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return controller.state_snapshot()

    def _control(name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        method = getattr(controller, name, None)
        if method is None:
            raise HTTPException(status_code=409, detail=f"{name} is unavailable in this mode.")
        try:
            if payload is None:
                method()
            else:
                method(payload)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return controller.state_snapshot()

    @app.post("/api/control/connect")
    def connect(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return _control("connect", payload)

    @app.post("/api/control/retry")
    def retry() -> dict[str, Any]:
        return _control("retry")

    @app.post("/api/control/confirm-channel-order")
    def confirm_channel_order() -> dict[str, Any]:
        return _control("confirm_channel_order")

    @app.post("/api/control/arm")
    def arm() -> dict[str, Any]:
        return _control("arm")

    @app.post("/api/control/disarm")
    def disarm() -> dict[str, Any]:
        return _control("disarm")

    @app.post("/api/control/stop")
    def finish() -> dict[str, Any]:
        return _control("finish_session")

    @app.post("/api/control/disconnect")
    def disconnect() -> dict[str, Any]:
        return _control("disconnect")

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

    def state_snapshot(self) -> dict[str, Any]:
        snapshot = self.monitor.snapshot()
        snapshot["interactive_setup"] = False
        snapshot["actions"] = {
            "can_connect": False,
            "can_retry": False,
            "can_confirm_channel_order": False,
            "can_arm": False,
            "can_disarm": False,
            "can_stop": False,
            "can_disconnect": False,
        }
        return snapshot


class LiveController(BaseController):
    def __init__(
        self,
        runner: Any | None,
        monitor: RuntimeMonitorState,
        *,
        runner_factory: Callable[[str, str, str | None, str | None], Any] | None = None,
        auto_connect: bool = True,
        default_eeg_stream_name: str | None = None,
        default_marker_stream_name: str | None = None,
    ) -> None:
        super().__init__(monitor)
        self.runner = runner
        self.runner_factory = runner_factory
        self.auto_connect = auto_connect
        self.default_eeg_stream_name = default_eeg_stream_name
        self.default_marker_stream_name = default_marker_stream_name
        self._last_connection: dict[str, str | None] | None = None
        self._controller_lock = threading.Lock()

    def start(self) -> None:
        self.monitor.set_models_loaded(True)
        self.monitor.set_status(
            engine_running=False,
            phase="waiting_for_equipment",
            message="Configure the participant and connect the equipment.",
        )
        if self.auto_connect:
            self.connect(
                {
                    "eeg": {"name": self.default_eeg_stream_name},
                    "markers": {"name": self.default_marker_stream_name},
                }
            )

    def _start_runner(self) -> None:
        if self.runner is None:
            raise RuntimeError("Live runner has not been configured.")
        self._started = True
        self._thread = threading.Thread(target=self._run, name="RealtimeGuiLiveRunner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.monitor.set_status(engine_running=True, message="Starting live runner.")
        try:
            self.runner.run()
        except Exception as exc:
            self.monitor.set_status(engine_running=False)
            self.monitor.fail(str(exc))
        else:
            if self.monitor.status.phase not in {"finished", "stopped"}:
                self.monitor.set_status(engine_running=False, message="Runner stopped.")
        finally:
            self._started = False

    def stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._started = False
        self.monitor.set_status(engine_running=False, phase="stopped", message="Live controller stopped.")

    def _stop_current_runner(self) -> None:
        if self.runner is not None:
            self.runner.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._started = False

    @staticmethod
    def _stream_selection(payload: dict[str, Any], key: str) -> tuple[str, str | None]:
        selection = payload.get(key) or {}
        name = str(selection.get("name") or "").strip()
        source_id = str(selection.get("source_id") or "").strip() or None
        if not name:
            raise ValueError(f"Select a {key} stream before connecting.")
        return name, source_id

    def connect(self, payload: dict[str, Any]) -> None:
        eeg_name, eeg_source_id = self._stream_selection(payload, "eeg")
        marker_name, marker_source_id = self._stream_selection(payload, "markers")
        with self._controller_lock:
            self._stop_current_runner()
            if self.runner_factory is not None:
                self.runner = self.runner_factory(
                    eeg_name,
                    marker_name,
                    eeg_source_id,
                    marker_source_id,
                )
                self.runner.set_participant_code(self.monitor.status.participant_code or "unseen-demo")
            elif self.runner is None:
                raise RuntimeError("Live runner factory is unavailable.")
            self._last_connection = {
                "eeg_name": eeg_name,
                "eeg_source_id": eeg_source_id,
                "marker_name": marker_name,
                "marker_source_id": marker_source_id,
            }
            self.monitor.disarm("Connecting to EEG and marker streams.")
            self._start_runner()

    def retry(self) -> None:
        if self._last_connection is None:
            raise RuntimeError("Connect equipment once before using Retry.")
        self.connect(
            {
                "eeg": {
                    "name": self._last_connection["eeg_name"],
                    "source_id": self._last_connection["eeg_source_id"],
                },
                "markers": {
                    "name": self._last_connection["marker_name"],
                    "source_id": self._last_connection["marker_source_id"],
                },
            }
        )

    def configure_session(self, participant_code: str) -> None:
        self.monitor.configure_session(participant_code)
        if self.runner is not None:
            self.runner.set_participant_code(participant_code)
        self.monitor.set_waiting_or_ready()

    def confirm_channel_order(self) -> None:
        self.monitor.confirm_channel_order(True)
        self.monitor.set_waiting_or_ready()

    def arm(self) -> None:
        if self.runner is None or not self._started:
            raise RuntimeError("Connect the equipment before arming the session.")
        armed, message = self.runner.arm()
        if not armed:
            raise RuntimeError(message)

    def disarm(self) -> None:
        if self.runner is not None:
            self.runner.disarm()

    def finish_session(self) -> None:
        if self.runner is None:
            raise RuntimeError("No live session is active.")
        self.runner.finish_session("Experiment stopped by the operator.")

    def disconnect(self) -> None:
        with self._controller_lock:
            self._stop_current_runner()
            self.runner = None if self.runner_factory is not None else self.runner
            self.monitor.disarm("Equipment disconnected. Select streams and connect again.")

    def discover_streams(self, wait_time: float = 0.5) -> dict[str, Any]:
        from maryam_rt.realtime.streaming.discovery import discover_lsl_streams, recommend_streams

        streams = discover_lsl_streams(wait_time=wait_time)
        status = self.monitor.status
        recommended = recommend_streams(
            streams,
            expected_channels=status.expected_eeg_channel_count or 32,
            expected_sampling_rate=status.expected_eeg_sampling_rate or 500.0,
            preferred_eeg_name=self.default_eeg_stream_name,
            preferred_marker_name=self.default_marker_stream_name,
        )
        return {
            "streams": streams,
            "recommended": recommended,
            "configured": {
                "eeg": {"name": self.default_eeg_stream_name, "type": "EEG", "source_id": "", "channel_count": status.expected_eeg_channel_count or 32, "nominal_sampling_rate": status.expected_eeg_sampling_rate or 500.0},
                "markers": {"name": self.default_marker_stream_name, "type": "Markers", "source_id": "", "channel_count": 1, "nominal_sampling_rate": 0.0},
            },
            "interactive_setup": True,
        }

    def state_snapshot(self) -> dict[str, Any]:
        snapshot = self.monitor.snapshot()
        status = snapshot["status"]
        snapshot["interactive_setup"] = True
        snapshot["actions"] = {
            "can_connect": status["phase"] not in {"armed", "running", "finished", "stopped"},
            "can_retry": self._last_connection is not None
            and status["phase"] not in {"armed", "running", "finished", "stopped"},
            "can_confirm_channel_order": status["eeg_connected"]
            and status["channel_order_confirmation_required"]
            and not status["channel_order_confirmed"],
            "can_arm": status["ready"] and status["phase"] == "ready",
            "can_disarm": status["phase"] == "armed",
            "can_stop": status["phase"] in {"armed", "running"},
            "can_disconnect": self.runner is not None and status["phase"] not in {"armed", "running"},
        }
        return snapshot

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]:
        if self.runner is None:
            return {"sampling_rate": None, "window_seconds": None, "traces": [], "markers": []}
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


class ErrorController(BaseController):
    """Keep the browser available when model or startup initialization fails."""

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def eeg_plot_data(self, seconds: float = 6.0, max_channels: int = 8) -> dict[str, Any]:
        return {"sampling_rate": None, "window_seconds": None, "traces": [], "markers": []}


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
