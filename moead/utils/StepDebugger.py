import json
from datetime import datetime
from pathlib import Path


class StepDebugger:
    """
    Registrar paso a paso en JSON con un pequeño estado de máquina.

    Cada paso puede tener los eventos:
    - start: inició la etapa
    - passed: se completó correctamente
    - failed: falló en esta etapa
    - warning: hubo una advertencia durante la etapa
    """

    def __init__(self, filename: str | Path = "debug_trace.json", run_id: str | None = None, overwrite: bool = True):
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_step = None
        self.data = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "current_state": None,
            "steps": [],
            "summary": {}
        }
        if overwrite or not self.filename.exists():
            self._save()
        else:
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                self.data.update(loaded)
            except Exception:
                self._save()

    def _save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _record(self, step: str, event_type: str, status: str, message: str | None = None, context: dict | None = None, error: str | None = None):
        now = datetime.now().isoformat()
        entry = {
            "step": step,
            "event_type": event_type,
            "status": status,
            "message": message,
            "timestamp": now,
            "context": context or {},
        }
        if error is not None:
            entry["error"] = error
        self.data["steps"].append(entry)
        self.data["current_state"] = step if event_type == "start" else status
        self._save()

    def start_step(self, step: str, context: dict | None = None):
        if self.current_step is not None and self.current_step != step:
            self._record(self.current_step, "auto_finish", "warning", message="Paso anterior no cerrado antes de iniciar uno nuevo.")
        self.current_step = step
        self._record(step, "start", "started", context=context)

    def pass_step(self, step: str, message: str | None = None, context: dict | None = None):
        self._record(step, "pass", "passed", message=message, context=context)
        self.current_step = None

    def fail_step(self, step: str, error: Exception | str, message: str | None = None, context: dict | None = None):
        err = str(error)
        self._record(step, "fail", "failed", message=message, context=context, error=err)
        self.current_step = None

    def warning_step(self, step: str, message: str, context: dict | None = None):
        self._record(step, "warning", "warning", message=message, context=context)

    def record_event(self, step: str, status: str, message: str | None = None, context: dict | None = None):
        self._record(step, "event", status, message=message, context=context)

    def set_summary(self, summary: dict):
        self.data["summary"].update(summary)
        self._save()
