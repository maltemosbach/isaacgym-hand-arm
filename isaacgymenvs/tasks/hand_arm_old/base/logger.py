from typing import Any, Dict


class LoggerMixin:
    log_data = {}  

    def log(self, data: Dict[str, Any]) -> None:
            self.log_data.update(data)
