"""Tests for flashstudio.utils.jobs — background job tracking."""

import os
import time
from flashstudio.utils.jobs import (
    is_pid_alive, format_job_duration,
)


class TestIsPidAlive:
    def test_current_process_alive(self):
        assert is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        assert is_pid_alive(99999999) is False


class TestFormatJobDuration:
    def test_seconds(self):
        result = format_job_duration(time.time() - 30)
        assert result.endswith("s")

    def test_minutes(self):
        result = format_job_duration(time.time() - 300)
        assert result.endswith("m")

    def test_hours(self):
        result = format_job_duration(time.time() - 7200)
        assert result.endswith("h")
