import pytest
import subprocess
import tempfile
from pathlib import Path

class TestCLI:
    def test_cli_help(self):
        """Test that CLI help command works"""
        result = subprocess.run(
            ["python", "-m", "minerva.cli", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()
        
    def test_ingest_command(self, tmp_path):
        """Test ingest command with temporary file"""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for Minerva ingestion.")
        
        # Run ingest command
        result = subprocess.run(
            ["python", "-m", "minerva.cli", "ingest", str(test_file)],
            capture_output=True,
            text=True
        )
        
        # Should complete without errors
        assert result.returncode == 0
        assert "ingested" in result.stdout.lower() or "error" not in result.stderr.lower()
