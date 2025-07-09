"""
Unit tests for data loading functionality.
"""
import pytest
import json
import os
from unittest.mock import patch, mock_open, MagicMock
from src.carmina.data_sources.data_loader import load_dataset
from src.carmina.data_sources.exceptions import DataLoadError


@pytest.mark.unit
class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_dataset_nonexistent_path(self):
        """Test loading from non-existent path raises error."""
        with pytest.raises(DataLoadError, match="Input path does not exist"):
            load_dataset("/nonexistent/path")
    
    def test_load_dataset_unsupported_format(self, tmp_path):
        """Test loading unsupported file format raises error."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test content")
        
        with pytest.raises(DataLoadError, match="Unsupported file format"):
            load_dataset(str(unsupported_file))
    
    @patch('src.carmina.data_sources.data_loader.load_json_file')
    def test_load_dataset_json_file(self, mock_load_json):
        """Test loading JSON file."""
        mock_data = [{"id": "1", "text": "test"}]
        mock_load_json.return_value = mock_data
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            result = load_dataset("test.json")
            
        mock_load_json.assert_called_once_with("test.json")
        assert result == mock_data
    
    @patch('src.carmina.data_sources.data_loader.load_csv_file')
    def test_load_dataset_csv_file(self, mock_load_csv):
        """Test loading CSV file."""
        mock_data = [{"id": "1", "text": "test"}]
        mock_load_csv.return_value = mock_data
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            result = load_dataset("test.csv")
            
        mock_load_csv.assert_called_once_with("test.csv")
        assert result == mock_data
    
    @patch('src.carmina.data_sources.data_loader.load_txt_file')
    def test_load_dataset_txt_file(self, mock_load_txt):
        """Test loading TXT file."""
        mock_data = [{"id": "1", "text": "test"}]
        mock_load_txt.return_value = mock_data
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            result = load_dataset("test.txt")
            
        mock_load_txt.assert_called_once_with("test.txt")
        assert result == mock_data
    
    @patch('src.carmina.data_sources.data_loader.load_directory')
    def test_load_dataset_directory(self, mock_load_directory):
        """Test loading directory."""
        mock_data = [{"id": "1", "text": "test"}]
        mock_load_directory.return_value = mock_data
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True):
            result = load_dataset("test_dir")
            
        mock_load_directory.assert_called_once_with("test_dir")
        assert result == mock_data