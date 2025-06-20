"""
Test visualizer functionality including buttons, display modes, and interactions.

This test exercises the visualizer without requiring a display by mocking pygame.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock pygame and matplotlib before importing visualizer
from unittest.mock import Mock, MagicMock, patch
sys.modules['pygame'] = MagicMock()
sys.modules['pygame.font'] = MagicMock()
sys.modules['pygame.display'] = MagicMock()

sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.backends.backend_agg'] = MagicMock()
sys.modules['matplotlib.backends'] = MagicMock()

from visualizer import GeologyVisualizer
from geo_game import GeoGame
from materials import MaterialType


class TestVisualizerFunctionality:
    """Test visualizer functionality without requiring display."""
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_visualizer_initialization(self, mock_font, mock_display):
        """Test that visualizer initializes correctly."""
        # Setup mocks
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        # Create visualizer
        viz = GeologyVisualizer(window_width=800, window_height=600)
        
        # Check basic attributes
        assert viz.sim_width > 0
        assert viz.sim_height > 0
        assert viz.paused == True  # Should start paused
        assert viz.display_mode == 'materials'
        assert viz.mouse_tool == 'heat'
        assert hasattr(viz, 'simulation')
        assert isinstance(viz.simulation, GeoGame)
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_display_modes(self, mock_font, mock_display):
        """Test switching between display modes."""
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        viz = GeologyVisualizer()
        
        # Test all display modes
        modes = ['materials', 'temperature', 'pressure', 'power', 'velocity', 'gravity', 'potential']
        
        for mode in modes:
            viz.display_mode = mode
            assert viz.display_mode == mode
            
            # Should be able to get display colors without error
            colors = viz._get_display_colors()
            assert colors is not None
            assert colors.shape == (16, 16, 3)
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_tool_switching(self, mock_font, mock_display):
        """Test switching between tools."""
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        viz = GeologyVisualizer()
        
        # Test all tools
        tools = ['heat', 'pressure', 'delete', 'add']
        
        for tool in tools:
            viz.mouse_tool = tool
            assert viz.mouse_tool == tool
            
        # Test tool parameters
        viz.tool_radius = 5
        assert viz.tool_radius == 5
        
        viz.tool_intensity = 200
        assert viz.tool_intensity == 200
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_sidebar_tabs(self, mock_font, mock_display):
        """Test sidebar tab switching."""
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        viz = GeologyVisualizer()
        
        # Test all tabs
        tabs = ['controls', 'stats', 'composition', 'graphs', 'info']
        
        for tab in tabs:
            viz.sidebar_tab = tab
            assert viz.sidebar_tab == tab
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_simulation_speed_control(self, mock_font, mock_display):
        """Test simulation speed multiplier."""
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        viz = GeologyVisualizer()
        
        # Test speed settings
        speeds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        
        for speed in speeds:
            viz.speed_multiplier = speed
            assert viz.speed_multiplier == speed
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_status_bar_format(self, mock_font, mock_display):
        """Test status bar has correct format."""
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        # Create mock screen and font
        mock_screen = MagicMock()
        mock_display.return_value = mock_screen
        mock_font_instance = MagicMock()
        mock_font.return_value = mock_font_instance
        
        # Mock render to capture text
        rendered_text = None
        def capture_render(text, *args):
            nonlocal rendered_text
            rendered_text = text
            return MagicMock()
        
        mock_font_instance.render = capture_render
        
        viz = GeologyVisualizer()
        viz.screen = mock_screen
        
        # Draw status bar
        viz._draw_status_bar()
        
        # Check format contains expected elements
        assert rendered_text is not None
        assert "PAUSED" in rendered_text or "PLAYING" in rendered_text
        assert "Step:" in rendered_text
        assert "t:" in rendered_text
        assert "Tool:" in rendered_text
        
        # Should NOT contain temperature or pressure
        assert "Temp:" not in rendered_text
        assert "Pressure:" not in rendered_text
        assert "°C" not in rendered_text
        assert "MPa" not in rendered_text
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_button_creation(self, mock_font, mock_display):
        """Test that buttons are created correctly."""
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        viz = GeologyVisualizer()
        
        # Check buttons exist
        assert hasattr(viz, 'buttons')
        assert len(viz.buttons) > 0
        
        # Check button types
        button_actions = [btn['action'] for btn in viz.buttons]
        
        # Should have control buttons
        assert 'toggle_pause' in button_actions
        assert 'step_forward' in button_actions
        assert 'step_backward' in button_actions
        
        # Should have speed buttons
        assert any('speed_' in action for action in button_actions)
        
        # Should have display mode buttons
        assert any('display_' in action for action in button_actions)
        
        # Should have tool buttons
        assert any('tool_' in action for action in button_actions)
        
        # Should have tab buttons
        assert any('tab_' in action for action in button_actions)
    
    @patch('pygame.display.set_mode')
    @patch('pygame.font.Font')
    def test_time_series_data_handling(self, mock_font, mock_display):
        """Test that time series data is handled correctly."""
        mock_display.return_value = MagicMock()
        mock_font.return_value = MagicMock()
        
        viz = GeologyVisualizer()
        
        # Test with time_series_data attribute
        viz.simulation.time_series_data = {
            'time': [0, 1, 2],
            'avg_temperature': [20, 21, 22],
            'net_power': [100, 95, 90]
        }
        
        # Should not raise error when accessing graphs tab
        try:
            # Mock the matplotlib imports
            with patch('matplotlib.pyplot'):
                with patch('matplotlib.backends.backend_agg'):
                    viz._draw_graphs_tab(10, 10)
        except AttributeError as e:
            if 'time_series' in str(e):
                pytest.fail(f"Should handle time_series_data attribute: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 