import unittest
import yaml
import os
import sys

# Adjust the Python path to include the root directory of the project
# This allows importing modules if needed, though not strictly for this test file.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

CONFIG_FILE_PATH = "config.yaml"

class TestConfigValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the YAML configuration file once for all tests in this class."""
        cls.config_data = None
        cls.config_load_error = None
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                cls.config_data = yaml.safe_load(f)
        except FileNotFoundError:
            cls.config_load_error = f"Configuration file not found: {CONFIG_FILE_PATH}"
        except yaml.YAMLError as e:
            cls.config_load_error = f"Error parsing YAML configuration file: {e}"
        except Exception as e:
            cls.config_load_error = f"An unexpected error occurred loading config: {e}"

    def setUp(self):
        """Skip tests or fail if config was not loaded."""
        if self.config_load_error:
            self.fail(self.config_load_error) # Fail all tests if config couldn't be loaded
        if self.config_data is None:
            self.skipTest(f"Skipping tests as config data is None (potentially empty config file: {CONFIG_FILE_PATH}).")


    # Define expected schema
    # { 'key_name': (expected_type, is_required, optional_validation_lambda), ... }
    # For nested dicts, the type is dict, and a sub-schema can be defined.
    SCHEMA = {
        'model': {
            'type': dict, 'required': True, 'schema': {
                'd_raw_feature': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'hidden_dim': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'num_layers': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'learning_rate': {'type': float, 'required': True, 'check': lambda x: x > 0},
                'num_actions': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'num_heads': {'type': int, 'required': True, 'check': lambda x: x > 0} # Added in AICFRTrainer
            }
        },
        'game_engine': {
            'type': dict, 'required': True, 'schema': {
                'num_players': {'type': int, 'required': True, 'check': lambda x: x >= 2},
                'starting_stack': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'big_blind': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'small_blind': {'type': int, 'required': True, 'check': lambda x: x > 0}
                # 'player_strategies' is handled by game/self-play, not directly in config for this test
            }
        },
        'training': {
            'type': dict, 'required': True, 'schema': {
                'num_training_hands': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'save_model_every_n_hands': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'batch_size': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'num_epochs': {'type': int, 'required': True, 'check': lambda x: x > 0},
                'save_model_path': {'type': str, 'required': True}
            }
        },
        'logging': {
            'type': dict, 'required': True, 'schema': {
                'level': {'type': str, 'required': True, 'check': lambda x: x in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]},
                'log_file': {'type': str, 'required': True}
            }
        }
    }

    def test_config_is_loaded_as_dict(self):
        """Test that the configuration file is loaded as a dictionary."""
        self.assertIsInstance(self.config_data, dict, "Config data should be a dictionary.")

    def _validate_section(self, section_name, section_schema, section_data):
        """Helper to validate a section of the config."""
        self.assertIn(section_name, self.config_data, f"Top-level key '{section_name}' missing in config.")
        self.assertIsInstance(section_data, section_schema['type'],
                              f"Section '{section_name}' expected type {section_schema['type']}, got {type(section_data)}.")

        for key, rules in section_schema['schema'].items():
            if rules['required']:
                self.assertIn(key, section_data, f"Required key '{key}' missing in '{section_name}' section.")

            if key in section_data: # Only check type and value if key exists
                value = section_data[key]
                self.assertIsInstance(value, rules['type'],
                                      f"Key '{key}' in '{section_name}' expected type {rules['type']}, got {type(value)}.")
                if 'check' in rules:
                    self.assertTrue(rules['check'](value),
                                    f"Key '{key}' in '{section_name}' failed validation check (value: {value}).")

    def test_model_section(self):
        """Validate the 'model' section of the config."""
        section_name = 'model'
        if section_name in self.config_data: # Ensure section exists before trying to validate
            self._validate_section(section_name, self.SCHEMA[section_name], self.config_data.get(section_name))
        elif self.SCHEMA[section_name]['required']: # Fail if required section is missing
             self.fail(f"Required top-level key '{section_name}' missing in config.")


    def test_game_engine_section(self):
        """Validate the 'game_engine' section of the config."""
        section_name = 'game_engine'
        if section_name in self.config_data:
            self._validate_section(section_name, self.SCHEMA[section_name], self.config_data.get(section_name))
        elif self.SCHEMA[section_name]['required']:
             self.fail(f"Required top-level key '{section_name}' missing in config.")


    def test_training_section(self):
        """Validate the 'training' section of the config."""
        section_name = 'training'
        if section_name in self.config_data:
            self._validate_section(section_name, self.SCHEMA[section_name], self.config_data.get(section_name))
        elif self.SCHEMA[section_name]['required']:
             self.fail(f"Required top-level key '{section_name}' missing in config.")


    def test_logging_section(self):
        """Validate the 'logging' section of the config."""
        section_name = 'logging'
        if section_name in self.config_data:
            self._validate_section(section_name, self.SCHEMA[section_name], self.config_data.get(section_name))
        elif self.SCHEMA[section_name]['required']:
            self.fail(f"Required top-level key '{section_name}' missing in config.")


if __name__ == '__main__':
    unittest.main()
