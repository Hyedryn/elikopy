#!/usr/bin/env python3
"""
Simple test script for the ElikoPy configuration system
"""

import sys
import tempfile
from pathlib import Path

# Add elikopy to path
sys.path.insert(0, str(Path(__file__).parent))

from elikopy.core.config import ElikopyConfig, ConfigValidationError

def test_default_config():
    """Test creating and validating default configuration"""
    print("Testing default configuration...")
    
    # Create default config
    config = ElikopyConfig()
    
    # Validate
    errors = config.validate()
    if errors:
        print(f"❌ Default config validation failed: {errors}")
        return False
    
    print("✅ Default configuration is valid")
    return True

def test_config_file_operations():
    """Test saving and loading configuration files"""
    print("\nTesting configuration file operations...")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test YAML format
        yaml_file = temp_path / "test_config.yaml"
        config = ElikopyConfig(study_name="test_study")
        
        try:
            config.to_file(yaml_file)
            print("✅ YAML config saved successfully")
        except Exception as e:
            print(f"❌ Failed to save YAML config: {e}")
            return False
        
        try:
            loaded_config = ElikopyConfig.from_file(yaml_file)
            if loaded_config.study_name == "test_study":
                print("✅ YAML config loaded successfully")
            else:
                print("❌ YAML config loaded with incorrect data")
                return False
        except Exception as e:
            print(f"❌ Failed to load YAML config: {e}")
            return False
        
        # Test JSON format
        json_file = temp_path / "test_config.json"
        try:
            config.to_file(json_file)
            print("✅ JSON config saved successfully")
        except Exception as e:
            print(f"❌ Failed to save JSON config: {e}")
            return False
        
        try:
            loaded_config = ElikopyConfig.from_file(json_file)
            if loaded_config.study_name == "test_study":
                print("✅ JSON config loaded successfully")
            else:
                print("❌ JSON config loaded with incorrect data")
                return False
        except Exception as e:
            print(f"❌ Failed to load JSON config: {e}")
            return False
    
    return True

def test_validation():
    """Test configuration validation"""
    print("\nTesting configuration validation...")
    
    # Test invalid DTI config
    config = ElikopyConfig()
    config.dti.fit_method = "INVALID_METHOD"
    config.dti.mask_threshold = 2.0  # Invalid threshold
    
    errors = config.validate()
    if not errors:
        print("❌ Validation should have failed for invalid config")
        return False
    
    print(f"✅ Validation correctly identified {len(errors)} errors")
    for error in errors:
        print(f"   - {error}")
    
    return True

def test_templates():
    """Test template creation"""
    print("\nTesting template creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test default template
        try:
            ElikopyConfig.create_default_template(temp_path / "default", format="yaml")
            if (temp_path / "default.yaml").exists():
                print("✅ Default template created successfully")
            else:
                print("❌ Default template file not found")
                return False
        except Exception as e:
            print(f"❌ Failed to create default template: {e}")
            return False
        
        # Test minimal template
        try:
            ElikopyConfig.create_minimal_template(
                temp_path / "minimal", 
                processing_types=['dti', 'noddi'],
                format="yaml"
            )
            if (temp_path / "minimal.yaml").exists():
                print("✅ Minimal template created successfully")
            else:
                print("❌ Minimal template file not found")
                return False
        except Exception as e:
            print(f"❌ Failed to create minimal template: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("Testing ElikoPy Configuration System")
    print("=" * 40)
    
    tests = [
        test_default_config,
        test_config_file_operations,
        test_validation,
        test_templates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())