#!/usr/bin/env python3
"""
Configuration management CLI for ElikoPy

This script provides command-line utilities for creating, validating,
and managing ElikoPy configuration files.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from elikopy.core.config import ElikopyConfig, ConfigValidationError


def create_template(args):
    """Create a configuration template"""
    try:
        if args.minimal:
            processing_types = args.processing_types or ['dti']
            ElikopyConfig.create_minimal_template(
                args.output,
                processing_types=processing_types,
                format=args.format
            )
            print(f"✅ Minimal template created: {args.output}")
        else:
            ElikopyConfig.create_default_template(args.output, format=args.format)
            print(f"✅ Default template created: {args.output}")
            
    except Exception as e:
        print(f"❌ Failed to create template: {e}")
        return 1
    
    return 0


def validate_config(args):
    """Validate a configuration file"""
    try:
        config = ElikopyConfig.from_file(args.config)
        errors = config.validate()
        
        if not errors:
            print(f"✅ Configuration is valid: {args.config}")
            if args.verbose:
                summary = config.get_summary()
                print("\nConfiguration Summary:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
        else:
            print(f"❌ Configuration has {len(errors)} error(s):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            return 1
            
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to validate configuration: {e}")
        return 1
    
    return 0


def convert_config(args):
    """Convert configuration between formats"""
    try:
        # Load configuration
        config = ElikopyConfig.from_file(args.input)
        
        # Save in new format
        config.to_file(args.output)
        
        print(f"✅ Configuration converted: {args.input} -> {args.output}")
        
    except Exception as e:
        print(f"❌ Failed to convert configuration: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ElikoPy Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default template
  python -m elikopy.cli.config_cli create-template config.yaml
  
  # Create minimal DTI template
  python -m elikopy.cli.config_cli create-template --minimal --processing dti config.yaml
  
  # Validate configuration
  python -m elikopy.cli.config_cli validate config.yaml
  
  # Convert YAML to JSON
  python -m elikopy.cli.config_cli convert config.yaml config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create template command
    template_parser = subparsers.add_parser(
        'create-template',
        help='Create configuration template'
    )
    template_parser.add_argument(
        'output',
        help='Output template file path'
    )
    template_parser.add_argument(
        '--format',
        choices=['yaml', 'json'],
        default='yaml',
        help='Template format (default: yaml)'
    )
    template_parser.add_argument(
        '--minimal',
        action='store_true',
        help='Create minimal template with only essential settings'
    )
    template_parser.add_argument(
        '--processing',
        dest='processing_types',
        nargs='+',
        choices=['preprocessing', 'dti', 'noddi', 'csd', 'msmt_csd', 
                'tracking', 'connectivity', 'fingerprinting'],
        help='Processing types to include in minimal template'
    )
    template_parser.set_defaults(func=create_template)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate configuration file'
    )
    validate_parser.add_argument(
        'config',
        help='Configuration file to validate'
    )
    validate_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed configuration summary'
    )
    validate_parser.set_defaults(func=validate_config)
    
    # Convert command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert configuration between formats'
    )
    convert_parser.add_argument(
        'input',
        help='Input configuration file'
    )
    convert_parser.add_argument(
        'output',
        help='Output configuration file'
    )
    convert_parser.set_defaults(func=convert_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())