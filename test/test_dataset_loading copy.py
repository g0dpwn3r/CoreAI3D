#!/usr/bin/env python3
"""
Test script for dataset loading functionality in CoreAI3D Dashboard
Tests the dataset loading, size calculations, and metadata handling.
"""

import sys
import os
import json
import csv

def test_dataset_metadata():
    """Test dataset metadata loading"""
    print("=" * 60)
    print("COREAI3D DATASET METADATA TEST")
    print("=" * 60)

    # Load the sample dataset metadata
    metadata_path = "training_data/sample_dataset/metadata.json"
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return False

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print("✓ Metadata loaded successfully")
        print(f"  Dataset name: {metadata.get('name', 'N/A')}")
        print(f"  Description: {metadata.get('description', 'N/A')}")
        print(f"  Data type: {metadata.get('data_type', 'N/A')}")
        print(f"  Files: {len(metadata.get('files', []))}")

        # Check dataset sizes
        dataset_metadata = metadata.get('metadata', {})
        expected_inputsize = 2
        expected_outputsize = 1
        expected_samplesize = 100

        actual_inputsize = dataset_metadata.get('inputsize', 'N/A')
        actual_outputsize = dataset_metadata.get('outputsize', 'N/A')
        actual_samplesize = dataset_metadata.get('samplesize', 'N/A')

        print("\nDataset sizes from metadata:")
        print(f"  Input size: {actual_inputsize} (expected: {expected_inputsize})")
        print(f"  Output size: {actual_outputsize} (expected: {expected_outputsize})")
        print(f"  Sample size: {actual_samplesize} (expected: {expected_samplesize})")

        # Check if sizes match expected
        if (actual_inputsize == expected_inputsize and
            actual_outputsize == expected_outputsize and
            actual_samplesize == expected_samplesize):
            print("✓ Metadata sizes match expected values")
            return True
        else:
            print("❌ Metadata sizes do not match expected values")
            return False

    except Exception as e:
        print(f"❌ Error loading metadata: {str(e)}")
        return False

def test_csv_file_analysis():
    """Test CSV file analysis and size calculation"""
    print("\n" + "=" * 60)
    print("CSV FILE ANALYSIS TEST")
    print("=" * 60)

    csv_path = "training_data/sample_dataset/test_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False

    try:
        # Read and analyze the CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            print("❌ CSV file is empty")
            return False

        print(f"✓ CSV file loaded successfully")
        print(f"  Total rows: {len(rows)}")
        print(f"  Columns: {len(rows[0]) if rows else 0}")

        # Check header
        first_row = rows[0]
        if first_row == ['x', 'y']:
            print("✓ Header detected: ['x', 'y']")
            data_rows = rows[1:]
        else:
            print("⚠ No header detected, treating first row as data")
            data_rows = rows

        print(f"  Data rows: {len(data_rows)}")

        # Analyze data structure
        if len(data_rows) > 0:
            sample_row = data_rows[0]
            print(f"  Sample data row: {sample_row}")

            # Check if data looks numeric
            try:
                numeric_row = [float(x) for x in sample_row]
                print("✓ Data appears to be numeric")
            except ValueError:
                print("⚠ Data contains non-numeric values")

        # Calculate expected sizes for CSV
        # For CSV files, typically last column is target/output, rest are inputs
        column_count = len(first_row)
        if column_count > 1:
            inputsize = column_count - 1  # All columns except last
            outputsize = 1  # Last column as target
            samplesize = len(data_rows)
        else:
            inputsize = 1
            outputsize = 0
            samplesize = len(data_rows)

        print("\nCalculated sizes:")
        print(f"  Input size: {inputsize}")
        print(f"  Output size: {outputsize}")
        print(f"  Sample size: {samplesize}")

        # Check against expected values
        expected_inputsize = 2
        expected_outputsize = 1
        expected_samplesize = 100

        if (inputsize == expected_inputsize and
            outputsize == expected_outputsize and
            samplesize == expected_samplesize):
            print("✓ Calculated sizes match expected values")
            return True
        else:
            print("❌ Calculated sizes do not match expected values")
            print(f"   Expected: input={expected_inputsize}, output={expected_outputsize}, samples={expected_samplesize}")
            print(f"   Calculated: input={inputsize}, output={outputsize}, samples={samplesize}")
            print("   NOTE: The CSV file has only 10 rows but metadata expects 100 samples")
            return False

    except Exception as e:
        print(f"❌ Error analyzing CSV file: {str(e)}")
        return False

def test_dataset_structure():
    """Test overall dataset structure"""
    print("\n" + "=" * 60)
    print("DATASET STRUCTURE TEST")
    print("=" * 60)

    dataset_dir = "training_data/sample_dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False

    print(f"✓ Dataset directory exists: {dataset_dir}")

    # Check for required files
    metadata_file = os.path.join(dataset_dir, "metadata.json")
    csv_file = os.path.join(dataset_dir, "test_data.csv")

    if os.path.exists(metadata_file):
        print("✓ Metadata file exists")
    else:
        print("❌ Metadata file missing")
        return False

    if os.path.exists(csv_file):
        print("✓ CSV data file exists")
    else:
        print("❌ CSV data file missing")
        return False

    # Check file sizes
    metadata_size = os.path.getsize(metadata_file)
    csv_size = os.path.getsize(csv_file)

    print(f"  Metadata file size: {metadata_size} bytes")
    print(f"  CSV file size: {csv_size} bytes")

    return True

def main():
    """Main test function"""
    print("Starting CoreAI3D Dataset Loading Tests...\n")

    # Test 1: Dataset metadata
    test1_passed = test_dataset_metadata()

    # Test 2: CSV file analysis
    test2_passed = test_csv_file_analysis()

    # Test 3: Dataset structure
    test3_passed = test_dataset_structure()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Dataset Metadata Test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"CSV File Analysis Test: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Dataset Structure Test: {'PASS' if test3_passed else 'FAIL'}")

    all_passed = test1_passed and test2_passed and test3_passed

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        print("\nISSUES IDENTIFIED:")
        if not test1_passed:
            print("- Dataset metadata sizes don't match expected values")
        if not test2_passed:
            print("- CSV file analysis shows mismatch between actual data (10 rows) and expected metadata (100 samples)")
        if not test3_passed:
            print("- Dataset structure issues")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)