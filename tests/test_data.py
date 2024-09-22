import json
import os
import unittest
from io import StringIO
from typing import List
from unittest.mock import mock_open, patch

# Importing classes from the original module
from training_analytics.data import (
    DataExporter,
    DataLoader,
    DataPipeline,
    DataProcessor,
    LapData,
    SampleData,
    SummaryData,
    TrainingSummary,
)


class TestDataModule(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.summary_json = """
        {
            "userId": "1234567890",
            "activityId": 9480958402,
            "activityName": "Indoor Cycling",
            "durationInSeconds": 3667,
            "startTimeInSeconds": 1661158927,
            "startTimeOffsetInSeconds": 7200,
            "activityType": "INDOOR_CYCLING",
            "averageHeartRateInBeatsPerMinute": 150,
            "activeKilocalories": 561,
            "deviceName": "instinct2",
            "maxHeartRateInBeatsPerMinute": 190
        }
        """
        self.laps_json = """
        [
            {
                "startTimeInSeconds": 1661158927,
                "airTemperatureCelsius": 28,
                "heartRate": 109,
                "totalDistanceInMeters": 15,
                "timerDurationInSeconds": 600
            },
            {
                "startTimeInSeconds": 1661158929,
                "airTemperatureCelsius": 28,
                "heartRate": 107,
                "totalDistanceInMeters": 30,
                "timerDurationInSeconds": 900
            }
        ]
        """
        self.samples_json = """
        [
            {
                "recording-rate": 5,
                "sample-type": "0",
                "data": "86,87,88,88,88,90,91"
            },
            {
                "recording-rate": 5,
                "sample-type": "2",
                "data": "120,126,122,140,142,155,145"
            },
            {
                "recording-rate": 5,
                "sample-type": "2",
                "data": "141,147,155,160,180,152,120"
            },
            {
                "recording-rate": 5,
                "sample-type": "0",
                "data": "86,87,88,88,88,90,91"
            },
            {
                "recording-rate": 5,
                "sample-type": "1",
                "data": "143,87,88,88,88,90,91"
            },
            {
                "recording-rate": 5,
                "sample-type": "2",
                "data": "143,151,164,null,173,181,180"
            },
            {
                "recording-rate": 5,
                "sample-type": "2",
                "data": "182,170,188,181,174,172,158"
            },
            {
                "recording-rate": 5,
                "sample-type": "3",
                "data": "143,87,88,88,88,90,91"
            }
        ]
        """

    @patch("builtins.open")
    def test_load_summary(self, mock_open_file):
        """Test loading of summary data."""
        mock_open_file.return_value = StringIO(self.summary_json)
        filepath = "summary.json"
        summary_data = DataLoader.load_summary(filepath)
        self.assertIsInstance(summary_data, dict)
        self.assertEqual(summary_data["userId"], "1234567890")
        self.assertEqual(summary_data["activityType"], "INDOOR_CYCLING")

    @patch("builtins.open")
    def test_load_laps(self, mock_open_file):
        """Test loading of laps data."""
        mock_open_file.return_value = StringIO(self.laps_json)
        filepath = "laps.json"
        laps_data = DataLoader.load_laps(filepath)
        self.assertIsInstance(laps_data, list)
        self.assertEqual(len(laps_data), 2)
        self.assertEqual(laps_data[0]["startTimeInSeconds"], 1661158927)

    @patch("builtins.open")
    def test_load_samples(self, mock_open_file):
        """Test loading of samples data with key mapping."""
        mock_open_file.return_value = StringIO(self.samples_json)
        filepath = "samples.json"
        samples_data = DataLoader.load_samples(filepath)
        self.assertIsInstance(samples_data, list)
        self.assertEqual(len(samples_data), 8)
        self.assertIn("sample_type", samples_data[0])
        self.assertEqual(samples_data[1]["sample_type"], "2")

    def test_process_data(self):
        """Test processing of data into the correct format."""
        # Load data using StringIO to simulate file operations
        summary_data = json.loads(self.summary_json)
        laps_data = json.loads(self.laps_json)
        raw_samples_data = json.loads(self.samples_json)

        # Map the sample data keys
        samples_data: List[SampleData] = []
        for item in raw_samples_data:
            mapped_item: SampleData = {
                "recording_rate": item["recording-rate"],
                "sample_type": item["sample-type"],
                "data": item["data"],
            }
            samples_data.append(mapped_item)

        processed_data = DataProcessor.process_data(
            summary_data,
            laps_data,
            samples_data,
            outlier_hr_jump_treshold=20,
            number_of_hr_samples_per_lap=2,
        )

        # Assertions for activity overview
        activity_overview = processed_data["activityOverview"]
        self.assertEqual(activity_overview["userId"], "1234567890")
        self.assertEqual(activity_overview["activityType"], "INDOOR_CYCLING")

        # Assertions for laps
        laps = processed_data["laps"]
        self.assertEqual(len(laps), 2)

        # Check first lap heart rate samples
        first_lap_hr_samples = laps[0]["heartRateSamples"]

        # Calculate the number of heart rate values per lap
        num_hr_values = sum(
            len(item["data"].split(","))
            for item in raw_samples_data[:3]  # Entries for the first lap
            if item["sample-type"] == "2"
        )
        expected_samples = 5 * (num_hr_values - 1)  # Correct calculation
        self.assertEqual(len(first_lap_hr_samples), expected_samples)

        # Check that the first sample matches the first heart rate value
        self.assertEqual(first_lap_hr_samples[0]["heart_rate"], 120.0)

        # Since NaN values are replaced, the last sample should be a float
        self.assertIsInstance(first_lap_hr_samples[-1]["heart_rate"], float)

    def test_outlier_detection_and_cleaning(self):
        """Test outlier identification and cleaning."""
        summary_data = json.loads(self.summary_json)
        laps_data = json.loads(self.laps_json)
        # Create sample data with outliers
        samples_data: List[SampleData] = [
            {
                "recording_rate": 5,
                "sample_type": "2",
                "data": "100,105,200,110,115,300,120",  # 200 and 300 are outliers
            },
            {
                "recording_rate": 5,
                "sample_type": "2",
                "data": "125,130,135,400,140,145,150",  # 400 is an outlier
            },
        ]

        processed_data = DataProcessor.process_data(
            summary_data,
            laps_data,
            samples_data,
            outlier_hr_jump_treshold=20,
            number_of_hr_samples_per_lap=2,
        )

        laps = processed_data["laps"]
        first_lap_hr_samples = laps[0]["heartRateSamples"]

        # Ensure that heart rates are within a reasonable range after outlier removal
        for sample in first_lap_hr_samples:
            hr = sample["heart_rate"]
            if hr is not None:
                self.assertLessEqual(hr, 180)

    def test_reverse_aggregation_and_interpolation(self):
        """Test reverse aggregation and interpolation."""
        summary_data = json.loads(self.summary_json)
        laps_data = json.loads(self.laps_json)
        # Sample data with known values
        samples_data: List[SampleData] = [
            {"recording_rate": 5, "sample_type": "2", "data": "100,110,120"}
        ]

        processed_data = DataProcessor.process_data(
            summary_data,
            [laps_data[0]],
            samples_data,
            outlier_hr_jump_treshold=20,
            number_of_hr_samples_per_lap=1,
        )

        laps = processed_data["laps"]
        hr_samples = laps[0]["heartRateSamples"]

        # Expected number of interpolated samples: 5 * (3 - 1) = 10
        self.assertEqual(len(hr_samples), 10)

        # Check that interpolation is correct
        expected_values = []
        hr_values = [100.0, 110.0, 120.0]
        for idx in range(len(hr_values) - 1):
            h0 = hr_values[idx]
            h1 = hr_values[idx + 1]
            for j in range(5):
                alpha = j / 5.0
                hr = h0 + alpha * (h1 - h0)
                expected_values.append(hr)

        for idx, sample in enumerate(hr_samples):
            self.assertAlmostEqual(sample["heart_rate"], expected_values[idx])

    @patch("builtins.open", new_callable=mock_open)
    def test_export_to_json(self, mock_open_file):
        """Test exporting processed data to a JSON file."""
        processed_data: TrainingSummary = {
            "activityOverview": {
                "userId": "1234567890",
                "activityType": "INDOOR_CYCLING",
                "deviceName": "instinct2",
                "maxHeartRateInBeatsPerMinute": 190,
                "durationInSeconds": 3667,
            },
            "laps": [],
        }
        filepath = "output.json"
        DataExporter.export_to_json(processed_data, filepath)
        mock_open_file.assert_called_with(filepath, "w")
        handle = mock_open_file()
        handle.write.assert_called()  # Check that write was called

    @patch("builtins.open")
    def test_data_pipeline_main(self, mock_open_file):
        """Test the full data pipeline."""
        # Mock the open function for each file
        mock_open_file.side_effect = [
            StringIO(self.summary_json),
            StringIO(self.laps_json),
            StringIO(self.samples_json),
            mock_open().return_value,  # For the output file
        ]

        summary_filepath = "summary.json"
        laps_filepath = "laps.json"
        samples_filepath = "samples.json"
        output_filepath = "output.json"

        processed_data: TrainingSummary = DataPipeline.main(
            summary_filepath,
            laps_filepath,
            samples_filepath,
            output_filepath,
        )

        # Assertions for processed data
        self.assertIn("activityOverview", processed_data)
        self.assertIn("laps", processed_data)
        self.assertEqual(len(processed_data["laps"]), 2)

    def test_empty_samples(self):
        """Test processing when samples data is empty."""
        summary_data: SummaryData = json.loads(self.summary_json)
        laps_data: List[LapData] = json.loads(self.laps_json)
        samples_data: List[SampleData] = []  # Empty samples list

        processed_data = DataProcessor.process_data(
            summary_data,
            laps_data,
            samples_data,
            outlier_hr_jump_treshold=20,
            number_of_hr_samples_per_lap=2,
        )

        laps = processed_data["laps"]
        for lap in laps:
            self.assertEqual(len(lap["heartRateSamples"]), 0)

 

    def test_invalid_json_format(self):
        """Test loading data with invalid JSON format."""
        invalid_json = "{ invalid json }"
        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with self.assertRaises(json.JSONDecodeError):
                DataLoader.load_summary("summary.json")

    def test_missing_keys_in_summary(self):
        """Test processing when summary data is missing required keys."""
        summary_data = {
            "userId": "1234567890",
            # Missing other required keys
        }
        laps_data = json.loads(self.laps_json)
        samples_data: List[SampleData] = []

        with self.assertRaises(KeyError):
            DataProcessor.process_data(
                summary_data,
                laps_data,
                samples_data,
                outlier_hr_jump_treshold=20,
                number_of_hr_samples_per_lap=2,
            )

    def test_large_number_of_laps(self):
        """Test processing with a large number of laps."""
        # Generate large laps data
        laps_data: List[LapData] = []
        heart_rate_samples: List[SampleData] = []
        for i in range(100):
            laps_data.append(
                {
                    "startTimeInSeconds": 1661158927 + i * 600,
                    "airTemperatureCelsius": 28,
                    "heartRate": 100 + i,
                    "totalDistanceInMeters": 15 + i * 10,
                    "timerDurationInSeconds": 600,
                }
            )
            # Create corresponding heart rate samples
            heart_rate_sample = {
                "recording-rate": 5,
                "sample-type": "2",
                "data": ",".join([str(100 + i)] * 7),
            }
            heart_rate_samples.extend([heart_rate_sample, heart_rate_sample])

        summary_data = json.loads(self.summary_json)
        samples_data: List[SampleData] = []
        for item in heart_rate_samples:
            mapped_item: SampleData = {
                "recording_rate": item["recording-rate"],
                "sample_type": item["sample-type"],
                "data": item["data"],
            }
            samples_data.append(mapped_item)

        processed_data = DataProcessor.process_data(
            summary_data,
            laps_data,
            samples_data,
            outlier_hr_jump_treshold=20,
            number_of_hr_samples_per_lap=2,
        )

        self.assertEqual(len(processed_data["laps"]), 100)
        for lap in processed_data["laps"]:
            # Each lap has 14 heart rate values (two samples of 7 values each)
            num_hr_values = 14
            expected_samples = 5 * (num_hr_values - 1)  # 5 * 13 = 65
            self.assertEqual(len(lap["heartRateSamples"]), expected_samples)

    def test_non_numeric_heart_rate_values(self):
        """Test handling of non-numeric values in heart rate data."""
        summary_data = json.loads(self.summary_json)
        laps_data = json.loads(self.laps_json)
        raw_samples_data = json.loads(self.samples_json)

        # Introduce non-numeric value
        raw_samples_data[1]["data"] = "120,abc,122,140"

        # Map the sample data keys
        samples_data: List[SampleData] = []
        for item in raw_samples_data:
            mapped_item: SampleData = {
                "recording_rate": item["recording-rate"],
                "sample_type": item["sample-type"],
                "data": item["data"],
            }
            samples_data.append(mapped_item)

        processed_data = DataProcessor.process_data(
            summary_data,
            laps_data,
            samples_data,
            outlier_hr_jump_treshold=20,
            number_of_hr_samples_per_lap=2,
        )

        # Check that the non-numeric value was handled
        laps = processed_data["laps"]
        first_lap_hr_samples = laps[0]["heartRateSamples"]

        # Since 'abc' was replaced with previous value, no NaN should remain
        for sample in first_lap_hr_samples:
            self.assertIsInstance(sample["heart_rate"], float)

    def test_output_file_creation(self):
        """Test if the output file is created successfully."""
        processed_data: TrainingSummary = {
            "activityOverview": {
                "userId": "1234567890",
                "activityType": "INDOOR_CYCLING",
                "deviceName": "instinct2",
                "maxHeartRateInBeatsPerMinute": 190,
                "durationInSeconds": 3667,
            },
            "laps": [],
        }
        filepath = "output.json"

        # Remove the file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)

        DataExporter.export_to_json(processed_data, filepath)
        self.assertTrue(os.path.exists(filepath))

        # Clean up
        os.remove(filepath)


# Run the tests
if __name__ == "__main__":
    unittest.main()