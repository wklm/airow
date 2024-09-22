import json
import math
from typing import Dict, List, Optional, TypedDict, Union


class SummaryData(TypedDict):
    userId: str
    activityId: int
    activityName: str
    durationInSeconds: int
    startTimeInSeconds: int
    startTimeOffsetInSeconds: int
    activityType: str
    averageHeartRateInBeatsPerMinute: int
    activeKilocalories: int
    deviceName: str
    maxHeartRateInBeatsPerMinute: int


class LapData(TypedDict):
    startTimeInSeconds: int
    airTemperatureCelsius: int
    heartRate: int
    totalDistanceInMeters: int
    timerDurationInSeconds: int


class SampleData(TypedDict):
    """TypedDict for sample data with keys containing hyphens."""

    recording_rate: int  # Maps 'recording-rate' to this field
    sample_type: str  # Maps 'sample-type' to this field
    data: str


class HeartRateSample(TypedDict):
    sample_index: int
    heart_rate: Optional[float]


class Lap(TypedDict):
    startTimeInSeconds: int
    totalDistanceInMeters: int
    timerDurationInSeconds: int
    heartRateSamples: List[HeartRateSample]


class TrainingSummary(TypedDict):
    activityOverview: Dict[str, Union[str, int]]
    laps: List[Lap]


class DataLoader:
    @staticmethod
    def load_summary(filepath: str) -> SummaryData:
        """
        Load summary data from a JSON file.

        This function reads the summary JSON file and returns the data as a
        dictionary conforming to the SummaryData TypedDict.

        Parameters
        ----------
        filepath : str
            The path to the summary JSON file.

        Returns
        -------
        SummaryData
            The summary data extracted from the JSON file.

        Examples
        --------
        >>> summary_data = DataLoader.load_summary('data/summary.json')
        >>> print(summary_data['activityType'])
        'INDOOR_CYCLING'
        """
        with open(filepath, "r") as file:
            data: SummaryData = json.load(file)
        return data

    @staticmethod
    def load_laps(filepath: str) -> List[LapData]:
        """
        Load laps data from a JSON file.

        This function reads the laps JSON file and returns the data as a list
        of dictionaries conforming to the LapData TypedDict.

        Parameters
        ----------
        filepath : str
            The path to the laps JSON file.

        Returns
        -------
        List[LapData]
            A list of lap data dictionaries.

        Examples
        --------
        >>> laps_data = DataLoader.load_laps('data/laps.json')
        >>> print(len(laps_data))
        2
        """
        with open(filepath, "r") as file:
            data: List[LapData] = json.load(file)
        return data

    @staticmethod
    def load_samples(filepath: str) -> List[SampleData]:
        """
        Load samples data from a JSON file.

        This method reads the samples JSON file, maps keys containing hyphens
        to underscores, and returns the data as a list of dictionaries
        conforming to the SampleData TypedDict.

        Parameters
        ----------
        filepath : str
            The path to the samples JSON file.

        Returns
        -------
        List[SampleData]
            A list of sample data dictionaries with corrected keys.

        Examples
        --------
        >>> samples_data = DataLoader.load_samples('data/samples.json')
        >>> print(len(samples_data))
        8
        """
        with open(filepath, "r") as file:
            raw_data: List[Dict[str, Union[str, int]]] = json.load(file)

        data: List[SampleData] = []
        for item in raw_data:
            mapped_item: SampleData = {
                "recording_rate": int(item["recording-rate"]),
                "sample_type": str(item["sample-type"]),
                "data": str(item["data"]),
            }
            data.append(mapped_item)
        return data


class DataProcessor:
    @staticmethod
    def process_data(
        summary_data: SummaryData,
        laps_data: List[LapData],
        samples_data: List[SampleData],
        outlier_hr_jump_treshold: float,
        number_of_hr_samples_per_lap: int,
    ) -> TrainingSummary:
        """
        Process the loaded data and return the consolidated output.

        This function consolidates the summary, laps, and samples data into a
        structured format specified by the Output JSON Requirements. It builds
        the activity overview and processes the heart rate samples for each lap.
        It also performs outlier identification and cleaning, reverses the
        aggregation of heart rate measurements, and interpolates the data to
        achieve a recording rate of 1.

        The pre-processing covers outlier identification and cleaning. The
        initial recording rate is set to 5, whereas each observation is a
        median aggregate of the 5 tick heart rate measurements. This function
        reverses the aggregation step and backward interpolates the observations
        to end up with 5 * (n - 1) heart rate measurements with a recording
        rate of 1, where n denotes the initial number of observations.

        Parameters
        ----------
        summary_data : SummaryData
            The summary data loaded from the summary JSON file.
        laps_data : List[LapData]
            The laps data loaded from the laps JSON file.
        samples_data : List[SampleData]
            The samples data loaded from the samples JSON file.
        outlier_hr_jump_treshold : int
            The threshold for identifying outliers in heart rate data.
        number_of_hr_samples_per_lap : int
            The number of heart rate samples per lap.

        Returns
        -------
        ProcessedData
            The processed data ready for export.

        Examples
        --------
        >>> processed_data = DataProcessor.process_data(
        ...     summary_data, laps_data, samples_data)
        >>> print(processed_data['activityOverview']['deviceName'])
        'instinct2'
        """
        # Build activity overview with all required keys
        activity_overview_keys = [
            "userId",
            "activityId",
            "activityName",
            "durationInSeconds",
            "startTimeInSeconds",
            "startTimeOffsetInSeconds",
            "activityType",
            "averageHeartRateInBeatsPerMinute",
            "activeKilocalories",
            "deviceName",
            "maxHeartRateInBeatsPerMinute",
        ]

        assert (
            number_of_hr_samples_per_lap > 0
        ), "Number of heart rate sensors must be greater than 0"

        activity_overview: Dict[str, Union[str, int]] = {
            k: summary_data[k] for k in activity_overview_keys
        }

        # Get heart rate samples (sample_type == "2")
        heart_rate_samples: List[SampleData] = [
            sample for sample in samples_data if sample["sample_type"] == "2"
        ]

        processed_laps: List[Lap] = []
        num_laps = len(laps_data)
        for i in range(num_laps):
            lap = laps_data[i]
            # Extract two heart rate sample entries per lap
            hr_samples_for_lap = heart_rate_samples[
                i * number_of_hr_samples_per_lap:
                    (i + 1) * number_of_hr_samples_per_lap
            ]

            def safe_str_to_float(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return math.nan

            hr_values: List[float] = []
            for sample in hr_samples_for_lap:
                hr_values += [
                    safe_str_to_float(s) for s in sample["data"].split(",")
                ]
                
            # Replace NaN values with the previous value 
            for idx in range(1, len(hr_values)):
                if math.isnan(hr_values[idx]):
                    hr_values[idx] = hr_values[idx - 1]
             
            # Replace outliers with the previous value
            for idx in range(1, len(hr_values) - 1):
                prev_hr = hr_values[idx - 1]
                curr_hr = hr_values[idx]
                next_hr = hr_values[idx + 1]
                if curr_hr is not None:
                    if prev_hr is not None:
                        if next_hr is not None:
                            hr_diff = abs(curr_hr - prev_hr)
                            if hr_diff > outlier_hr_jump_treshold:
                                if hr_diff > outlier_hr_jump_treshold:
                                    hr_values[idx] = prev_hr
                                    
            # Interpolate to generate data at recording rate of 1
            interpolated_hr_values: List[Optional[float]] = []
            for idx in range(len(hr_values) - 1):
                h0 = hr_values[idx]
                h1 = hr_values[idx + 1]
                if h0 is None or h1 is None:
                    # Append None values for the 5 intervals
                    interpolated_hr_values.extend([None] * 5)
                else:
                    for j in range(5):
                        alpha = j / 5.0  # From 0.0 to 0.8
                        hr = h0 + alpha * (h1 - h0)
                        interpolated_hr_values.append(hr)

            # Build heart rate samples
            heart_rate_samples_list: List[HeartRateSample] = [
                {"sample_index": idx, "heart_rate": hr}
                for idx, hr in enumerate(interpolated_hr_values)
            ]

            # Build processed lap
            processed_lap: Lap = {
                "startTimeInSeconds": lap["startTimeInSeconds"],
                "totalDistanceInMeters": lap["totalDistanceInMeters"],
                "timerDurationInSeconds": lap["timerDurationInSeconds"],
                "heartRateSamples": heart_rate_samples_list,
            }
            processed_laps.append(processed_lap)

        processed_data: TrainingSummary = {
            "activityOverview": activity_overview,
            "laps": processed_laps,
        }

        return processed_data


class DataExporter:
    @staticmethod
    def export_to_json(data: TrainingSummary, filepath: str) -> None:
        """
        Export the processed data to a JSON file.

        This function writes the processed data to a JSON file with indentation
        for better readability.

        Parameters
        ----------
        data : ProcessedData
            The processed data to be exported.
        filepath : str
            The path where the output JSON file will be saved.

        Returns
        -------
        None

        Examples
        --------
        >>> DataExporter.export_to_json(
        ...     processed_data, 'data/training_summary.json')
        """
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)


class DataPipeline:
    @staticmethod
    def main(
        summary_filepath: str,
        laps_filepath: str,
        samples_filepath: str,
        output_filepath: Optional[str] = None,
        outlier_hr_jump_treshold: float = math.inf,
        number_of_hr_sensors: int = 2,
    ) -> TrainingSummary:
        """
        Main method to load, process, and optionally export data.

        This method orchestrates the data loading, processing, and exporting
        steps. It uses the DataLoader to load data from JSON files,
        DataProcessor to process the data, and DataExporter to export the
        processed data to a JSON file if an output path is provided.

        Parameters
        ----------
        summary_filepath : str
            Path to the summary JSON file.
        laps_filepath : str
            Path to the laps JSON file.
        samples_filepath : str
            Path to the samples JSON file.
        outlier_hr_jump_treshold : int
            The threshold for identifying outliers in heart rate data.
        output_filepath : Optional[str], default None
            Path to the output JSON file. If not provided, the data will not be
            exported.

        Returns
        -------
        ProcessedData
            The processed data after consolidation.

        Examples
        --------
        >>> processed_data = DataPipeline.main(
        ...     'data/summary.json',
        ...     'data/laps.json',
        ...     'data/samples.json',
        ...     'data/training_summary.json'
        ... )
        """
        # Load data
        summary_data = DataLoader.load_summary(summary_filepath)
        laps_data = DataLoader.load_laps(laps_filepath)
        samples_data = DataLoader.load_samples(samples_filepath)

        processed_data: TrainingSummary = DataProcessor.process_data(
            summary_data,
            laps_data,
            samples_data,
            outlier_hr_jump_treshold,
            number_of_hr_sensors,
        )

        # Export data if output filepath is provided
        if output_filepath:
            DataExporter.export_to_json(processed_data, output_filepath)

        return processed_data


# Example usage:
if __name__ == "__main__":
    summary_filepath = "data/summary.json"
    laps_filepath = "data/laps.json"
    samples_filepath = "data/samples.json"
    output_filepath = "data/training_summary.json"
    outlier_hr_jump_treshold: int

    DataPipeline.main(
        summary_filepath, laps_filepath, samples_filepath, output_filepath
    )
