import numpy as np
import scipy.stats as stats
from mcp.server.fastmcp import FastMCP
from numpy import floating
from typing import Tuple, Optional, List, Union

basic_stats_mcp = FastMCP(
    name="basic_stats_mcp",
)


@basic_stats_mcp.tool()
def calculate_mean(collection: List[Union[float, int]]) -> float | int:
    """Calculate mean or average value of a collection of numbers."""
    return sum(collection) / len(collection)


@basic_stats_mcp.tool()
def calculate_standard_deviation(collection: List[Union[float, int]]) -> floating:
    """Calculate standard deviation of a collection."""
    return np.std(np.array(collection))


@basic_stats_mcp.tool()
def calculate_median(collection: List[Union[float, int]]) -> floating:
    """Calculate median of a collection."""
    return np.median(collection)


@basic_stats_mcp.tool()
def calculate_mode(collection: List[Union[float, int]]) -> Union[float, int, List[Union[float, int]]]:
    """Calculate the mode (most frequent value) of a collection."""
    values, counts = np.unique(collection, return_counts=True)
    max_count = np.max(counts)
    modes = values[counts == max_count]
    if len(modes) == 1:
        return float(modes[0])
    return [float(mode) for mode in modes]


@basic_stats_mcp.tool()
def calculate_range(collection: List[Union[float, int]]) -> float:
    """Calculate the range (difference between max and min) of a collection."""
    return max(collection) - min(collection)


@basic_stats_mcp.tool()
def calculate_variance(collection: List[Union[float, int]]) -> floating:
    """Calculate the variance of a collection."""
    return np.var(np.array(collection))


@basic_stats_mcp.tool()
def calculate_quartiles(collection: List[Union[float, int]]) -> Tuple[floating, floating, floating]:
    """Calculate the quartiles (Q1, Q2, Q3) of a collection."""
    q1 = np.percentile(collection, 25)
    q2 = np.percentile(collection, 50)
    q3 = np.percentile(collection, 75)
    return q1, q2, q3


@basic_stats_mcp.tool()
def calculate_iqr(collection: List[Union[float, int]]) -> floating:
    """Calculate the interquartile range (IQR) of a collection."""
    q1, q3 = np.percentile(collection, 25), np.percentile(collection, 75)
    return q3 - q1


@basic_stats_mcp.tool()
def calculate_skewness(collection: List[Union[float, int]]) -> floating:
    """Calculate the skewness (measure of asymmetry) of a collection."""
    return stats.skew(collection)


@basic_stats_mcp.tool()
def calculate_covariance(collection1: List[Union[float, int]], collection2: list[Union[float, int]]) -> floating:
    """Calculate the covariance between two collections."""
    if len(collection1) != len(collection2):
        raise ValueError("Collections must be of the same length")
    return np.cov(collection1, collection2)[0, 1]


@basic_stats_mcp.tool()
def calculate_kurtosis(collection: List[Union[float, int]]) -> floating:
    """Calculate the kurtosis (measure of 'tailedness') of a collection."""
    return stats.kurtosis(collection)


@basic_stats_mcp.tool()
def calculate_correlation(collection1: List[Union[float, int]], collection2: list[Union[float, int]]) -> Tuple[floating, floating]:
    """Calculate the Pearson correlation coefficient and p-value between two collections."""
    if len(collection1) != len(collection2):
        raise ValueError("Collections must be of the same length")
    return stats.pearsonr(collection1, collection2)


@basic_stats_mcp.tool()
def calculate_z_scores(collection: List[Union[float, int]]) -> List[floating]:
    """Calculate z-scores (standard scores) for a collection."""
    return list(stats.zscore(collection))


@basic_stats_mcp.tool()
def calculate_confidence_interval(collection: List[Union[float, int]], confidence: float = 0.95) -> Tuple[floating, floating]:
    """Calculate the confidence interval for the mean of a collection."""
    mean = np.mean(collection)
    sem = stats.sem(collection)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(collection) - 1)
    return mean - interval, mean + interval


@basic_stats_mcp.tool()
def perform_t_test(collection: List[Union[float, int]], popmean: float = 0) -> Tuple[floating, floating]:
    """Perform a one-sample t-test comparing a collection to a population mean."""
    t_stat, p_value = stats.ttest_1samp(collection, popmean)
    return t_stat, p_value


@basic_stats_mcp.tool()
def detect_outliers(collection: List[Union[float, int]]) -> List[Union[float, int]]:
    """Detect outliers in a collection using the IQR method."""
    q1, q3 = np.percentile(collection, 25), np.percentile(collection, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in collection if x < lower_bound or x > upper_bound]


@basic_stats_mcp.tool()
def perform_anova(collections: List[Union[float, int]]) -> Tuple[floating, floating]:
    """Perform one-way ANOVA (Analysis of Variance) on multiple collections.

    Returns:
        Tuple containing the F-statistic and p-value.
    """
    return stats.f_oneway(*collections)


@basic_stats_mcp.tool()
def perform_chi_square_test(observed: List[List[int]]) -> Tuple[floating, floating]:
    """Perform chi-square test of independence between two categorical variables.

    Args:
        observed: A contingency table (matrix) of observed frequencies.

    Returns:
        Tuple containing the chi-square statistic and p-value.
    """
    return stats.chi2_contingency(observed)[:2]


@basic_stats_mcp.tool()
def perform_linear_regression(x: List[Union[float, int]], y: List[Union[float, int]]) -> Tuple[floating, floating, floating, floating, floating]:
    """Perform simple linear regression between two variables.

    Returns:
        Tuple containing slope, intercept, r-value, p-value, and standard error.
    """
    if len(x) != len(y):
        raise ValueError("Collections must be of the same length")
    result = stats.linregress(x, y)
    return result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr


@basic_stats_mcp.tool()
def calculate_moving_average(collection: List[Union[float, int]], window_size: int) -> List[floating]:
    """Calculate the moving average of a time series."""
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if window_size > len(collection):
        raise ValueError("Window size cannot be larger than the collection size")

    result = []
    for i in range(len(collection) - window_size + 1):
        window = collection[i:i + window_size]
        result.append(np.mean(window))
    return result


@basic_stats_mcp.tool()
def calculate_geometric_mean(collection: List[Union[float, int]]) -> floating:
    """Calculate the geometric mean of a collection of positive numbers."""
    if any(x <= 0 for x in collection):
        raise ValueError("All values must be positive for geometric mean calculation")
    return stats.gmean(collection)


@basic_stats_mcp.tool()
def calculate_harmonic_mean(collection: List[Union[float, int]]) -> floating:
    """Calculate the harmonic mean of a collection of positive numbers."""
    if any(x <= 0 for x in collection):
        raise ValueError("All values must be positive for harmonic mean calculation")
    return stats.hmean(collection)


@basic_stats_mcp.tool()
def calculate_percentile(collection: List[Union[float, int]], percentile: float) -> floating:
    """Calculate a specific percentile of a collection."""
    if percentile < 0 or percentile > 100:
        raise ValueError("Percentile must be between 0 and 100")
    return np.percentile(collection, percentile)


@basic_stats_mcp.tool()
def calculate_spearman_correlation(collection1: List[Union[float, int]], collection2: list[float | int]) -> Tuple[floating, floating]:
    """Calculate the Spearman rank correlation coefficient and p-value between two collections."""
    if len(collection1) != len(collection2):
        raise ValueError("Collections must be of the same length")
    return stats.spearmanr(collection1, collection2)


@basic_stats_mcp.tool()
def calculate_kendall_tau(collection1: List[Union[float, int]], collection2: list[float | int]) -> Tuple[floating, floating]:
    """Calculate Kendall's tau correlation coefficient and p-value between two collections."""
    if len(collection1) != len(collection2):
        raise ValueError("Collections must be of the same length")
    return stats.kendalltau(collection1, collection2)


@basic_stats_mcp.tool()
def test_normality(collection: List[Union[float, int]]) -> Tuple[floating, floating]:
    """Test if a collection comes from a normal distribution using the Shapiro-Wilk test."""
    return stats.shapiro(collection)


@basic_stats_mcp.tool()
def calculate_bootstrap_ci(collection: List[Union[float, int]], confidence: float = 0.95, n_resamples: int = 1000) -> Tuple[floating, floating]:
    """Calculate bootstrap confidence interval for the mean of a collection."""
    if confidence <= 0 or confidence >= 1:
        raise ValueError("Confidence must be between 0 and 1")

    data = np.array(collection)
    n = len(data)
    means = np.zeros(n_resamples)

    for i in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)

    return lower_bound, upper_bound


@basic_stats_mcp.tool()
def calculate_trimmed_mean(collection: List[Union[float, int]], proportion: float = 0.1) -> floating:
    """Calculate the trimmed mean of a collection by removing extreme values."""
    if proportion < 0 or proportion >= 0.5:
        raise ValueError("Proportion must be between 0 and 0.5")
    return stats.trim_mean(collection, proportion)


@basic_stats_mcp.tool()
def create_frequency_table(collection: List[Union[float, int]]) -> dict:
    """Create a frequency table from a collection."""
    unique_values, counts = np.unique(collection, return_counts=True)
    return dict(zip(unique_values, counts))


@basic_stats_mcp.tool()
def perform_mann_whitney_test(sample1: List[Union[float, int]], sample2: List[Union[float, int]]) -> Tuple[floating, floating]:
    """Perform the Mann-Whitney U test for two independent samples."""
    return stats.mannwhitneyu(sample1, sample2)


@basic_stats_mcp.tool()
def perform_wilcoxon_test(sample1: List[Union[float, int]], sample2: List[Union[float, int]] = None) -> Tuple[floating, floating]:
    """Perform the Wilcoxon signed-rank test for two related samples or a single sample against zero."""
    if sample2 is None:
        return stats.wilcoxon(sample1)

    if len(sample1) != len(sample2):
        raise ValueError("Samples must be of the same length for paired test")

    return stats.wilcoxon(sample1, sample2)


@basic_stats_mcp.tool()
def generate_descriptive_statistics(collection: List[Union[float, int]]) -> dict:
    """Generate a comprehensive summary of descriptive statistics for a collection."""
    data = np.array(collection)
    q1, q2, q3 = np.percentile(data, [25, 50, 75])

    return {
        "count": len(collection),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "range": float(np.max(data) - np.min(data)),
        "variance": float(np.var(data)),
        "std_dev": float(np.std(data)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data))
    }


@basic_stats_mcp.tool()
def calculate_coefficient_of_variation(collection: List[Union[float, int]]) -> floating:
    """Calculate the coefficient of variation (relative standard deviation) of a collection."""
    data = np.array(collection)
    mean = np.mean(data)

    if mean == 0:
        raise ValueError("Cannot calculate coefficient of variation when mean is zero")

    std_dev = np.std(data)
    return std_dev / abs(mean)


@basic_stats_mcp.tool()
def perform_binomial_test(successes: int, trials: int, p: float = 0.5) -> floating:
    """Perform a binomial test to test the null hypothesis that the probability of success is p."""
    if not 0 <= p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    if successes > trials:
        raise ValueError("Number of successes cannot exceed number of trials")

    result = stats.binomtest(successes, trials, p)
    return result.pvalue


@basic_stats_mcp.tool()
def calculate_quantiles(collection: List[Union[float, int]], q: List[float]) -> List[floating]:
    """Calculate quantiles of a collection."""
    if any(not 0 <= x <= 1 for x in q):
        raise ValueError("Quantiles must be between 0 and 1")

    return list(np.quantile(collection, q))

