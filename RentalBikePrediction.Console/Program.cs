﻿using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Data;
using System.Data.SqlClient;

string modelPath = "Model.zip";
var connectionString = $"Server=;Database=BikeRental;User Id=sa;Password=pass;";


MLContext mlContext = new MLContext();

DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();

string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, query);

IDataView dataView = loader.Load(dbSource);

IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedRentals",
    inputColumnName: "TotalRentals",
    windowSize: 7,
    seriesLength: 30,
    trainSize: 365,
    horizon: 7,
    confidenceLevel: 0.95f,
    confidenceLowerBoundColumn: "LowerBoundRentals",
    confidenceUpperBoundColumn: "UpperBoundRentals");

SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);

Evaluate(secondYearData, forecaster, mlContext);

var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
forecastEngine.CheckPoint(mlContext, modelPath);

Forecast(secondYearData, 7, forecastEngine, mlContext);

void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
{
    // Make predictions
    IDataView predictions = model.Transform(testData);

    // Actual values
    IEnumerable<float> actual =
        mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
            .Select(observed => observed.TotalRentals);

    // Predicted values
    IEnumerable<float> forecast =
        mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
            .Select(prediction => prediction.ForecastedRentals[0]);

    // Calculate error (actual - forecast)
    var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

    // Get metric averages
    var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
    var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

    // Output metrics
    Console.WriteLine("Evaluation Metrics");
    Console.WriteLine("---------------------");
    Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
    Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
}

void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
{

    ModelOutput forecast = forecaster.Predict();

    IEnumerable<string> forecastOutput =
        mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
            .Take(horizon)
            .Select((ModelInput rental, int index) =>
            {
                string rentalDate = rental.RentalDate.ToShortDateString();
                float actualRentals = rental.TotalRentals;
                float lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
                float estimate = forecast.ForecastedRentals[index];
                float upperEstimate = forecast.UpperBoundRentals[index];
                return $"Date: {rentalDate}\n" +
                $"Actual Rentals: {actualRentals}\n" +
                $"Lower Estimate: {lowerEstimate}\n" +
                $"Forecast: {estimate}\n" +
                $"Upper Estimate: {upperEstimate}\n";
            });

    // Output predictions
    Console.WriteLine("Rental Forecast");
    Console.WriteLine("---------------------");
    foreach (var prediction in forecastOutput)
    {
        Console.WriteLine(prediction);
    }
}

public class ModelInput
{
    public DateTime RentalDate { get; set; }

    public float Year { get; set; }

    public float TotalRentals { get; set; }
}

public class ModelOutput
{
    public float[] ForecastedRentals { get; set; }

    public float[] LowerBoundRentals { get; set; }

    public float[] UpperBoundRentals { get; set; }
}