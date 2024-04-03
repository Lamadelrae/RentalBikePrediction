using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using RentalBikePrediction.Models;
using System.Data;
using System.Data.SqlClient;

namespace RentalBikePrediction.MachineLearning;

public class ForecastingService(IConfiguration configuration)
{
    private readonly MLContext _context = new();
    private readonly string _connection = configuration.GetConnectionString("BikeRental") ?? string.Empty;

    //This method loads data from the database.
    private IDataView GetAllData()
    {
        var query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

        var loader = _context.Data.CreateDatabaseLoader<Rental.Input>();
        var source = new DatabaseSource(SqlClientFactory.Instance, _connection, query);

        return loader.Load(source);
    }

    //Trains and gets transformer
    private ITransformer TrainWith(IDataView data)
    {
        var forecastingPipeline = _context.Forecasting.ForecastBySsa(
            inputColumnName: "TotalRentals",
            windowSize: 7,
            seriesLength: 30,
            trainSize: 365,
            horizon: 7,
            confidenceLevel: 0.95f,
            outputColumnName: "ForecastedRentals",
            confidenceLowerBoundColumn: "LowerBoundRentals",
            confidenceUpperBoundColumn: "UpperBoundRentals");

        return forecastingPipeline.Fit(data);
    }

    private TimeSeriesPredictionEngine<Rental.Input, Rental.Output> CreateEngineWith(ITransformer transformer)
    {
        return transformer.CreateTimeSeriesEngine<Rental.Input, Rental.Output>(_context);
    }

    void Evaluate(IDataView testData, ITransformer model)
    {
        var engine = CreateEngineWith(model);
        var a = engine.Predict();

        // Make predictions
        IDataView predictions = model.Transform(testData);

        // Actual values
        IEnumerable<float> actual =
            _context.Data.CreateEnumerable<Rental.Input>(testData, true)
                .Select(observed => observed.TotalRentals);

        // Predicted values
        IEnumerable<float> forecast =
            _context.Data.CreateEnumerable<Rental.Output>(predictions, true)
                .Select(prediction => prediction.ForecastedRentals[0]);

        // Calculate error (actual - forecast)
        var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

        // Get metric averages
        var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
        var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

        var anotherEngine = CreateEngineWith(model);
        var b = anotherEngine.Predict();

        // Output metrics
        //Console.WriteLine("Evaluation Metrics");
        //Console.WriteLine("---------------------");
        //Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
        //Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
    }

    public Rental.Output Forecast()
    {
        var data = GetAllData();
        var firstYearData = _context.Data.FilterRowsByColumn(data, "Year", upperBound: 1);
        var secondYearData = _context.Data.FilterRowsByColumn(data, "Year", lowerBound: 1);

        var transformer = TrainWith(firstYearData);
        Evaluate(secondYearData, transformer);
        var engine = CreateEngineWith(transformer);

        return engine.Predict();
    }
}
