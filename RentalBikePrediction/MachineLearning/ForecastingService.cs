using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using RentalBikePrediction.Models;
using System.Data.SqlClient;

namespace RentalBikePrediction.MachineLearning;

public class ForecastingService
{
    private MLContext _context = new();
    private string _connection;

    public ForecastingService(IConfiguration configuration)
    {
        _connection = configuration.GetConnectionString("BikeRental") ?? string.Empty;
    }

    //This method loads data from the database.
    private IDataView GetDataViewWithAllData()
    {
        var query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

        var loader = _context.Data.CreateDatabaseLoader<Rental>();
        var source = new DatabaseSource(SqlClientFactory.Instance, _connection, query);

        return loader.Load(source);
    }

    private SsaForecastingTransformer TrainTransformer(IDataView data)
    {
        var forecastingPipeline = _context.Forecasting.ForecastBySsa(
            outputColumnName: "ForecastedRentals",
            inputColumnName: "TotalRentals",
            windowSize: 7,
            seriesLength: 30,
            trainSize: 365,
            horizon: 7,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: "LowerBoundRentals",
            confidenceUpperBoundColumn: "UpperBoundRentals");

        return forecastingPipeline.Fit(data);
    }
}
