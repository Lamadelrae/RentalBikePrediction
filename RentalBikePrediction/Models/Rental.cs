namespace RentalBikePrediction.Models;

public class Rental
{
    public class Input
    {
        public DateTime RentalDate { get; set; }
        public float Year { get; set; }
        public float TotalRentals { get; set; }
    }

    public class Output
    {
        public float[] ForecastedRentals { get; set; }
        public float[] LowerBoundRentals { get; set; }
        public float[] UpperBoundRentals { get; set; }
    }
}
