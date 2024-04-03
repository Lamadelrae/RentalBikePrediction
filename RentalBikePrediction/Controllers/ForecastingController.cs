using Microsoft.AspNetCore.Mvc;
using RentalBikePrediction.MachineLearning;

namespace RentalBikePrediction.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ForecastingController(ForecastingService forecaster) : ControllerBase
    {
        private readonly ForecastingService _forecaster = forecaster;

        [HttpGet]
        public IEnumerable<float> Get()
        {
            var @return = _forecaster.Forecast();

            return @return.ForecastedRentals;
        }
    }
}
