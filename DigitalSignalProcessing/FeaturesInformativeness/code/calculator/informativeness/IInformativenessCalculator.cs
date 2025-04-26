using calculator.distance;
using code.data;
using data;

namespace calculator.informativeness
{
    public interface IInformativenessCalculator
    {
        public InformativenessCalculationResult Calculate(FeaturesData data, IDistanceCalculator distanceCalculator);
    }
}
