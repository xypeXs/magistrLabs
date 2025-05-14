using core.calculator.distance;
using core.data;

namespace core.calculator.informativeness
{
    public interface IInformativenessCalculator
    {
        public InformativenessCalculationResult Calculate(FeaturesData data, IDistanceCalculator distanceCalculator);
    }
}
