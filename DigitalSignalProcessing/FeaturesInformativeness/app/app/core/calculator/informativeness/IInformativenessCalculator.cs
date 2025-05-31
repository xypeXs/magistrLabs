using app.core.calculator.distance;
using app.core.data;

namespace app.core.calculator.informativeness
{
    public interface IInformativenessCalculator
    {
        public InformativenessCalculationResult Calculate(FeaturesData data, IDistanceCalculator distanceCalculator);
    }
}
