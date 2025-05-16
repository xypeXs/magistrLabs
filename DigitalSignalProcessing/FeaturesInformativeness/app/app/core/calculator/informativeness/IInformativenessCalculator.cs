using informativeness.app.core.calculator.distance;
using informativeness.app.core.data;

namespace informativeness.app.core.calculator.informativeness
{
    public interface IInformativenessCalculator
    {
        public InformativenessCalculationResult Calculate(FeaturesData data, IDistanceCalculator distanceCalculator);
    }
}
