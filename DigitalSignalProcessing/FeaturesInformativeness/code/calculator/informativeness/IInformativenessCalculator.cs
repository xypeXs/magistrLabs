using calculator.distance;
using code.data;
using data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace calculator.informativeness
{
    public interface IInformativenessCalculator
    {
        public InformativenessCalculationResult Calculate(FeaturesData data, IDistanceCalculator distanceCalculator);
    }
}
