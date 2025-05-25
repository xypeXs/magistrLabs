using System.Collections.Generic;

namespace app.core.data
{
    public class InformativenessCalculationResult
    {
        public FeaturesData featureData {  get; set; }
        public List<double> informativenessList { get; set; }
        public string metricName { get; set; }
    }
}
