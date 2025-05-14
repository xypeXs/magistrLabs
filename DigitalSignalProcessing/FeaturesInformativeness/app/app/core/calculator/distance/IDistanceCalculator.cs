using core.data;
using System.Collections.Generic;

namespace core.calculator.distance
{
    public interface IDistanceCalculator
    {
        public string GetType();
        public void initialize(FeaturesData featuresData);
        public double Calculate(double x1, double x2);
        public double Calculate(double[] vecotr1, double[] vector2);
        public double Calculate(List<double> vecotr1, List<double> vector2);
    }
}
