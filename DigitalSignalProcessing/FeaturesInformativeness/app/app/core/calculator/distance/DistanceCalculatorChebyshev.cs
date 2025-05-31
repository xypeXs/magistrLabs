using System;
using System.Collections.Generic;

using app.core.constant;
using app.core.data;

namespace app.core.calculator.distance
{
    public class DistanceCalculatorChebyshev : IDistanceCalculator
    {

        public string GetType()
        {
            return Constant.CHEBYSHEV;
        }

        public void initialize(FeaturesData featuresData) { }

        public double Calculate(double x1, double x2)
        {
            return Calculate(new double[] { x1 }, new double[] { x2 });
        }

        public double Calculate(List<double> vecotr1, List<double> vector2)
        {
            return Calculate(vecotr1.ToArray(), vector2.ToArray());
        }

        public double Calculate(double[] vector1, double[] vector2)
        {
            double maxAbsDiff = -1;
            for (int i = 0; i < vector1.Length; i++)
            {
                double diff = Math.Abs(vector1[i] - vector2[i]);
                if (diff > maxAbsDiff)
                    maxAbsDiff = diff;
            }

            return maxAbsDiff;
        }
    }
}
