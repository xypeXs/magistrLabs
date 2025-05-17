using app.core.constant;
using app.core.data;
using System;
using System.Collections.Generic;

namespace app.core.calculator.distance
{
	public class DistanceCalculatorCanberra : IDistanceCalculator
	{
        public string GetType()
        {
            return Constant.CANBERRA;
        }

        public void initialize(FeaturesData featuresData) { }

        public double Calculate(double x1, double x2)
		{
			return Calculate(new double[] { x1 }, new double[] { x2 });
		}

		public double Calculate(List<double> vector1, List<double> vector2)
		{
			return Calculate(vector1.ToArray(), vector2.ToArray());
		}

		public double Calculate(double[] vector1, double[] vector2)
		{
			double sum = 0;
			for (int i = 0; i < vector1.Length; i++)
			{
				double absDiff = Math.Abs(vector1[i] - vector2[i]);
				double denom = Math.Abs(vector1[i]) + Math.Abs(vector2[i]);
				if (denom > 0) {
					sum += absDiff / denom;
				}
				// если сумма модулей двух координат = 0, вклад этой пары в расстояние равен 0
			}
			return sum;
		}
    }
}
