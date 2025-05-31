namespace calculator.distance
{
    public class DistanceCalculatorEuclidean : IDistanceCalculator
    {

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
            double diffSum = 0;
            for (int i = 0; i < vector1.Length; i++)
            {
                var diff = vector1[i] - vector2[i];
                diffSum += diff * diff;
            }

            return Math.Sqrt(diffSum);
        }
    }
}
