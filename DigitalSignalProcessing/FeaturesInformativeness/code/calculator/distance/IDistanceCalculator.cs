using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace calculator.distance
{
    public interface IDistanceCalculator
    {
        public double Calculate(double x1, double x2);
        public double Calculate(double[] vecotr1, double[] vector2);
        public double Calculate(List<double> vecotr1, List<double> vector2);
    }
}
