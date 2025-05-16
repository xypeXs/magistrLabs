using informativeness.app.core.calculator.distance;
using informativeness.app.core.data;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double; // DenseMatrix, DenseVector
using System;
using System.Collections.Generic;
using System.Linq;

using informativeness.app.core.constant;

namespace informativeness.app.core.calculator.distance
{

    public class DistanceCalculatorMahalanobis : IDistanceCalculator
    {
        private Matrix<double> _invCovMatrix;
        private int _dimension;
        private double[] _stdDevs;
        private int _currentFeature = 0;

        public Matrix<double> InvCovMatrix => _invCovMatrix;

        public string GetType()
        {
            return Constant.MAHALANOBIS;
        }

        public void initialize(FeaturesData data)
        {
            if (data == null || data.imageList == null || data.imageList.Count == 0)
                throw new ArgumentException("Нет данных для вычисления ковариации.");

            var vectors = data.imageList.Select(img => DenseVector.OfEnumerable(img.featureList)).ToList();
            _dimension = vectors[0].Count;

            if (vectors.Any(v => v.Count != _dimension))
                throw new ArgumentException("Все векторы должны иметь одинаковую длину.");

            var covMatrix = TwoPassCovariance(vectors);

            // covMatrix должна быть положительно определенной и хорошо обусловлена
            // при последовательности схожих признаков [Ew0, ..., EwN] они сильно коррелируют
            var regularization = 1e-5;
            for (int i = 0; i < covMatrix.RowCount; i++)
            {
                covMatrix[i, i] += regularization;
            }

            _invCovMatrix = covMatrix.Inverse();

            _stdDevs = new double[_dimension];
            for (int i = 0; i < _dimension; i++)
            {
                _stdDevs[i] = Math.Sqrt(covMatrix[i, i]);
            }
        }

        public DistanceCalculatorMahalanobis() {}

        private Matrix<double> TwoPassCovariance(List<DenseVector> vectors)
        {
            int n = vectors.Count;
            int dim = vectors[0].Count;
            var means = new double[dim];

            for (int i = 0; i < dim; i++)
                means[i] = vectors.Average(v => v[i]);

            var covariance = DenseMatrix.Create(dim, dim, 0.0);

            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        double a = vectors[k][i] - means[i];
                        double b = vectors[k][j] - means[j];
                        sum += a * b;
                    }
                    covariance[i, j] = sum / (n - 1);
                }
            }

            return covariance;
        }

        public double Calculate(double x1, double x2)
        {
            if (_currentFeature >= _stdDevs.Length)
                _currentFeature = 0;

            double diff = x1 - x2;
            double scale = _stdDevs[_currentFeature];
            _currentFeature++;

            if (scale < 1e-12)
                return 0.0;

            return Math.Abs(diff) / scale;
        }

        public double Calculate(List<double> v1, List<double> v2)
        {
            if (v1.Count != v2.Count)
                throw new ArgumentException("Векторы должны иметь одинаковую длину.");
            return Calculate(v1.ToArray(), v2.ToArray());
        }

        public double Calculate(double[] v1, double[] v2)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("Векторы должны иметь одинаковую длину.");

            if (v1.Length != _dimension)
                throw new ArgumentException("Размерность векторов не соответствует обучающим данным.");

            var diff = DenseVector.OfArray(v1).Subtract(DenseVector.OfArray(v2));
            var temp = _invCovMatrix.Multiply(diff);
            var mahalanobisSquared = diff.DotProduct(temp);

            return Math.Sqrt(mahalanobisSquared);
        }
    }
}
