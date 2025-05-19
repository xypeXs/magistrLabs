using app.core.calculator.distance;
using app.core.data;
using System;
using System.Linq;

namespace app.core.calculator.informativeness
{
    public class InformativenessCalculatorDefault : IInformativenessCalculator
    {
        private double EPS = 0.0001;

        public InformativenessCalculationResult Calculate(FeaturesData data, IDistanceCalculator distanceCalculator)
        {
            int numfeatures = getNumfeatures(data);
            int numclasses = getNumclasses(data);
            double[,] intraClassDistancesByFeature = CalculateIntraClassDistances(data, distanceCalculator);
            double[,,] interClassDistancesByFeature = CalculateInterClassDistances(data, distanceCalculator);

            if (numclasses == 2)
            {
                return calculateTwoClassesInformativeness(intraClassDistancesByFeature, interClassDistancesByFeature, numfeatures, numclasses);
            }

            return calculateManyClassesInformativeness(intraClassDistancesByFeature, interClassDistancesByFeature, numfeatures, numclasses);
        }

        
        // Расчёт для двух классов
        private InformativenessCalculationResult calculateTwoClassesInformativeness(double[,] intraClassDistancesByFeature, double[,,] interClassDistancesByFeature, int numfeatures, int numclasses)
        {
            double[] informativenessByFeature = new double[numfeatures];

            for (int featureind = 0; featureind < numfeatures; featureind++)
            {
                double intraClassDistanceSum = 0;
                for (int i = 0; i < intraClassDistancesByFeature.GetLength(1); i++)
                {
                    intraClassDistanceSum += intraClassDistancesByFeature[featureind, i];
                }

                double interClassDistanceSum = 0;
                for (int i = 0; i < interClassDistancesByFeature.GetLength(1); i++)
                {
                    for (int j = i + 1; j < interClassDistancesByFeature.GetLength(2); j++)
                    {
                        interClassDistanceSum += interClassDistancesByFeature[featureind, i, j];
                    }
                }
                if (Math.Abs(intraClassDistanceSum * (numclasses - 1)) < EPS)
                    informativenessByFeature[featureind] = 0;
                else
                    informativenessByFeature[featureind] = interClassDistanceSum / (intraClassDistanceSum * (numclasses - 1));
            }

            return new InformativenessCalculationResult
            {
                informativenessList = informativenessByFeature.ToList()
            };
        }

        // Расчёт для нескольких классов (>2)
        private InformativenessCalculationResult calculateManyClassesInformativeness(double[,] intraClassDistancesByFeature, double[,,] interClassDistancesByFeature, int numfeatures, int numclasses)
        {
            double[] informativenessByFeature = new double[numfeatures];

            for (int featureind = 0; featureind < numfeatures; featureind++)
            {

                double mininformativeness = double.PositiveInfinity;

                for (int i = 0; i < interClassDistancesByFeature.GetLength(1); i++)
                {
                    for (int j = i + 1; j < interClassDistancesByFeature.GetLength(2); j++)
                    {
                        double curinformativeness;
                        double intraSum = intraClassDistancesByFeature[featureind, i] + intraClassDistancesByFeature[featureind, j];
                        if (Math.Abs(intraSum) < EPS)
                            curinformativeness = double.PositiveInfinity;
                        else
                            curinformativeness = interClassDistancesByFeature[featureind, i, j] / intraSum;

                        mininformativeness = Math.Min(curinformativeness, mininformativeness);
                    }
                }
                if (mininformativeness == double.PositiveInfinity)
                    mininformativeness = 0;

                informativenessByFeature[featureind] = mininformativeness;
            }

            return new InformativenessCalculationResult
            {
                informativenessList = informativenessByFeature.ToList()
            };
        }

        // Расчет внутриклассовых расстояний
        // Первое измерение - индекс признака
        // Второе измерение - внутриклассовое расстояние
        public double[,] CalculateIntraClassDistances(FeaturesData data, IDistanceCalculator distanceCalculator)
        {
            var groupedByClass = data.imageList.GroupBy(d => d.classIndex);
            var numclasses = data.imageList.Select(d => d.classIndex).Distinct().Count();
            var minclassindex = getMinclassindex(data);
            var numfeatures = getNumfeatures(data);
            var intraDistances = new double[numfeatures, numclasses];

            foreach (var group in groupedByClass)
            {
                var classData = group.ToList();

                // Считаем расстояние по каждому признаку отдельно
                for (int featureind = 0; featureind < numfeatures; featureind++)
                {
                    double sum = 0;
                    for (int k = 0; k < classData.Count; k++)
                    {
                        for (int l = k + 1; l < classData.Count; l++)
                        {
                            sum += distanceCalculator.Calculate(classData[k].featureList[featureind], classData[l].featureList[featureind]);
                        }
                    }

                    // Предполагается, что классы индексируются с шагом 1
                    intraDistances[featureind, int.Parse(group.Key.ToString()) - minclassindex] = (sum * 2) / (classData.Count * (classData.Count - 1));
                }
            }

            return intraDistances;
        }

        // Расчет межклассовых расстояний
        // Первое измерение - индекс признака
        // Второе, третье измерение - расстояние между классами i и j
        public double[,,] CalculateInterClassDistances(FeaturesData data, IDistanceCalculator distanceCalculator)
        {
            var classes = data.imageList.Select(d => d.classIndex).Distinct().ToList();
            var minclassindex = getMinclassindex(data);
            var numfeatures = getNumfeatures(data);
            var interDistances = new double[numfeatures, classes.Count, classes.Count];
            int totalPairs = classes.Count * (classes.Count - 1) / 2;

            for (int i = 0; i < classes.Count; i++)
            {
                for (int j = i + 1; j < classes.Count; j++)
                {
                    var class1 = data.imageList.Where(d => d.classIndex == classes[i]).ToList();
                    var class2 = data.imageList.Where(d => d.classIndex == classes[j]).ToList();

                    for (int featureind = 0; featureind < numfeatures; featureind++)
                    {
                        double sum = 0;
                        for (int k = 0; k < class1.Count; k++)
                        {
                            for (int l = 0; l < class2.Count; l++)
                            {
                                sum += distanceCalculator.Calculate(class1[k].featureList[featureind], class2[l].featureList[featureind]);
                            }
                        }
                        // Предполагается, что классы индексируются с шагом 1
                        interDistances[featureind, classes[i] - minclassindex, classes[j] - minclassindex] = sum / (class1.Count * class2.Count);
                        interDistances[featureind, classes[j] - minclassindex, classes[i] - minclassindex] = interDistances[featureind, classes[i] - minclassindex, classes[j] - minclassindex];
                    }
                }
            }

            return interDistances;
        }

        private int getNumfeatures(FeaturesData data)
        {
            if (data == null || data.imageList == null || data.imageList.Count == 0) return 0;
            return data.imageList[0].featureList.Count;
        }

        private int getNumclasses(FeaturesData data)
        {
            if (data == null || data.imageList == null || data.imageList.Count == 0) return 0;
            return data.imageList.Select(d => d.classIndex).Distinct().ToList().Count();
        }

        private int getMinclassindex(FeaturesData data)
        {
            if (data == null || data.imageList == null || data.imageList.Count == 0) return 0;
            return data.imageList.Select(d => d.classIndex).Min();
        }
    }
}
