using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using data;
using loader;
//using ScottPlot;

public class InformativenessCalculator
{

    // Расчет внутриклассовых расстояний
    public double[] calculateIntraClassDistances(FeaturesData data, int numFeatures)
    {
        var intraDistances = new double[numFeatures];
        var groupedByClass = data.imageList.GroupBy(d => d.classIndex);

        foreach (var group in groupedByClass)
        {
            var classData = group.ToList();
            for (int i = 0; i < numFeatures; i++)
            {
                double sum = 0;
                int count = 0;
                for (int j = 0; j < classData.Count; j++)
                {
                    for (int k = j + 1; k < classData.Count; k++)
                    {
                        sum += Math.Abs(classData[j].featureList[i] - classData[k].featureList[i]);
                        count++;
                    }
                }
                if (count > 0) intraDistances[i] += sum / count;
            }
        }

        // Усреднение по классам
        for (int i = 0; i < numFeatures; i++)
            intraDistances[i] /= groupedByClass.Count();

        return intraDistances;
    }

    // Расчет межклассовых расстояний
    public double[] calculateInterClassDistances(FeaturesData data, int numFeatures)
    {
        var interDistances = new double[numFeatures];
        var classes = data.imageList.Select(d => d.classIndex).Distinct().ToList();
        int totalPairs = classes.Count * (classes.Count - 1) / 2;

        for (int i = 0; i < classes.Count; i++)
        {
            for (int j = i + 1; j < classes.Count; j++)
            {
                var class1 = data.imageList.Where(d => d.classIndex == classes[i]).ToList();
                var class2 = data.imageList.Where(d => d.classIndex == classes[j]).ToList();

                for (int f = 0; f < numFeatures; f++)
                {
                    double sum = 0;
                    int count = 0;
                    foreach (var d1 in class1)
                        foreach (var d2 in class2)
                        {
                            sum += Math.Abs(d1.featureList[f] - d2.featureList[f]);
                            count++;
                        }
                    if (count > 0) interDistances[f] += sum / count;
                }
            }
        }

        // Усреднение по парам классов
        for (int f = 0; f < numFeatures; f++)
            interDistances[f] /= totalPairs;

        return interDistances;
    }

    // Расчет информативности
    public double[] calculateInformativeness(double[] inter, double[] intra)
    {
        var info = new double[inter.Length];
        for (int i = 0; i < inter.Length; i++)
            info[i] = (intra[i] == 0) ? double.PositiveInfinity : inter[i] / intra[i];
        return info;
    }

    // Вывод таблицы
    public void printTable(double[] informativeness)
    {
        Console.WriteLine("Номер\tИнформативность");
        for (int i = 0; i < informativeness.Length; i++)
            Console.WriteLine($"{i + 1}\t{informativeness[i]:F2}");
    }

    //// Построение графика
    //public void PlotResults(double[] informativeness, string outputPath = "informativeness.png")
    //{
    //    var plt = new Plot(800, 400);
    //    plt.Title("Информативность признаков");
    //    plt.XLabel("Номер признака");
    //    plt.YLabel("Коэффициент информативности");

    //    double[] positions = Enumerable.Range(1, informativeness.Length).Select(x => (double)x).ToArray();
    //    plt.AddBar(informativeness, positions);
    //    plt.SaveFig(outputPath);
    //}
}

// Пример использования
class Program
{
    static void Main()
    {
        IDataLoader dataLoader = new DataLoadTxt();

        var calculator = new InformativenessCalculator();
        var data = dataLoader.loadData("./resources/Th.txt");
        int numFeatures = data.imageList[0].featureList.Count;

        var intra = calculator.calculateIntraClassDistances(data, numFeatures);
        var inter = calculator.calculateInterClassDistances(data, numFeatures);
        var info = calculator.calculateInformativeness(inter, intra);

        calculator.printTable(info);
        //calculator.PlotResults(info);
    }
}